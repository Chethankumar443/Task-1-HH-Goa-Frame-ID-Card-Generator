"""
Sub-50ms Hybrid Retrieval Engine (FAISS + BM25 + Weighted Fusion)
================================================================

This module implements an ultra-fast, in-memory hybrid retrieval engine that combines:
1. Dense Retrieval: In-memory FAISS (IndexFlatIP for exact cosine similarity or IndexIVFFlat)
2. Sparse Retrieval: rank-bm25 (BM25Okapi for keyword matching)
3. Fusion Engine: Reciprocal Rank Fusion (RRF) & 70/30 Weighted Linear Combination

Strict Latency SLA: Sub-50ms (P99) for end-to-end hybrid search queries.
Author: Senior Backend / RAG Engineer
Date: 2026-08-21
"""

import gc
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Union

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

# FastEmbed ONNX loader (40MB RAM vs 500MB PyTorch)
try:
    from fastembed import TextEmbedding
    HAS_FASTEMBED = True
except ImportError:
    HAS_FASTEMBED = False
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_grad_enabled(False)
        torch.set_num_threads(1)
    except ImportError:
        pass

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def simple_tokenize(text: str) -> List[str]:
    """
    Fast lowercasing word tokenizer for BM25 indexing.
    """
    return re.findall(r"\w+", text.lower())


class HybridRetriever:
    """
    Production-grade in-memory Hybrid Retriever using FAISS and BM25.
    
    Guarantees <50ms query latency by maintaining all indexes in RAM.
    Uses ultra-lightweight FastEmbed ONNX (40MB RAM) for low-resource environments.
    """

    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: Optional[np.ndarray] = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        use_ivf: bool = False,
        nlist: int = 100,
        nprobe: int = 10,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize and build in-memory FAISS and BM25 indexes.
        """
        if not chunks:
            raise ValueError("chunks list cannot be empty")

        self.chunks = chunks
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.nprobe = nprobe

        logger.info(f"Initializing HybridRetriever with {len(chunks)} chunks...")

        # 1. Initialize Lightweight FastEmbed ONNX or SentenceTransformer model
        if HAS_FASTEMBED:
            logger.info("Using FastEmbed ONNX runtime (ultra-low ~40MB memory footprint)...")
            self.model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
            self.embedding_dim = 384
        else:
            logger.info(f"Loading SentenceTransformer model '{model_name}'...")
            self.model = SentenceTransformer(model_name, device=device or "cpu")
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # 2. Prepare or compute dense embeddings
        if embeddings is None:
            logger.info("Computing dense embeddings for chunks...")
            texts = [c["text"] for c in self.chunks]
            embeddings = self.encode(texts, normalize=True)
        else:
            embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            embeddings = embeddings / norms

        self.embeddings = embeddings.astype(np.float32)

        # 3. Build in-memory FAISS Index (IndexFlatIP or IndexIVFFlat)
        logger.info("Building in-memory FAISS index...")
        if use_ivf and len(self.chunks) >= nlist:
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self.faiss_index.train(self.embeddings)
            self.faiss_index.add(self.embeddings)
            self.faiss_index.nprobe = self.nprobe
        else:
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            self.faiss_index.add(self.embeddings)

        # 4. Build in-memory BM25 Index using rank-bm25
        logger.info("Building in-memory BM25 index...")
        tokenized_corpus = [simple_tokenize(c["text"]) for c in self.chunks]
        self.bm25_index = BM25Okapi(tokenized_corpus)

        gc.collect()
        logger.info("HybridRetriever ready and fully loaded into RAM.")

    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """
        Encode text or list of texts into normalized float32 embeddings.
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if hasattr(self.model, "embed"):
            embs = np.array(list(self.model.embed(texts)), dtype=np.float32)
        else:
            embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

        if normalize:
            norms = np.linalg.norm(embs, axis=-1, keepdims=True)
            norms[norms == 0] = 1e-10
            embs = embs / norms

        return embs

    def search(
        self,
        query: str,
        top_k: int = 5,
        fusion_method: str = "weighted_linear",
        rrf_k: int = 60,
        query_vec: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute ultra-fast sub-50ms hybrid retrieval for a search query.

        Args:
            query: Natural language search query.
            top_k: Number of top documents to return.
            fusion_method: 'weighted_linear' (70/30) or 'rrf' (Reciprocal Rank Fusion).
            rrf_k: Smoothing constant for RRF fusion.
            query_vec: Optional pre-computed normalized 384d numpy array to eliminate redundant encoding.
        """
        if not query or not query.strip():
            return []

        fetch_k = min(len(self.chunks), max(top_k * 10, 100))

        # --- A. Dense Retrieval (FAISS) ---
        if query_vec is None:
            query_vec = self.encode(query, normalize=True)
        elif query_vec.ndim == 1:
            query_vec = np.expand_dims(query_vec, axis=0).astype(np.float32)

        dense_distances, dense_indices = self.faiss_index.search(query_vec, fetch_k)
        dense_distances = dense_distances[0]
        dense_indices = dense_indices[0]

        # --- B. Sparse Retrieval (BM25) ---
        tokenized_query = simple_tokenize(query)
        bm25_scores = self.bm25_index.get_scores(tokenized_query)

        # --- C. Score Fusion & Normalization ---
        if fusion_method == "rrf":
            # Reciprocal Rank Fusion
            score_map: Dict[int, float] = {}

            # Dense RRF ranks
            for rank, idx in enumerate(dense_indices):
                if idx < 0:
                    continue
                score_map[idx] = score_map.get(idx, 0.0) + self.dense_weight * (1.0 / (rrf_k + rank + 1))

            # Sparse RRF ranks
            bm25_top_indices = np.argsort(bm25_scores)[::-1][:fetch_k]
            for rank, idx in enumerate(bm25_top_indices):
                score_map[idx] = score_map.get(idx, 0.0) + self.sparse_weight * (1.0 / (rrf_k + rank + 1))

            sorted_indices = sorted(score_map.keys(), key=lambda i: score_map[i], reverse=True)[:top_k]
            
            # Normalize RRF scores to [0.0, 1.0]
            max_score = max(score_map.values()) if score_map else 1.0
            min_score = min(score_map.values()) if score_map else 0.0
            score_range = (max_score - min_score) if max_score > min_score else 1.0

            results = []
            for idx in sorted_indices:
                raw_score = score_map[idx]
                norm_score = (raw_score - min_score) / score_range if score_range > 0 else 1.0
                chunk = self.chunks[idx]
                results.append({
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "metadata": chunk.get("metadata", {}),
                    "relevance_score": round(float(norm_score), 4),
                })
            return results

        else:
            # Weighted Linear Combination (70% Dense + 30% Sparse)
            # Normalize Dense scores (Cosine similarity in [-1, 1] mapped to [0, 1])
            dense_map: Dict[int, float] = {}
            if len(dense_distances) > 0:
                d_min, d_max = float(np.min(dense_distances)), float(np.max(dense_distances))
                d_range = (d_max - d_min) if d_max > d_min else 1.0
                for dist, idx in zip(dense_distances, dense_indices):
                    if idx >= 0:
                        dense_map[idx] = (dist - d_min) / d_range if d_range > 0 else 1.0

            # Normalize BM25 scores to [0, 1]
            b_min, b_max = float(np.min(bm25_scores)), float(np.max(bm25_scores))
            b_range = (b_max - b_min) if b_max > b_min else 1.0

            candidate_indices = set(dense_map.keys()).union(
                np.argsort(bm25_scores)[::-1][:fetch_k]
            )

            fused_scores: List[Tuple[int, float]] = []
            for idx in candidate_indices:
                idx = int(idx)
                d_score = dense_map.get(idx, 0.0)
                b_score = (bm25_scores[idx] - b_min) / b_range if b_range > 0 else 0.0
                combined_score = (self.dense_weight * d_score) + (self.sparse_weight * b_score)
                fused_scores.append((idx, combined_score))

            # Sort by fused score descending
            fused_scores.sort(key=lambda x: x[1], reverse=True)
            top_fused = fused_scores[:top_k]

            # Min-Max normalize final top relevance scores to [0.0, 1.0]
            if top_fused:
                max_f = top_fused[0][1]
                min_f = top_fused[-1][1] if len(top_fused) > 1 else 0.0
                f_range = (max_f - min_f) if max_f > min_f else 1.0
            else:
                f_range = 1.0
                min_f = 0.0

            results = []
            for idx, raw_score in top_fused:
                norm_score = (raw_score - min_f) / f_range if f_range > 0 else 1.0
                # Clamp strictly to [0.0, 1.0]
                norm_score = max(0.0, min(1.0, norm_score))
                chunk = self.chunks[idx]
                results.append({
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "metadata": chunk.get("metadata", {}),
                    "relevance_score": round(float(norm_score), 4),
                })

            return results


def benchmark_retriever(
    retriever: HybridRetriever,
    num_queries: int = 100,
    target_ms: float = 50.0,
) -> Dict[str, float]:
    """
    Benchmark function to prove sub-50ms search latency across 100 queries.

    Args:
        retriever: Initialized HybridRetriever instance.
        num_queries: Number of random benchmark queries to run.
        target_ms: Maximum allowed threshold for P99 latency SLA.

    Returns:
        Dictionary of latency statistics (mean, p50, p90, p99, max).
    """
    logger.info(f"Running benchmark with {num_queries} test queries...")

    sample_queries = [
        "What is the capital of India?",
        "Explain hybrid chunking strategies for RAG",
        "How to optimize vector search latency below 50ms",
        "MS MARCO dataset retrieval benchmarks",
        "Sentence transformers embedding dimensions",
        "Fast API endpoint latency targets for voice assistant",
        "Reciprocal rank fusion algorithm implementation",
        "FAISS IndexFlatIP vs IndexIVFFlat memory usage",
        "Python memory optimization techniques",
        "Speech to text streaming integration Sarvam ElevenLabs",
    ]

    # Generate 100 query variations
    queries = [sample_queries[i % len(sample_queries)] + f" variation {i}" for i in range(num_queries)]

    # Warmup query
    retriever.search("Warmup query", top_k=5)

    latencies_ms: List[float] = []

    for q in queries:
        t0 = time.perf_counter()
        results = retriever.search(q, top_k=5)
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

    latencies = np.array(latencies_ms)
    stats = {
        "mean_ms": round(float(np.mean(latencies)), 2),
        "p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "p90_ms": round(float(np.percentile(latencies, 90)), 2),
        "p99_ms": round(float(np.percentile(latencies, 99)), 2),
        "max_ms": round(float(np.max(latencies)), 2),
    }

    print("\n" + "=" * 60)
    print("           SUB-50MS HYBRID RETRIEVAL BENCHMARK           ")
    print("=" * 60)
    print(f" Total Queries Executed : {num_queries}")
    print(f" Mean Latency          : {stats['mean_ms']} ms")
    print(f" P50 Latency           : {stats['p50_ms']} ms")
    print(f" P90 Latency           : {stats['p90_ms']} ms")
    print(f" P99 Latency           : {stats['p99_ms']} ms")
    print(f" Max Latency           : {stats['max_ms']} ms")
    print("=" * 60)

    if stats["p99_ms"] < target_ms:
        print(f" SUCCESS: SLA PASSED! P99 ({stats['p99_ms']} ms) < {target_ms} ms target.")
    else:
        print(f" WARNING: P99 ({stats['p99_ms']} ms) exceeded {target_ms} ms target.")

    print("=" * 60 + "\n")

    return stats


if __name__ == "__main__":
    # Create sample synthetic documents for verification
    sample_docs = [
        {
            "chunk_id": i + 1,
            "text": f"Document {i+1}: This is a sample document discussing RAG vector retrieval, FAISS indexing, and BM25 sparse search with sub-50ms latency optimization.",
            "metadata": {"doc_id": 1000 + i, "category": "tech"},
        }
        for i in range(500)
    ]

    # Add specific domain documents
    sample_docs.append({
        "chunk_id": 9991,
        "text": "Hybrid search combines dense vector similarity from FAISS with sparse keyword frequency from BM25 using reciprocal rank fusion.",
        "metadata": {"doc_id": 2001, "topic": "search_architecture"},
    })
    sample_docs.append({
        "chunk_id": 9992,
        "text": "Sarvam AI and ElevenLabs provide high-accuracy real-time speech to text transcription for voice-enabled applications.",
        "metadata": {"doc_id": 2002, "topic": "voice_stt"},
    })

    print("Building HybridRetriever in RAM...")
    retriever = HybridRetriever(sample_docs, dense_weight=0.7, sparse_weight=0.3)

    print("\nExecuting sample search query...")
    results = retriever.search("What is hybrid search with FAISS and BM25?", top_k=3)
    print(json.dumps(results, indent=2, ensure_ascii=False))

    print("\nRunning latency benchmark...")
    benchmark_retriever(retriever, num_queries=100)
