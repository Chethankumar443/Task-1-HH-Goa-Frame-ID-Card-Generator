"""
MSMARCO-XI Dataset Ingestion & Hybrid Semantic Chunking Pipeline
================================================================

This module implements an ultra-fast, production-grade hybrid chunking strategy
for the MSMARCO-XI dataset. It avoids naive fixed-size splitting by leveraging
lightweight sentence-transformers for semantic boundary detection (cosine similarity < 0.6)
and applies 15% token overlap between adjacent chunks while preserving essential metadata.

Author: Senior ML Engineer
Date: 2026-08-21
"""

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HybridSemanticChunker:
    """
    Hybrid chunker combining semantic topic boundary detection via sentence embeddings
    with strategic 15% token overlap between adjacent chunks.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.6,
        overlap_ratio: float = 0.15,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the HybridSemanticChunker.

        Args:
            model_name: Name of the lightweight sentence-transformer model.
            similarity_threshold: Cosine similarity threshold below which a split occurs.
            overlap_ratio: Fraction of tokens (e.g. 0.15 = 15%) to overlap between chunks.
            device: PyTorch device ('cuda', 'cpu', or auto-detected).
        """
        self.similarity_threshold = similarity_threshold
        self.overlap_ratio = overlap_ratio
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Loading embedding model '{model_name}' on device '{self.device}'...")
        self.model = SentenceTransformer(model_name, device=self.device)
        # Regex to split text cleanly on sentence boundaries (. ! ?)
        self.sentence_regex = re.compile(r'(?<=[.!?])\s+')

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences cleanly using regex boundary detection."""
        if not text or not text.strip():
            return []
        sentences = [s.strip() for s in self.sentence_regex.split(text) if s.strip()]
        return sentences if sentences else [text.strip()]

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text. Uses whitespace/word tokenization as high-performance proxy.
        """
        return len(text.split())

    def _extract_overlap_prefix(self, prev_text: str) -> str:
        """Extract exact 15% trailing token overlap from previous chunk."""
        words = prev_text.split()
        if not words:
            return ""
        overlap_count = max(1, int(len(words) * self.overlap_ratio))
        return " ".join(words[-overlap_count:])

    def process_batch(
        self,
        batch_docs: List[Dict[str, Any]],
        start_chunk_id: int = 1
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Process a batch of documents using batched embeddings and GPU/CPU tensor math.

        Args:
            batch_docs: List of parsed document dictionaries containing metadata & text.
            start_chunk_id: The starting ID for chunk indexing.

        Returns:
            Tuple of (list of processed chunk dicts, next_chunk_id).
        """
        all_sentences: List[str] = []
        doc_sentence_spans: List[Tuple[int, int, Dict[str, Any]]] = []

        # Step 1: Collect sentences across batch and map to parent documents
        for doc in batch_docs:
            sentences = self.split_into_sentences(doc["text"])
            if not sentences:
                continue
            start_idx = len(all_sentences)
            all_sentences.extend(sentences)
            end_idx = len(all_sentences)
            doc_sentence_spans.append((start_idx, end_idx, doc))

        if not all_sentences:
            return [], start_chunk_id

        # Step 2: Batch compute sentence embeddings with normalization
        embeddings = self.model.encode(
            all_sentences,
            batch_size=512,
            show_progress_bar=False,
            convert_to_tensor=True,
            normalize_embeddings=True,
            device=self.device
        )

        processed_chunks: List[Dict[str, Any]] = []
        current_chunk_id = start_chunk_id

        # Step 3: Process each document's sentences and detect semantic boundaries
        for start_idx, end_idx, doc in doc_sentence_spans:
            doc_sentences = all_sentences[start_idx:end_idx]
            doc_embeds = embeddings[start_idx:end_idx]

            if len(doc_sentences) == 1:
                semantic_blocks = [doc_sentences]
            else:
                # Vectorized cosine similarity calculation between adjacent sentence pairs
                sims = (doc_embeds[:-1] * doc_embeds[1:]).sum(dim=-1).cpu().tolist()
                
                semantic_blocks = []
                current_block = [doc_sentences[0]]

                for i, sim in enumerate(sims):
                    if sim < self.similarity_threshold:
                        # Cosine similarity < 0.6: Topic boundary detected -> split chunk
                        semantic_blocks.append(current_block)
                        current_block = [doc_sentences[i + 1]]
                    else:
                        current_block.append(doc_sentences[i + 1])
                
                if current_block:
                    semantic_blocks.append(current_block)

            # Step 4: Construct chunks with strategic 15% token overlap & metadata preservation
            prev_chunk_text: Optional[str] = None

            for block in semantic_blocks:
                block_raw_text = " ".join(block)
                
                if prev_chunk_text is not None:
                    overlap_prefix = self._extract_overlap_prefix(prev_chunk_text)
                    chunk_text = f"{overlap_prefix} {block_raw_text}".strip()
                else:
                    chunk_text = block_raw_text

                token_length = self.count_tokens(chunk_text)

                metadata = {
                    "doc_id": int(doc["doc_id"]),
                    "language": str(doc["language"]),
                    "source": str(doc["source"]),
                    "query_type": str(doc["query_type"]),
                }

                chunk_obj = {
                    "chunk_id": current_chunk_id,
                    "doc_id": int(doc["doc_id"]),
                    "text": chunk_text,
                    "metadata": metadata,
                    "token_length": token_length,
                }

                processed_chunks.append(chunk_obj)
                prev_chunk_text = block_raw_text
                current_chunk_id += 1

        return processed_chunks, current_chunk_id


def extract_doc_fields(sample: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
    """
    Extract standard text and metadata from dataset row.
    Handles MSMARCO-XI schema variations cleanly.
    """
    # 1. doc_id
    doc_id = sample.get("doc_id") or sample.get("query_id") or index
    try:
        doc_id = int(doc_id)
    except (ValueError, TypeError):
        doc_id = index

    # 2. language
    language = (
        sample.get("language")
        or sample.get("target_lang")
        or sample.get("source_lang")
        or "en"
    )

    # 3. source
    source = sample.get("source") or "MSMARCO-XI"

    # 4. query_type
    query_type = sample.get("query_type") or "description"

    # 5. text extraction
    text = ""
    passages = sample.get("passages")
    if isinstance(passages, dict):
        eng_passages = passages.get("English_passages") or []
        trans_passages = passages.get("Translated_passages") or []
        p_texts = eng_passages if eng_passages else trans_passages
        if not p_texts:
            p_texts = passages.get("passage_text") or passages.get("text") or []
        if isinstance(p_texts, list):
            text = " ".join([str(t) for t in p_texts if t])
        elif isinstance(p_texts, str):
            text = p_texts
    elif isinstance(passages, list):
        extracted = []
        for p in passages:
            if isinstance(p, dict):
                p_text = p.get("passage_text") or p.get("text") or ""
                if p_text:
                    extracted.append(p_text)
            elif isinstance(p, str):
                extracted.append(p)
        text = " ".join(extracted)

    if not text or not text.strip():
        text = sample.get("Eng_Answer") or sample.get("Answer") or sample.get("Eng_Query") or sample.get("query") or sample.get("text") or ""

    if not text or not text.strip():
        return None

    return {
        "doc_id": doc_id,
        "language": str(language),
        "source": str(source),
        "query_type": str(query_type),
        "text": text.strip(),
    }


def ingest_and_chunk_dataset(
    dataset_name: str = "ai4bharat/MSMARCO-XI",
    split: str = "train",
    sample_size: int = 10000,
    batch_size: int = 100,
    output_filepath: str = "msmarco_chunks.jsonl",
) -> None:
    """
    Main ingestion and chunking pipeline execution function.

    Args:
        dataset_name: Hugging Face dataset identifier.
        split: Dataset split to load.
        sample_size: Number of records to process (default 10,000).
        batch_size: Batch size for batched sentence processing.
        output_filepath: Path to save JSONL output.
    """
    start_time = time.time()
    logger.info(f"Loading dataset '{dataset_name}' (split: '{split}', sample: {sample_size})...")

    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    # Load dataset sample using fast streaming with HF authentication token
    try:
        ds = load_dataset(dataset_name, split=split, streaming=True, token=hf_token)
    except Exception as e:
        logger.warning(f"Could not stream dataset from HF ({e}). Falling back to standard load.")
        ds = load_dataset(dataset_name, split=split, token=hf_token)

    logger.info(f"Dataset streaming initialized. Processing up to {sample_size} documents...")

    chunker = HybridSemanticChunker(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold=0.6,
        overlap_ratio=0.15,
    )

    current_chunk_id = 1
    total_docs_processed = 0
    total_chunks_generated = 0

    with open(output_filepath, "w", encoding="utf-8") as f_out:
        batch_docs: List[Dict[str, Any]] = []

        for idx, row in enumerate(ds):
            if total_docs_processed >= sample_size:
                break

            doc = extract_doc_fields(row, idx + 1)
            if doc:
                batch_docs.append(doc)
                total_docs_processed += 1

            if len(batch_docs) >= batch_size:
                chunks, current_chunk_id = chunker.process_batch(
                    batch_docs, start_chunk_id=current_chunk_id
                )
                for chunk in chunks:
                    f_out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                f_out.flush()
                total_chunks_generated += len(chunks)
                logger.info(f"Processed {total_docs_processed}/{sample_size} documents -> {total_chunks_generated} total chunks generated")
                batch_docs = []

        # Process remaining documents in batch
        if batch_docs:
            chunks, current_chunk_id = chunker.process_batch(
                batch_docs, start_chunk_id=current_chunk_id
            )
            for chunk in chunks:
                f_out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            f_out.flush()
            total_chunks_generated += len(chunks)

    elapsed_time = time.time() - start_time
    logger.info(
        f"Processing complete in {elapsed_time:.2f} seconds!\n"
        f"  - Total Documents Processed: {total_docs_processed}\n"
        f"  - Total Chunks Generated: {total_chunks_generated}\n"
        f"  - Throughput: {total_docs_processed / max(elapsed_time, 0.001):.1f} docs/sec\n"
        f"  - Output Saved To: {output_filepath}"
    )


if __name__ == "__main__":
    ingest_and_chunk_dataset(
        dataset_name="ai4bharat/MSMARCO-XI",
        split="train",
        sample_size=1000,
        batch_size=100,
        output_filepath="msmarco_chunks.jsonl",
    )
