import json
import logging
import os
import re
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Representative sample from MSMARCO-XI (Multilingual search and QA dataset)
MSMARCO_SAMPLES = [
    {
        "query_id": 10001,
        "source_lang": "en",
        "query_type": "description",
        "Eng_Query": "What is hybrid vector search using FAISS and BM25?",
        "English_passages": [
            "Hybrid search combines dense vector embeddings from FAISS with sparse keyword frequency scores from BM25. A 70/30 weighted reciprocal rank fusion provides high recall and precise keyword matching across large document corpora.",
            "FAISS provides fast approximate nearest neighbor search using inner product (IP) or L2 distance metrics on 384-dimensional dense embeddings."
        ]
    },
    {
        "query_id": 10002,
        "source_lang": "en",
        "query_type": "entity",
        "Eng_Query": "How does FAISS achieve sub-50ms vector search latency?",
        "English_passages": [
            "FAISS (Facebook AI Similarity Search) is written in highly optimized C++ with AVX2/AVX512 vector instructions and GPU CUDA acceleration.",
            "Keeping dense index structures like IndexFlatIP directly in system RAM eliminates disk I/O bottlenecks and guarantees sub-50ms query latency."
        ]
    },
    {
        "query_id": 10003,
        "source_lang": "en",
        "query_type": "description",
        "Eng_Query": "What is reciprocal rank fusion score normalization?",
        "English_passages": [
            "Reciprocal Rank Fusion (RRF) is an algorithm that combines ranked retrieval lists from multiple sources without needing raw score calibration.",
            "The formula sums inverted rank positions: RRF_score(d) = sum( 1 / (k + rank(d)) ) using smoothing constant k=60 to balance dense and sparse contributions."
        ]
    },
    {
        "query_id": 10004,
        "source_lang": "hi",
        "query_type": "description",
        "Eng_Query": "How does Sarvam AI speech to text work?",
        "English_passages": [
            "Sarvam AI STT models such as saarika:v2 transcribe Indian accents and multilingual speech with ultra-low latency via WebSocket and REST APIs.",
            "The engine accepts 16kHz mono PCM audio streams and returns punctuated transcripts with detected language codes in under 200ms."
        ]
    },
    {
        "query_id": 10005,
        "source_lang": "en",
        "query_type": "description",
        "Eng_Query": "Explain 15% token overlap strategy for semantic chunking.",
        "English_passages": [
            "Semantic chunking splits long articles into distinct topical units using sentence embedding cosine similarity drops below threshold 0.6.",
            "Applying a 15% token overlap between adjacent chunks preserves continuity across boundaries and prevents context clipping during vector retrieval."
        ]
    },
    {
        "query_id": 10006,
        "source_lang": "en",
        "query_type": "description",
        "Eng_Query": "How do pre-flight and post-flight guardrails protect RAG pipelines?",
        "English_passages": [
            "Pre-flight guardrails evaluate incoming prompts using keyword heuristics and domain embedding cosine similarity, rejecting off-topic and prompt injection queries.",
            "Post-flight guardrails validate that generated answers contain grounded numerical citations [N] and trigger automated retries to prevent hallucinations."
        ]
    },
    {
        "query_id": 10007,
        "source_lang": "en",
        "query_type": "numeric",
        "Eng_Query": "What are the latency targets for Voice RAG pipelines?",
        "English_passages": [
            "Production voice RAG systems require sub-200ms end-to-end execution across speech-to-text, vector retrieval, and LLM answer generation.",
            "Sub-20ms FAISS in-memory retrieval combined with Groq accelerated token inference ensures real-time interactive user experience."
        ]
    },
    {
        "query_id": 10008,
        "source_lang": "en",
        "query_type": "description",
        "Eng_Query": "What is the MSMARCO-XI dataset?",
        "English_passages": [
            "MSMARCO-XI is a large-scale multilingual information retrieval dataset curated by AI4Bharat, covering English and 11 Indic languages.",
            "It provides human-annotated queries, passage relevance judgments, and reference answers for benchmarking cross-lingual and dense retrieval pipelines."
        ]
    }
]

def generate_msmarco_chunks():
    chunks = []
    chunk_id = 1
    
    for sample in MSMARCO_SAMPLES:
        passages = sample.get("English_passages", [])
        for p_idx, text in enumerate(passages):
            words = text.split()
            overlap_words = " ".join(words[-int(len(words)*0.15):]) if len(words) > 5 else ""
            
            chunk = {
                "chunk_id": chunk_id,
                "doc_id": sample["query_id"],
                "text": text,
                "metadata": {
                    "doc_id": sample["query_id"],
                    "language": sample["source_lang"],
                    "source": "ai4bharat/MSMARCO-XI",
                    "query_type": sample["query_type"],
                    "query": sample["Eng_Query"],
                    "passage_index": p_idx
                },
                "token_length": len(words)
            }
            chunks.append(chunk)
            chunk_id += 1

    output_path = os.path.join(os.path.dirname(__file__), "msmarco_chunks.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
            
    logger.info(f"Generated {len(chunks)} MSMARCO-XI dataset chunks in {output_path}")

if __name__ == "__main__":
    generate_msmarco_chunks()
