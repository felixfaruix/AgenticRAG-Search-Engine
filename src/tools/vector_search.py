"""Vector search tool: BM25, dense, and hybrid retrieval."""

from typing import Any, Literal
import numpy as np
from rank_bm25 import BM25Okapi
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from src.models.agent_contracts import Passage


def reciprocal_rank_fusion(ranked_lists: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
    """Combine ranked lists via RRF. Each list contains (index, score) tuples sorted descending.
    """
    fused: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (idx, score) in enumerate(ranked):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)

def chunk_to_passage(chunk: Any, score: float, method: str, agent: str = "vector_rag") -> Passage:
    """Converting a Chunk or EnrichedChunk to a Passage for the Synthesis agent and retreival metadata. 
    """
    return Passage(book_id=chunk.book_id, book_title=chunk.book_title, chapter_number=chunk.chapter_number,
                    chapter_title=getattr(chunk, "chapter_title", None), chunk_index=chunk.chunk_index,
                    text=chunk.text, score=score, retrieval_method=method, retrieval_agent=agent)

def bm25_search(query: str, bm25_index: BM25Okapi, chunks: list[Any], book_id: str | None, top_k: int) -> list[tuple[int, float]]:
    """Score all chunks via BM25, optionally filter by book, return top-k (index, score) pairs.
    """
    scores: np.ndarray = bm25_index.get_scores(query.lower().split())
    indexed: list[tuple[int, float]] = list(enumerate(scores.tolist()))

    if book_id:
        indexed = [(i, s) for i, s in indexed if chunks[i].book_id == book_id]
    indexed.sort(key=lambda x: x[1], reverse=True)

    return indexed[:top_k]

def dense_search(query: str, embeddings: np.ndarray, chunks: list[Any], embedding_model: TextEmbeddingModel, 
                book_id: str | None, top_k: int) -> list[tuple[int, float]]:
    """Score all chunks via cosine similarity, optionally filter by book, return top k (indexa nd score) pairs.
    """
    q_input: TextEmbeddingInput = TextEmbeddingInput(text=query, task_type="RETRIEVAL_QUERY")
    q_vec: np.ndarray = np.array(embedding_model.get_embeddings([q_input])[0].values, dtype=np.float32)
    norms: np.ndarray = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_vec)
    scores: np.ndarray = (embeddings @ q_vec) / np.where(norms > 0, norms, 1.0)
    indexed: list[tuple[int, float]] = list(enumerate(scores.tolist()))

    if book_id:
        indexed = [(i, s) for i, s in indexed if chunks[i].book_id == book_id]
    indexed.sort(key=lambda x: x[1], reverse=True)

    return indexed[:top_k]

def vector_search(query: str, method: Literal["bm25", "dense", "hybrid"], chunks: list[Any],
                enriched_chunks: list[Any], bm25_index: BM25Okapi, embeddings: np.ndarray,
                embedding_model: TextEmbeddingModel, book_id: str | None = None, top_k: int = 10) -> list[Passage]:
    """Retrieve passages via BM25, dense cosine similarity, or hybrid RRF fusion.
    chunks: list[Chunk] from recursive splitting, aligned with bm25_index.
    enriched_chunks: list[EnrichedChunk] with contextual headers, aligned with embeddings.
    Both lists share the same ordering and length so indices are interchangeable.
    """
    if method == "bm25":
        results: list[tuple[int, float]] = bm25_search(query, bm25_index, chunks, book_id, top_k)
        return [chunk_to_passage(chunks[i], s, "bm25") for i, s in results]

    if method == "dense":
        results = dense_search(query, embeddings, enriched_chunks, embedding_model, book_id, top_k)
        return [chunk_to_passage(enriched_chunks[i], s, "dense") for i, s in results]

    # hybrid: RRF over BM25 + dense, fetch 2x candidates from each before fusing
    bm25_results: list[tuple[int, float]] = bm25_search(query, bm25_index, chunks, book_id, top_k * 2)
    dense_results: list[tuple[int, float]] = dense_search(query, embeddings, enriched_chunks, embedding_model, book_id, top_k * 2)
    fused: list[tuple[int, float]] = reciprocal_rank_fusion([bm25_results, dense_results])[:top_k]

    return [chunk_to_passage(enriched_chunks[i], s, "hybrid") for i, s in fused]
