"""vector search: bm25 (in-memory via rank_bm25) and dense (qdrant) retrieval."""

from typing import Literal
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from src.models.agent_contracts import Passage


def book_filter(book_id: str) -> models.Filter:
    """payload filter restricting results to a single book."""
    return models.Filter(must=[models.FieldCondition(key="book_id", match=models.MatchValue(value=book_id))])


def point_to_passage(point: models.ScoredPoint, method: str, agent: str = "vector_rag") -> Passage:
    """map a qdrant scored point to a passage with retrieval provenance."""
    p: dict = point.payload
    return Passage(book_id=p["book_id"], book_title=p["book_title"], chapter_number=p["chapter_number"],
                   chapter_title=p.get("chapter_title"), chunk_index=p["chunk_index"], text=p["text"],
                   score=point.score, retrieval_method=method, retrieval_agent=agent)


def chunk_to_passage(chunk: dict, score: float, method: str, agent: str = "vector_rag") -> Passage:
    """map a raw chunk dict + bm25 score to a passage."""
    return Passage(book_id=chunk["book_id"], book_title=chunk["book_title"], chapter_number=chunk["chapter_number"],
                   chapter_title=chunk.get("chapter_title"), chunk_index=chunk["chunk_index"], text=chunk["text"],
                   score=score, retrieval_method=method, retrieval_agent=agent)


def bm25_search(query: str, bm25_index: BM25Okapi, chunks: list[dict],
                book_id: str | None = None, top_k: int = 10) -> list[Passage]:
    """bm25 keyword search over in-memory index. optionally filtered by book."""
    scores: list[float] = bm25_index.get_scores(query.lower().split()).tolist()
    scored: list[tuple[int, float]] = [(i, s) for i, s in enumerate(scores) if s > 0]

    if book_id:
        scored = [(i, s) for i, s in scored if chunks[i]["book_id"] == book_id]

    scored.sort(key=lambda x: x[1], reverse=True)
    return [chunk_to_passage(chunks[i], s, "bm25") for i, s in scored[:top_k]]


def dense_search(query: str, qdrant_client: QdrantClient, collection_name: str,
                 embedding_model: TextEmbeddingModel, book_id: str | None = None, top_k: int = 10) -> list[Passage]:
    """dense cosine similarity search via qdrant."""
    qf: models.Filter | None = book_filter(book_id) if book_id else None
    q_vec: list[float] = embedding_model.get_embeddings([TextEmbeddingInput(text=query, task_type="RETRIEVAL_QUERY")])[0].values
    results: models.QueryResponse = qdrant_client.query_points(
        collection_name=collection_name, query=q_vec, using="contextual", query_filter=qf, limit=top_k)
    return [point_to_passage(p, "dense") for p in results.points]


def hybrid_search(query: str, bm25_index: BM25Okapi, chunks: list[dict], qdrant_client: QdrantClient,
                  collection_name: str, embedding_model: TextEmbeddingModel,
                  book_id: str | None = None, top_k: int = 10,
                  dense_weight: float = 1.0, bm25_weight: float = 0.6) -> list[Passage]:
    """weighted reciprocal rank fusion of bm25 and dense. dense_weight > bm25_weight
    keeps dense in charge of semantic precision while bm25 still contributes rank
    signal for queries rich in named entities. raise bm25_weight if the corpus
    has very strong quoted-entity recall requirements.
    """
    bm25_results: list[Passage] = bm25_search(query, bm25_index, chunks, book_id, top_k=top_k * 2)
    dense_results: list[Passage] = dense_search(query, qdrant_client, collection_name, embedding_model, book_id, top_k=top_k * 2)
    rrf_k: int = 60
    rrf_scores: dict[str, float] = {}
    rrf_passages: dict[str, Passage] = {}

    for rank, p in enumerate(bm25_results):
        key: str = f"{p.book_id}:{p.chapter_number}:{p.chunk_index}"
        rrf_scores[key] = rrf_scores.get(key, 0.0) + bm25_weight / (rrf_k + rank + 1)
        rrf_passages[key] = p

    for rank, p in enumerate(dense_results):
        key = f"{p.book_id}:{p.chapter_number}:{p.chunk_index}"
        rrf_scores[key] = rrf_scores.get(key, 0.0) + dense_weight / (rrf_k + rank + 1)
        if key not in rrf_passages:
            rrf_passages[key] = p

    ranked: list[tuple[str, float]] = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [rrf_passages[key].model_copy(update={"score": score, "retrieval_method": "hybrid"}) for key, score in ranked]


def vector_search(query: str, method: Literal["bm25", "dense", "hybrid"], qdrant_client: QdrantClient,
                  collection_name: str, embedding_model: TextEmbeddingModel, bm25_index: BM25Okapi,
                  chunks: list[dict], book_id: str | None = None, top_k: int = 10) -> list[Passage]:
    """unified entry point. bm25 runs in-memory, dense runs via qdrant, hybrid fuses both."""
    if method == "bm25":
        return bm25_search(query, bm25_index, chunks, book_id, top_k)
    if method == "dense":
        return dense_search(query, qdrant_client, collection_name, embedding_model, book_id, top_k)
    return hybrid_search(query, bm25_index, chunks, qdrant_client, collection_name, embedding_model, book_id, top_k)
