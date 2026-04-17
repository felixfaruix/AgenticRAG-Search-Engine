"""vector rag agent: single-hop retrieval via bm25, dense, or hybrid search.
activated when the router selects vector_rag based on single hop_count and passage/book scope.
on retry, broadens retrieval by switching method, increasing top_k, or removing the book filter.
"""
from typing import Any
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from supermemory import Supermemory
from vertexai.language_models import TextEmbeddingModel
from src.models.agent_contracts import AgentResult, Passage, ResolvedEntity, ScratchpadEntry
from src.tools.vector_search import vector_search
from src.tools.write_scratchpad import write_scratchpad, read_scratchpad

method_map: dict[str, str] = {"factual": "bm25", "fuzzy": "dense", "mixed": "hybrid"}


def run_vector_rag(state: dict[str, Any], qdrant_client: QdrantClient, collection_name: str,
                   embedding_model: TextEmbeddingModel, bm25_index: BM25Okapi, chunks: list[dict],
                   sm_client: Supermemory) -> AgentResult:
    """retrieve passages via bm25, dense cosine similarity, or hybrid rrf fusion.
    reads sub_classification from state to select retrieval method.
    on retry, reads grounding feedback from state and adjusts strategy.
    """
    session_id: str = state["session_id"]
    query: str = state["query"]
    understanding = state["understanding"]
    attempt: int = state["attempt_number"]
    resolved: list[ResolvedEntity] = state.get("resolved_entities", [])
    feedback: str | None = state.get("grounding_feedback")

    method: str = method_map.get(understanding.sub_classification, "hybrid")
    top_k: int = 10
    book_ids: list[str] = list({e.book_id for e in resolved})
    book_id: str | None = book_ids[0] if len(book_ids) == 1 else None

    if attempt > 1 and feedback:
        prior: list[ScratchpadEntry] = read_scratchpad("vector_rag", session_id, sm_client)
        stagnant: bool = bool(prior) and (prior[-1].top_score or 0.0) < 0.3

        if method in ("bm25", "dense"):
            method = "hybrid"
        top_k = min(top_k + 5 * (attempt - 1), 30)
        if attempt >= 3 or stagnant:
            book_id = None

    passages: list[Passage] = vector_search(query, method, qdrant_client, collection_name,
                                            embedding_model, bm25_index, chunks, book_id, top_k)

    scratchpad: ScratchpadEntry = ScratchpadEntry(session_id=session_id, agent_type="vector_rag",
                                                  attempt_number=attempt, tool_name="vector_search",
                                                  tool_params={"method": method, "book_id": book_id, "top_k": top_k},
                                                  passages_returned=len(passages), top_score=passages[0].score if passages else None,
                                                  success=True, grounding_feedback=feedback)
    write_scratchpad(scratchpad, sm_client)

    return AgentResult(session_id=session_id, agent_type="vector_rag", query_text=query, retrieved_passages=passages,
                       identified_books=list({p.book_id for p in passages}),
                       tool_calls_made=[{"tool": "vector_search", "method": method, "book_id": book_id, "top_k": top_k}])
