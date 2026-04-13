"""Vector RAG agent: single-hop retrieval via BM25, dense, or hybrid search.
Activated when the router selects vector_rag based on single hop_count and passage/book scope.
On retry, broadens retrieval by switching method, increasing top_k, or removing book filter.
"""
from typing import Any
import numpy as np
from rank_bm25 import BM25Okapi
from vertexai.language_models import TextEmbeddingModel
from src.models.agent_contracts import AgentResult, Passage, ResolvedEntity, ScratchpadEntry
from src.tools.vector_search import vector_search
from src.tools.write_scratchpad import write_scratchpad

method_map: dict[str, str] = {"factual": "bm25", "fuzzy": "dense", "mixed": "hybrid"}

def run_vector_rag(state: dict[str, Any], chunks: list[Any], enriched_chunks: list[Any],
                   bm25_index: BM25Okapi, embeddings: np.ndarray, embedding_model: TextEmbeddingModel, sm_client: Any) -> AgentResult:
    """Retrieve passages via BM25, dense cosine similarity, or hybrid RRF fusion.
    Reads sub_classification from state to select retrieval method.
    On retry, reads grounding feedback from state and adjusts strategy.
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

    # adjust strategy on retry based on grounding feedback
    if attempt > 1 and feedback:
        if method in ("bm25", "dense"):
            method = "hybrid"
        top_k = min(top_k + 5 * (attempt - 1), 30)
        if attempt >= 3:
            book_id = None

    passages: list[Passage] = vector_search(query, method, chunks, enriched_chunks, bm25_index,
                                            embeddings, embedding_model, book_id, top_k)

    scratchpad: ScratchpadEntry = ScratchpadEntry(session_id=session_id, agent_type="vector_rag", 
                                                attempt_number=attempt, tool_name="vector_search",
                                                tool_params={"method": method, "book_id": book_id, "top_k": top_k},
                                                passages_returned=len(passages), top_score=passages[0].score if passages else None,
                                                success=True, grounding_feedback=feedback)
    write_scratchpad(scratchpad, sm_client)

    return AgentResult(session_id=session_id, agent_type="vector_rag", query_text=query, retrieved_passages=passages,
                        identified_books=list({p.book_id for p in passages}), 
                        confidence=passages[0].score if passages else 0.0,
                        tool_calls_made=[{"tool": "vector_search", "method": method, "book_id": book_id, "top_k": top_k}])
