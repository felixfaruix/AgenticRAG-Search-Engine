"""Thematic agent: book-level summary search for broad thematic and exploratory queries.
Activated when the router selects thematic based on exploratory scope.
On retry, increases the number of book summaries returned.
"""
from typing import Any
import numpy as np
from vertexai.language_models import TextEmbeddingModel
from src.models.agent_contracts import AgentResult, Passage, ScratchpadEntry
from src.tools.book_summary_search import book_summary_search
from src.tools.write_scratchpad import write_scratchpad

def run_thematic(state: dict[str, Any], summaries: list[dict[str, Any]], summary_embeddings: np.ndarray,
                 embedding_model: TextEmbeddingModel, sm_client: Any) -> AgentResult:
    """Dense cosine similarity search over book-level summaries.
    Returns book-level passages ranked by relevance to the query.
    """
    session_id: str = state["session_id"]
    query: str = state["query"]
    attempt: int = state["attempt_number"]
    top_k: int = 3

    if attempt > 1:
        top_k = min(top_k + 2 * (attempt - 1), len(summaries))

    passages: list[Passage] = book_summary_search(query, summaries, summary_embeddings, embedding_model, top_k)

    scratchpad: ScratchpadEntry = ScratchpadEntry(
        session_id=session_id, agent_type="thematic", attempt_number=attempt, tool_name="book_summary_search",
        tool_params={"top_k": top_k}, passages_returned=len(passages),
        top_score=passages[0].score if passages else None, success=True,
        grounding_feedback=state.get("grounding_feedback"))
    write_scratchpad(scratchpad, sm_client)

    return AgentResult(session_id=session_id, agent_type="thematic", query_text=query, retrieved_passages=passages,
                        identified_books=list({p.book_id for p in passages}), confidence=passages[0].score if passages else 0.0,
                        tool_calls_made=[{"tool": "book_summary_search", "top_k": top_k}])
