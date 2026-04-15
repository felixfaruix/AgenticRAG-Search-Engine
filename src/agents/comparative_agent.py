"""Comparative agent: parallel cross-book search combining vector and graph retrieval.
Activated when the router selects comparative based on cross_book scope.
Searches each target book via hybrid vector search and graph traversal,
then deduplicates and ranks the combined results.
"""
from typing import Any
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from supermemory import Supermemory
from vertexai.language_models import TextEmbeddingModel
from src.models.agent_contracts import AgentResult, Passage, ResolvedEntity, ScratchpadEntry
from src.tools.vector_search import vector_search
from src.tools.graph_search import graph_search
from src.tools.write_scratchpad import write_scratchpad

def run_comparative(state: dict[str, Any], qdrant_client: QdrantClient, collection_name: str,
                    embedding_model: TextEmbeddingModel, bm25_index: BM25Okapi, chunks: list[dict],
                    sm_client: Supermemory) -> AgentResult:
    """Search multiple books in parallel using hybrid vector search and graph traversal.
    The orchestrator passes resolved entities spanning multiple books.
    """
    session_id: str = state["session_id"]
    query: str = state["query"]
    attempt: int = state["attempt_number"]
    resolved: list[ResolvedEntity] = state.get("resolved_entities", [])
    book_ids: list[str] = list({e.book_id for e in resolved})
    top_k_per_book: int = 5 if attempt == 1 else 8
    all_passages: list[Passage] = []
    tool_calls: list[dict[str, Any]] = []

    for bid in book_ids:
        vec_passages: list[Passage] = vector_search(query, "hybrid", qdrant_client, collection_name,
                                                    embedding_model, bm25_index, chunks, book_id=bid, top_k=top_k_per_book)
        all_passages.extend([p.model_copy(update={"retrieval_agent": "comparative"}) for p in vec_passages])
        tool_calls.append({"tool": "vector_search", "method": "hybrid", "book_id": bid, "top_k": top_k_per_book})

        book_entities: list[ResolvedEntity] = [e for e in resolved if e.book_id == bid]

        if book_entities:
            node_ids: list[str] = [e.canonical_id for e in book_entities]
            graph_passages: list[Passage] = graph_search(node_ids, sm_client, bid, top_k=top_k_per_book)
            all_passages.extend([p.model_copy(update={"retrieval_agent": "comparative"}) for p in graph_passages])
            tool_calls.append({"tool": "graph_search", "start_node_ids": node_ids, "book_id": bid})

    # deduplicate by (book, chapter, chunk), keep highest score
    seen: dict[str, Passage] = {}
    
    for p in all_passages:
        key: str = f"{p.book_id}:{p.chapter_number}:{p.chunk_index}"
        if key not in seen or p.score > seen[key].score:
            seen[key] = p
            
    deduped: list[Passage] = sorted(seen.values(), key=lambda p: p.score, reverse=True)

    scratchpad: ScratchpadEntry = ScratchpadEntry(session_id=session_id, agent_type="comparative", attempt_number=attempt,
                                                    tool_name="vector_search+graph_search", 
                                                    tool_params={"book_ids": book_ids, "top_k_per_book": top_k_per_book},
                                                    passages_returned=len(deduped), top_score=deduped[0].score if deduped else None,
                                                    success=True, grounding_feedback=state.get("grounding_feedback"))
    write_scratchpad(scratchpad, sm_client)

    return AgentResult(session_id=session_id, agent_type="comparative", query_text=query, retrieved_passages=deduped,
                        identified_books=book_ids, confidence=deduped[0].score if deduped else 0.0, tool_calls_made=tool_calls)
