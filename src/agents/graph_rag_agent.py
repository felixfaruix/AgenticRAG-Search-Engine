"""Graph RAG agent: multi-hop traversal over Supermemory knowledge graph.
Activated when the router selects graph_rag based on multi hop_count.
For temporal queries, restricts traversal to OCCURS_BEFORE edges.
On retry, increases max_hops, broadens top_k, or drops relationship filter.
"""
from typing import Any
from supermemory import Supermemory
from src.models.agent_contracts import AgentResult, Passage, ResolvedEntity, ScratchpadEntry
from src.tools.graph_search import graph_search
from src.tools.write_scratchpad import write_scratchpad

def run_graph_rag(state: dict[str, Any], sm_client: Supermemory) -> AgentResult:
    """Traverse typed edges in the Supermemory memory graph from resolved entity nodes.
    Collects source chunk text from each traversed node as retrieval output.
    """
    session_id: str = state["session_id"]
    query: str = state["query"]
    understanding = state["understanding"]
    attempt: int = state["attempt_number"]
    resolved: list[ResolvedEntity] = state.get("resolved_entities", [])
    feedback: str | None = state.get("grounding_feedback")

    node_ids: list[str] = [e.canonical_id for e in resolved]
    book_ids: list[str] = list({e.book_id for e in resolved})
    book_id: str = book_ids[0] if book_ids else ""
    relationship_type: str | None = "OCCURS_BEFORE" if understanding.query_type == "temporal" else None
    max_hops: int = 2
    top_k: int = 10

    # adjust strategy on retry
    if attempt > 1 and feedback:
        max_hops = min(max_hops + 1, 4)
        top_k = min(top_k + 5, 25)
        if attempt >= 3 and relationship_type:
            relationship_type = None

    passages: list[Passage] = graph_search(node_ids, sm_client, book_id, relationship_type, max_hops, top_k)

    scratchpad: ScratchpadEntry = ScratchpadEntry(
        session_id=session_id, agent_type="graph_rag", attempt_number=attempt, tool_name="graph_search",
        tool_params={"start_node_ids": node_ids, "book_id": book_id,
                     "relationship_type": relationship_type, "max_hops": max_hops, "top_k": top_k},
        passages_returned=len(passages), top_score=passages[0].score if passages else None,
        success=True, grounding_feedback=feedback)
    
    write_scratchpad(scratchpad, sm_client)

    return AgentResult(session_id=session_id, agent_type="graph_rag", query_text=query, retrieved_passages=passages,
        identified_books=list({p.book_id for p in passages}),
        tool_calls_made=[{"tool": "graph_search", "start_node_ids": node_ids, "book_id": book_id,
                         "relationship_type": relationship_type, "max_hops": max_hops}])
