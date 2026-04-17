"""Orchestrator: LangGraph state graph wiring all agents into a single retrieval pipeline."""

from typing import Any, TypedDict
import instructor
import numpy as np
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from supermemory import Supermemory
from vertexai.language_models import TextEmbeddingModel
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from src.models.agent_contracts import (QueryUnderstanding, ResolvedEntity, AgentResult, SynthesizedAnswer, ScratchpadEntry, Passage)
from src.session import create_session_id
from src.tools.resolve_entities import resolve_entities
from src.tools.route_query import route_query
from src.tools.write_scratchpad import write_scratchpad
from src.tools.vector_search import vector_search
from src.agents.intent_classifier import classify_query
from src.agents.vector_rag_agent import run_vector_rag
from src.agents.graph_rag_agent import run_graph_rag
from src.agents.thematic_agent import run_thematic
from src.agents.comparative_agent import run_comparative
from src.agents.synthesis_agent import run_synthesis

disambiguation_threshold: float = 0.5

class OrchestratorState(TypedDict):
    """LangGraph state schema carrying all data between nodes."""
    query: str
    session_id: str
    understanding: QueryUnderstanding | None
    resolved_entities: list[ResolvedEntity]
    routed_agent: str
    agent_results: list[AgentResult]
    synthesized_answer: SynthesizedAnswer | None
    attempt_number: int
    max_attempts: int
    grounding_feedback: str | None
    production_mode: bool
    fallback_attempted: bool

def initial_state(query: str, production_mode: bool = False) -> OrchestratorState:
    """Create the initial state dict for a new query.
    """
    return OrchestratorState(
        query=query, session_id=create_session_id(), understanding=None,
        resolved_entities=[], routed_agent="", agent_results=[],
        synthesized_answer=None, attempt_number=1, max_attempts=3,
        grounding_feedback=None, production_mode=production_mode, fallback_attempted=False)

def build_graph(qdrant_client: QdrantClient, collection_name: str, embedding_model: TextEmbeddingModel,
                bm25_index: BM25Okapi, chunks: list[dict], summaries: list[dict[str, Any]],
                summary_embeddings: np.ndarray, sm_client: Supermemory,
                alias_index: dict[str, list[dict[str, Any]]],
                classifier_model: str, classifier_client: instructor.Instructor,
                synthesis_model: str, synthesis_client: instructor.Instructor) -> Any:
    """Build and compile the orchestrator state graph with checkpointing.

    Two model tiers:
      classifier_model / classifier_client — fine-tuned lightweight model for intent classification.
      synthesis_model / synthesis_client   — powerfu.  njqu8l model for answer synthesis and grounding.
    Retrieval agents use no LLM; they operate through Qdrant, Supermemory, and embeddings.
    """

    def classify_node(state: OrchestratorState) -> dict[str, Any]:
        """Classify user query into structured intent fields."""
        return {"understanding": classify_query(state["query"], classifier_model, classifier_client)}

    def resolve_node(state: OrchestratorState) -> dict[str, Any]:
        """Resolve raw entity mentions to canonical graph nodes via RapidFuzz.
        Only triggers disambiguation when the top match for a mention is below threshold.
        The resolved list is already sorted by confidence descending, so we only need to
        check the first result per mention.
        """
        understanding: QueryUnderstanding | None = state["understanding"]

        if not understanding or not understanding.extracted_entities:
            return {"resolved_entities": []}

        resolved: list[ResolvedEntity] = resolve_entities(understanding.extracted_entities, alias_index)

        if resolved and resolved[0].confidence < disambiguation_threshold:
            ambiguous: ResolvedEntity = resolved[0]
            clarification: dict[str, Any] = interrupt({
                "type": "disambiguation",
                "message": f"Which character do you mean by \"{ambiguous.raw_mention}\"? "
                           f"The closest match is {ambiguous.canonical_name} from {ambiguous.book_id}, "
                           f"but I'm not confident. Could you clarify?"})

            if isinstance(clarification, dict) and "confirmed_entities" in clarification:
                resolved = resolve_entities(clarification["confirmed_entities"], alias_index)

        return {"resolved_entities": resolved}

    def route_node(state: OrchestratorState) -> dict[str, Any]:
        """Deterministic routing from intent classification to agent type."""
        return {"routed_agent": route_query(state["understanding"]) if state["understanding"] else "vector_rag"}

    def dispatch_edge(state: OrchestratorState) -> str:
        """Dispatch to the agent node selected by route_node."""
        return state["routed_agent"]

    def vector_rag_node(state: OrchestratorState) -> dict[str, Any]:
        """Run vector RAG retrieval."""
        return {"agent_results": [run_vector_rag(state, qdrant_client, collection_name, embedding_model, bm25_index, chunks, sm_client)]}

    def graph_rag_node(state: OrchestratorState) -> dict[str, Any]:
        """Run graph RAG traversal."""
        return {"agent_results": [run_graph_rag(state, sm_client)]}

    def thematic_node(state: OrchestratorState) -> dict[str, Any]:
        """Run thematic book-level search."""
        return {"agent_results": [run_thematic(state, summaries, summary_embeddings, embedding_model, sm_client)]}

    def comparative_node(state: OrchestratorState) -> dict[str, Any]:
        """Run comparative cross-book search."""
        return {"agent_results": [run_comparative(state, qdrant_client, collection_name, embedding_model, bm25_index, chunks, sm_client)]}

    def synthesize_node(state: OrchestratorState) -> dict[str, Any]:
        """Generate grounded answer, verify, write feedback on failure.
        """
        answer: SynthesizedAnswer = run_synthesis(state, synthesis_model, synthesis_client, sm_client)

        if answer.grounding_passed:
            return {"synthesized_answer": answer}
        
        write_scratchpad(ScratchpadEntry(
            session_id=state["session_id"], agent_type=state["routed_agent"],
            attempt_number=state["attempt_number"], tool_name="grounding_check",
            tool_params={}, passages_returned=len(answer.cited_passages),
            top_score=answer.confidence, success=False,
            grounding_feedback="Grounding failed. Broaden or adjust retrieval strategy."), sm_client)
        
        return {"synthesized_answer": answer, "grounding_feedback": "Grounding failed. Adjust retrieval strategy.",
                "attempt_number": state["attempt_number"] + 1}

    def fallback_node(state: OrchestratorState) -> dict[str, Any]:
        """Production fallback: graph_rag exhausted retries, try hybrid vector search.
        """
        passages: list[Passage] = vector_search(state["query"], "hybrid", qdrant_client, collection_name, embedding_model, bm25_index, chunks, top_k=15)

        result: AgentResult = AgentResult(
            session_id=state["session_id"], agent_type="graph_rag_fallback", query_text=state["query"],
            retrieved_passages=passages, identified_books=list({p.book_id for p in passages}),
            tool_calls_made=[{"tool": "vector_search", "method": "hybrid", "fallback": True}])
        
        return {"agent_results": [result], "fallback_attempted": True, "attempt_number": 1}

    def post_synthesis_edge(state: OrchestratorState) -> str:
        """Route after synthesis: done, retry, or fallback.
        """
        answer: SynthesizedAnswer | None = state.get("synthesized_answer")
        
        if not answer or answer.grounding_passed:
            return "done"
        if state["attempt_number"] > state["max_attempts"]:
            if state.get("production_mode") and state["routed_agent"] == "graph_rag" and not state.get("fallback_attempted"):
                return "fallback"
            return "done"
        
        return state["routed_agent"]

    graph: StateGraph = StateGraph(OrchestratorState)
    for name, fn in [("classify", classify_node), ("resolve", resolve_node), ("route", route_node),
                     ("vector_rag", vector_rag_node), ("graph_rag", graph_rag_node),
                     ("thematic", thematic_node), ("comparative", comparative_node),
                     ("synthesize", synthesize_node), ("fallback", fallback_node)]:
        graph.add_node(name, fn)

    graph.add_edge(START, "classify")
    graph.add_edge("classify", "resolve")
    graph.add_edge("resolve", "route")
    graph.add_conditional_edges("route", dispatch_edge,
                                {"vector_rag": "vector_rag", "graph_rag": "graph_rag",
                                 "thematic": "thematic", "comparative": "comparative"})
    for agent in ("vector_rag", "graph_rag", "thematic", "comparative", "fallback"):
        graph.add_edge(agent, "synthesize")
    graph.add_conditional_edges("synthesize", post_synthesis_edge,
                                {"done": END, "vector_rag": "vector_rag", "graph_rag": "graph_rag",
                                 "thematic": "thematic", "comparative": "comparative", "fallback": "fallback"})

    return graph.compile(checkpointer=MemorySaver())
