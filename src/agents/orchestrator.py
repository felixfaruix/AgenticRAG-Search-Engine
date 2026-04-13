"""Orchestrator: LangGraph state graph wiring all agents into a single retrieval pipeline.
Defines the graph state schema, node functions, routing edges, the grounding retry loop,
and human-in-the-loop disambiguation. All resources (indexes, models, clients) are captured
in closures by build_graph so node functions match LangGraph's (state) -> dict signature.
"""
from typing import Any, TypedDict
import instructor
import numpy as np
from rank_bm25 import BM25Okapi
from vertexai.language_models import TextEmbeddingModel
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from src.models.agent_contracts import (QueryUnderstanding, ResolvedEntity, AgentResult,
                                        SynthesizedAnswer, ScratchpadEntry)
from src.session import create_session_id
from src.tools.resolve_entities import resolve_entities
from src.tools.route_query import route_query
from src.tools.write_scratchpad import write_scratchpad
from src.agents.intent_classifier import classify_query
from src.agents.vector_rag_agent import run_vector_rag
from src.agents.graph_rag_agent import run_graph_rag
from src.agents.thematic_agent import run_thematic
from src.agents.comparative_agent import run_comparative
from src.agents.synthesis_agent import run_synthesis

disambiguation_threshold: float = 0.5

class OrchestratorState(TypedDict):
    """LangGraph state schema carrying all data between nodes.
    """
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

def initial_state(query: str) -> OrchestratorState:
    """Create the initial state dict for a new query. Called once per user query.
    """
    return OrchestratorState(query=query, session_id=create_session_id(), understanding=None,
                             resolved_entities=[], routed_agent="", agent_results=[],
                             synthesized_answer=None, attempt_number=1, max_attempts=3,
                             grounding_feedback=None)

def build_graph(chunks: list[Any], enriched_chunks: list[Any], bm25_index: BM25Okapi,
                embeddings: np.ndarray, embedding_model: TextEmbeddingModel,
                summaries: list[dict[str, Any]], summary_embeddings: np.ndarray,
                sm_client: Any, alias_index: dict[str, list[dict[str, Any]]],
                model: str, client: instructor.Instructor) -> Any:
    """Build and compile the orchestrator state graph with checkpointing.
    All resources are captured in closures so node functions match LangGraph's (state) -> dict signature.
    Returns a compiled LangGraph StateGraph ready for .invoke() or .stream().
    """

    def classify_node(state: OrchestratorState) -> dict[str, Any]:
        """Classify user query into structured intent fields via single LLM call.
        """
        understanding: QueryUnderstanding = classify_query(state["query"], model, client)
        return {"understanding": understanding}

    def resolve_node(state: OrchestratorState) -> dict[str, Any]:
        """Resolve raw entity mentions to canonical graph nodes via RapidFuzz.
        Triggers human-in-the-loop interrupt when any entity confidence falls below threshold.
        """
        understanding: QueryUnderstanding | None = state["understanding"]

        if not understanding or not understanding.extracted_entities:
            return {"resolved_entities": []}

        resolved: list[ResolvedEntity] = resolve_entities(understanding.extracted_entities, alias_index)
        low_confidence: list[ResolvedEntity] = [e for e in resolved if e.confidence < disambiguation_threshold]

        if low_confidence:
            clarification: dict[str, Any] = interrupt({"type": "disambiguation",
                "message": "Some entities could not be resolved with high confidence. Please clarify.",
                "ambiguous_entities": [{"mention": e.raw_mention, "best_match": e.canonical_name,
                                        "confidence": e.confidence, "book": e.book_id} for e in low_confidence]})

            if isinstance(clarification, dict) and "confirmed_entities" in clarification:
                resolved = resolve_entities(clarification["confirmed_entities"], alias_index)

        return {"resolved_entities": resolved}

    def route_node(state: OrchestratorState) -> dict[str, Any]:
        """Deterministic routing from intent classification to agent type. Stores routed agent in state.
        """
        understanding: QueryUnderstanding | None = state["understanding"]
        agent: str = route_query(understanding) if understanding else "vector_rag"

        return {"routed_agent": agent}

    def dispatch_edge(state: OrchestratorState) -> str:
        """Read routed_agent from state and dispatch to the corresponding agent node.
        """
        return state["routed_agent"]

    def vector_rag_node(state: OrchestratorState) -> dict[str, Any]:
        """Run vector RAG retrieval and update state with results.
        """
        result: AgentResult = run_vector_rag(state, chunks, enriched_chunks, bm25_index,
                                             embeddings, embedding_model, sm_client)
        return {"agent_results": [result]}

    def graph_rag_node(state: OrchestratorState) -> dict[str, Any]:
        """Run graph RAG traversal and update state with results.
        """
        result: AgentResult = run_graph_rag(state, sm_client)
        return {"agent_results": [result]}

    def thematic_node(state: OrchestratorState) -> dict[str, Any]:
        """Run thematic book-level search and update state with results.
        """
        result: AgentResult = run_thematic(state, summaries, summary_embeddings, embedding_model, sm_client)
        return {"agent_results": [result]}

    def comparative_node(state: OrchestratorState) -> dict[str, Any]:
        """Run comparative cross-book search and update state with results.
        """
        result: AgentResult = run_comparative(state, chunks, enriched_chunks, bm25_index,
                                              embeddings, embedding_model, sm_client)
        return {"agent_results": [result]}

    def synthesize_node(state: OrchestratorState) -> dict[str, Any]:
        """Generate grounded answer, verify, and handle retry state on failure.
        On grounding failure, writes feedback to the originating agent's scratchpad
        and increments attempt_number so the agent adjusts its strategy on retry.
        """
        answer: SynthesizedAnswer = run_synthesis(state, model, client, sm_client)

        if answer.grounding_passed:
            return {"synthesized_answer": answer}

        feedback_entry: ScratchpadEntry = ScratchpadEntry(
            session_id=state["session_id"], agent_type=state["routed_agent"],
            attempt_number=state["attempt_number"], tool_name="grounding_check",
            tool_params={}, passages_returned=len(answer.cited_passages),
            top_score=answer.confidence, success=False,
            grounding_feedback="Grounding failed. Broaden or adjust retrieval strategy.")

        write_scratchpad(feedback_entry, sm_client)

        return {"synthesized_answer": answer, "grounding_feedback": "Grounding failed. Adjust retrieval strategy.",
                "attempt_number": state["attempt_number"] + 1}

    def post_synthesis_edge(state: OrchestratorState) -> str:
        """Route after synthesis: end if grounding passed or max retries exhausted, retry otherwise.
        """
        answer: SynthesizedAnswer | None = state.get("synthesized_answer")
        
        if not answer or answer.grounding_passed:
            return "done"
        if state["attempt_number"] > state["max_attempts"]:
            return "done"
        return state["routed_agent"]

    graph: StateGraph = StateGraph(OrchestratorState)
    graph.add_node("classify", classify_node)
    graph.add_node("resolve", resolve_node)
    graph.add_node("route", route_node)
    graph.add_node("vector_rag", vector_rag_node)
    graph.add_node("graph_rag", graph_rag_node)
    graph.add_node("thematic", thematic_node)
    graph.add_node("comparative", comparative_node)
    graph.add_node("synthesize", synthesize_node)

    graph.add_edge(START, "classify")
    graph.add_edge("classify", "resolve")
    graph.add_edge("resolve", "route")
    graph.add_conditional_edges("route", dispatch_edge,
                                {"vector_rag": "vector_rag", "graph_rag": "graph_rag",
                                 "thematic": "thematic", "comparative": "comparative"})
    graph.add_edge("vector_rag", "synthesize")
    graph.add_edge("graph_rag", "synthesize")
    graph.add_edge("thematic", "synthesize")
    graph.add_edge("comparative", "synthesize")
    graph.add_conditional_edges("synthesize", post_synthesis_edge,
                                {"done": END, "vector_rag": "vector_rag", "graph_rag": "graph_rag",
                                 "thematic": "thematic", "comparative": "comparative"})

    return graph.compile(checkpointer=MemorySaver())
