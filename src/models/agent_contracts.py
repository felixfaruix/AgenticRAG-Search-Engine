"""
Data contracts for the agent layer.

These models define what flows between the orchestrator, retrieval agents,
and synthesis agent. LangGraph manages orchestration state, checkpointing,
and inter-agent transitions. These models define the data shapes, not the
control flow.

Overlaps with ingestion models:
  - Passage carries fields similar to Chunk.
    but adds retrieval provenance (score, method, agent, graph_path).
  - ResolvedEntity here is the query-time resolution output; the ingestion.
    ResolvedEntity (extraction.ipynb) tracks merged_names from offline.
    entity resolution. Different lifecycle, different contract.
  - source_triple on Passage references the Triple schema from ingestion.
"""
from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field

class QueryUnderstanding(BaseModel):
    """Intent classifier output. Produced by a lightweight model via Instructor.
    """
    original_query: str = Field(description="Raw user query before any processing")
    hop_count: Literal["single", "multi", "unknown"] = Field(description="Estimated retrieval hops needed")
    scope: Literal["passage", "book", "cross_book", "exploratory"] = Field(description="Breadth of the answer")
    query_type: Literal["factual", "fuzzy_recall", "relational", "temporal", "comparative", "thematic", "exploratory"] = Field(description="Semantic category of the query")
    sub_classification: Literal["factual", "fuzzy", "mixed"] = Field(description="Vector agent sub-routing hint")
    extracted_entities: list[str] = Field(description="Raw entity mentions before resolution")
    confidence: float = Field(ge=0.0, le=1.0, description="Classifier self-reported confidence")
    requires_decomposition: bool = Field(description="True when the query should be split into sub-queries")

class ResolvedEntity(BaseModel):
    """Query-time entity resolution output from the orchestrator via RapidFuzz.
    """
    raw_mention: str = Field(description="Surface form extracted from the query")
    canonical_name: str = Field(description="Canonical entity name after resolution")
    canonical_id: str = Field(description="Node ID in the Supermemory graph")
    entity_type: str = Field(description="Ontology entity type, e.g. Character")
    book_id: str = Field(description="Book container the entity belongs to")
    confidence: float = Field(ge=0.0, le=1.0, description="RapidFuzz match confidence")

class Passage(BaseModel):
    """A retrieved chunk with full provenance for citation and evaluation.
    """
    book_id: str = Field(description="Book container key, e.g. frankenstein")
    book_title: str = Field(description="Human-readable book title")
    chapter_number: int = Field(description="1-based chapter number")
    chapter_title: str | None = Field(default=None, description="Chapter heading if available")
    chunk_index: int = Field(description="Position of this chunk within the chapter")
    text: str = Field(description="Full passage text")
    score: float = Field(description="Retrieval relevance score")
    retrieval_method: Literal["bm25", "dense", "hybrid", "graph_traversal"] = Field(description="How this passage was retrieved")
    retrieval_agent: Literal["vector_rag", "graph_rag", "thematic", "comparative"] = Field(description="Which agent retrieved this passage")
    graph_path: list[str] | None = Field(default=None, description="Traversal path for graph-retrieved passages")
    source_triple: str | None = Field(default=None, description="Serialized triple that led to this passage")

class AgentResult(BaseModel):
    """Standardized return value from every retrieval agent.
    """
    session_id: str = Field(description="Session identifier for tracing")
    agent_type: str = Field(description="Agent that produced this result")
    query_text: str = Field(description="Query or sub-query the agent processed")
    retrieved_passages: list[Passage] = Field(description="Passages retrieved for this query")
    identified_books: list[str] = Field(description="Book IDs the agent determined as relevant")
    confidence: float = Field(ge=0.0, le=1.0, description="Agent self-reported confidence")
    tool_calls_made: list[dict[str, Any]] = Field(description="Logged tool invocations for observability")

class GroundingResult(BaseModel):
    """Output of the synthesis agent's grounding verification check.
    """
    passed: bool = Field(description="Whether the answer is supported by the passages")
    feedback: str = Field(description="Explanation of what passed or failed")
    confidence: float = Field(ge=0.0, le=1.0, description="Grounding confidence score")
    cited_passages: list[Passage] = Field(description="Passages that support the answer")
    uncovered_aspects: list[str] = Field(description="Query aspects not addressed by retrieved passages")

class SynthesizedAnswer(BaseModel):
    """Final system output returned to the user.
    """
    session_id: str = Field(description="Session identifier for tracing")
    answer_text: str = Field(description="Generated answer with inline citations")
    cited_passages: list[Passage] = Field(description="All passages cited in the answer")
    confidence: float = Field(ge=0.0, le=1.0, description="Overall answer confidence")
    agents_used: list[str] = Field(description="Agent types that contributed passages")
    grounding_passed: bool = Field(description="Whether the grounding check passed")
    attempt_number: int = Field(ge=1, description="1-indexed attempt after grounding retries")

class ScratchpadEntry(BaseModel):
    """Written to the agent's private Supermemory container, one per retrieval attempt. Not read by other agents.
    """
    session_id: str = Field(description="Session identifier for tracing")
    agent_type: str = Field(description="Agent that owns this scratchpad")
    attempt_number: int = Field(ge=1, description="1-indexed attempt number")
    tool_name: str = Field(description="Tool invoked in this attempt")
    tool_params: dict[str, Any] = Field(description="Parameters passed to the tool")
    passages_returned: int = Field(ge=0, description="Number of passages the tool returned")
    top_score: float | None = Field(default=None, description="Highest passage score, if available")
    success: bool = Field(description="Whether the tool call succeeded")
    grounding_feedback: str | None = Field(default=None, description="Synthesis feedback that triggered this retry")

class SharedResultEntry(BaseModel):
    """Written to the agent's shared Supermemory container after grounding passes or max retries. Read by the orchestrator and other agents.
    """
    session_id: str = Field(description="Session identifier for tracing")
    agent_type: str = Field(description="Agent that produced this result")
    query_text: str = Field(description="Query or sub-query that was processed")
    passages: list[Passage] = Field(description="Final retrieved passages")
    confidence: float = Field(ge=0.0, le=1.0, description="Agent confidence in these results")
    attempt_number: int = Field(ge=1, description="How many attempts it took")
    grounding_passed: bool = Field(description="Whether synthesis grounding accepted these results")

class ToolCallLog(BaseModel):
    """One row per tool invocation, logged to PostgreSQL via decorator.
    """
    timestamp: datetime = Field(description="UTC timestamp of the tool call")
    session_id: str = Field(description="Session identifier for tracing")
    agent_type: str = Field(description="Agent that invoked the tool")
    tool_name: str = Field(description="Name of the tool function")
    input_params: dict[str, Any] = Field(description="Serialized tool input parameters")
    output_summary: str = Field(description="Brief summary of the tool output")
    output_passage_count: int = Field(ge=0, description="Number of passages in the output")
    top_score: float | None = Field(default=None, description="Highest passage score, if available")
    latency_ms: int = Field(ge=0, description="Wall-clock latency in milliseconds")
    tokens_used: int | None = Field(default=None, description="LLM tokens consumed, if applicable")
    success: bool = Field(description="Whether the tool call completed without error")
    error: str | None = Field(default=None, description="Error message on failure")
    retry_attempt: int = Field(ge=0, description="0 for first attempt, increments on retries")