"""Synthesis agent: generates grounded answers with citations and verifies via grounding check.
Activated after all retrieval agents complete. Collects all AgentResults from state,
generates an answer with inline citations, runs grounding verification, and persists
results to Supermemory on success.
"""
import instructor
from typing import Any
from supermemory import Supermemory
from pydantic import BaseModel, Field
from src.models.agent_contracts import AgentResult, GroundingResult, Passage, SharedResultEntry, SynthesizedAnswer
from src.tools.grounding_check import grounding_check
from src.tools.write_results import write_results

synthesis_prompt: str = """\
You are the synthesis agent for a literary search engine. Generate a comprehensive answer \
to the user's query using ONLY the retrieved passages below.

**User query:** {query}

**Retrieved passages:**
{passages}

Instructions:
- Answer the query using only information from the passages.
- Cite every factual claim with inline citations: [Book Title, Ch.N].
- If the passages do not fully cover the query, state what is missing.
- Do not add information not present in the passages.

Return the answer text, the 1-based indices of passages you cited, and your confidence.
"""


class SynthesisLLMResponse(BaseModel):
    """Internal structured output for the synthesis LLM call. Mapped to SynthesizedAnswer by the agent.
    """
    answer_text: str = Field(description="Generated answer with inline citations")
    cited_passage_indices: list[int] = Field(description="1-based indices of cited passages")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the generated answer")

def format_passages_for_synthesis(passages: list[Passage]) -> str:
    """Format passages with numbered indices, provenance, and scores for the synthesis prompt.
    """
    parts: list[str] = []
    for i, p in enumerate(passages, 1):
        header: str = f"[{i}] {p.book_title}, Ch.{p.chapter_number}"
        if p.chapter_title:
            header += f" ({p.chapter_title})"
        header += f" | {p.retrieval_method} via {p.retrieval_agent} | score: {p.score:.3f}"
        parts.append(f"{header}\n{p.text}")
    return "\n\n".join(parts)

def run_synthesis(state: dict[str, Any], model: str, client: instructor.Instructor,
                  sm_client: Supermemory) -> SynthesizedAnswer:
    """Generate a grounded answer from retrieved passages and verify via grounding check.
    On grounding success, persists each contributing agent's results to Supermemory.
    """
    session_id: str = state["session_id"]
    query: str = state["query"]
    attempt: int = state["attempt_number"]
    agent_results: list[AgentResult] = state.get("agent_results", [])

    all_passages: list[Passage] = []
    agents_used: list[str] = []
    for ar in agent_results:
        all_passages.extend(ar.retrieved_passages)
        if ar.agent_type not in agents_used:
            agents_used.append(ar.agent_type)

    llm_response: SynthesisLLMResponse = client.chat.completions.create(
        model=model, response_model=SynthesisLLMResponse, temperature=0.2, max_retries=2,
        messages=[{"role": "user", "content": synthesis_prompt.format(
            query=query, passages=format_passages_for_synthesis(all_passages))}])

    cited: list[Passage] = [all_passages[i - 1] for i in llm_response.cited_passage_indices
                            if 1 <= i <= len(all_passages)]

    grounding: GroundingResult = grounding_check(llm_response.answer_text, all_passages, query, model, client)

    if grounding.passed:
        for ar in agent_results:
            entry: SharedResultEntry = SharedResultEntry(
                session_id=session_id, agent_type=ar.agent_type, query_text=ar.query_text,
                passages=ar.retrieved_passages, confidence=ar.confidence,
                attempt_number=attempt, grounding_passed=True)
            write_results(entry, sm_client)

    return SynthesizedAnswer(session_id=session_id, answer_text=llm_response.answer_text, cited_passages=cited,
                            confidence=llm_response.confidence, agents_used=agents_used,
                            grounding_passed=grounding.passed, attempt_number=attempt)
