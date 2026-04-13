"""Grounding check: verify that a generated answer is supported by retrieved passages."""
import instructor
from pydantic import BaseModel, Field
from src.models.agent_contracts import GroundingResult, Passage

graunding_prompt: str = """\
You are a grounding verifier for a literary search engine.

**User query:** {query}

**Generated answer:** {answer}

**Retrieved passages:**
{passages}

Evaluate:
1. Is every factual claim in the answer directly supported by at least one passage?
2. Does the answer address all aspects of the user query?
3. List any query aspects not covered by the passages.

Return:
- passed: true only if every claim is supported AND the query is fully addressed.
- feedback: specific explanation of what passed or failed.
- confidence: your confidence in this evaluation (0.0 to 1.0).
- cited_passage_indices: 1-based indices of passages that support claims in the answer.
- uncovered_aspects: query aspects not addressed (empty list if fully covered).
"""

class GroundingLLMResponse(BaseModel):
    """Internal structured output for the grounding LLM call. Mapped to GroundingResult by the tool.
    """
    passed: bool = Field(description="True only if every claim is supported and query fully addressed")
    feedback: str = Field(description="Specific explanation of what passed or failed")
    confidence: float = Field(ge=0.0, le=1.0, description="Evaluation confidence")
    cited_passage_indices: list[int] = Field(description="1-based indices of passages that support claims")
    uncovered_aspects: list[str] = Field(description="Query aspects not addressed by passages")

def format_passages(passages: list[Passage]) -> str:
    """Format passages with numbered indices for the grounding prompt.
    """
    parts: list[str] = []

    for i, p in enumerate(passages, 1):
        parts.append(f"[{i}] {p.book_title}, Ch.{p.chapter_number}, Chunk {p.chunk_index}\n{p.text}")

    return "\n\n".join(parts)

def grounding_check(answer_text: str, passages: list[Passage], original_query: str, model: str, 
                    client: instructor.Instructor) -> GroundingResult:
    """Verify that answer_text is grounded in the retrieved passages and addresses the query.
    client: instructor.Instructor wrapping an OpenAI-compatible client.
    """
    llm_result: GroundingLLMResponse = client.chat.completions.create(model=model, response_model=GroundingLLMResponse,
                                                                    temperature=0.1,
                                                                    max_retries=2,
                                                                    messages=[{"role": "user", 
                                                                    "content": graunding_prompt.format(
                                                                            query=original_query, answer=answer_text, 
                                                                            passages=format_passages(passages))}])

    cited: list[Passage] = [passages[i - 1] for i in llm_result.cited_passage_indices if 1 <= i <= len(passages)]

    return GroundingResult(passed=llm_result.passed, feedback=llm_result.feedback, confidence=llm_result.confidence, cited_passages=cited,
                            uncovered_aspects=llm_result.uncovered_aspects)
