"""Write results: persist agent's final retrieval output to Supermemory."""

from supermemory import Supermemory
from src.models.agent_contracts import SharedResultEntry
from src.session import results_container


def format_result_content(entry: SharedResultEntry) -> str:
    """Build searchable text from query and retrieved passages."""
    parts: list[str] = [f"Query: {entry.query_text}", ""]
    for i, p in enumerate(entry.passages, 1):
        parts.append(f"[{i}] {p.book_title}, Ch.{p.chapter_number}\n{p.text}")
    return "\n".join(parts)


def write_results(entry: SharedResultEntry, sm_client: Supermemory) -> str:
    """Write a SharedResultEntry to the agent's shared Supermemory container."""
    container: str = results_container(entry.agent_type, entry.session_id)
    result = sm_client.add(
        content=format_result_content(entry), container_tag=container,
        metadata={"session_id": entry.session_id, "agent_type": entry.agent_type,
                  "query_text": entry.query_text, "confidence": entry.confidence,
                  "attempt_number": float(entry.attempt_number),
                  "grounding_passed": entry.grounding_passed,
                  "entry_json": entry.model_dump_json()})
    return result.id
