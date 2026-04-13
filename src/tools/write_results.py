"""Write results: persist agent's final retrieval output to Supermemory.
The content is the query text followed by the retrieved passage texts so other
agents and the orchestrator can find relevant prior results via semantic search.
The full structured entry goes into metadata for programmatic reconstruction.
"""
from typing import Any
from src.models.agent_contracts import SharedResultEntry
from src.session import results_container

def format_result_content(entry: SharedResultEntry) -> str:
    """Builds a searchable text from the query and retrieved passages.
    """
    parts: list[str] = [f"Query: {entry.query_text}", ""]
    for i, p in enumerate(entry.passages, 1):
        parts.append(f"[{i}] {p.book_title}, Ch.{p.chapter_number}\n{p.text}")
    return "\n".join(parts)

def write_results(entry: SharedResultEntry, sm_client: Any) -> str:
    """Writes a SharedResultEntry to the agent's shared Supermemory container.
    Content is query + passage texts (semantically searchable by other agents).
    Metadata carries structured fields for filtering and the full entry JSON for reconstruction.
    Returns the Supermemory memory ID.
    """
    container: str = results_container(entry.agent_type, entry.session_id)
    result = sm_client.memory.create(content=format_result_content(entry),
                                    metadata={"session_id": entry.session_id, "agent_type": entry.agent_type,
                                                "query_text": entry.query_text, "confidence": entry.confidence,
                                                "attempt_number": entry.attempt_number, 
                                                "grounding_passed": entry.grounding_passed,
                                                "entry_json": entry.model_dump_json()},
                                    container=container)
    return result.id
