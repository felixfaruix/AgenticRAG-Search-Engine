"""scratchpad io: persist agent's private retrieval attempts to supermemory
and read them back on retry so the agent can adapt to what it already tried.
"""

import json
from supermemory import Supermemory
from src.models.agent_contracts import ScratchpadEntry
from src.session import scratchpad_container


def format_attempt_log(entry: ScratchpadEntry) -> str:
    """build a searchable text summary of one retrieval attempt."""
    lines: list[str] = [
        f"Attempt {entry.attempt_number} by {entry.agent_type}",
        f"Tool: {entry.tool_name}({json.dumps(entry.tool_params, default=str)})",
        f"Result: {entry.passages_returned} passages" + (f", top score {entry.top_score:.3f}" if entry.top_score is not None else ""),
        f"Success: {entry.success}"]
    if entry.grounding_feedback:
        lines.append(f"Grounding feedback: {entry.grounding_feedback}")
    return "\n".join(lines)


def write_scratchpad(entry: ScratchpadEntry, sm_client: Supermemory) -> str:
    """write a scratchpadentry to the agent's private supermemory container."""
    container: str = scratchpad_container(entry.agent_type, entry.session_id)
    result = sm_client.add(
        content=format_attempt_log(entry), container_tag=container,
        metadata={"session_id": entry.session_id, "agent_type": entry.agent_type,
                  "attempt_number": float(entry.attempt_number), "tool_name": entry.tool_name,
                  "success": entry.success, "passages_returned": float(entry.passages_returned),
                  "entry_json": entry.model_dump_json()})
    return result.id


def read_scratchpad(agent_type: str, session_id: str, sm_client: Supermemory) -> list[ScratchpadEntry]:
    """read past retrieval attempts for this agent+session ordered by attempt_number.
    called on retry so the agent can see what it already tried and escalate strategy
    instead of blindly progressing on attempt_number alone.
    """
    container: str = scratchpad_container(agent_type, session_id)

    try:
        response = sm_client.search.documents(q=agent_type, container_tag=container, limit=10, rerank=False)
    except Exception:
        return []

    entries: list[ScratchpadEntry] = []

    for r in response.results:
        meta: dict = dict(r.metadata or {})
        payload: str | None = meta.get("entry_json")
        if not payload:
            continue
        try:
            entries.append(ScratchpadEntry.model_validate_json(str(payload)))
        except Exception:
            continue

    entries.sort(key=lambda e: e.attempt_number)
    return entries
