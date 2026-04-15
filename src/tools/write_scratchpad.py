"""Write scratchpad: persist agent's private retrieval attempts to Supermemory."""

import json
from supermemory import Supermemory
from src.models.agent_contracts import ScratchpadEntry
from src.session import scratchpad_container


def format_attempt_log(entry: ScratchpadEntry) -> str:
    """Build a searchable text summary of one retrieval attempt."""
    lines: list[str] = [
        f"Attempt {entry.attempt_number} by {entry.agent_type}",
        f"Tool: {entry.tool_name}({json.dumps(entry.tool_params, default=str)})",
        f"Result: {entry.passages_returned} passages" + (f", top score {entry.top_score:.3f}" if entry.top_score is not None else ""),
        f"Success: {entry.success}"]
    if entry.grounding_feedback:
        lines.append(f"Grounding feedback: {entry.grounding_feedback}")
    return "\n".join(lines)


def write_scratchpad(entry: ScratchpadEntry, sm_client: Supermemory) -> str:
    """Write a ScratchpadEntry to the agent's private Supermemory container."""
    container: str = scratchpad_container(entry.agent_type, entry.session_id)
    result = sm_client.add(
        content=format_attempt_log(entry), container_tag=container,
        metadata={"session_id": entry.session_id, "agent_type": entry.agent_type,
                  "attempt_number": float(entry.attempt_number), "tool_name": entry.tool_name,
                  "success": entry.success, "passages_returned": float(entry.passages_returned),
                  "entry_json": entry.model_dump_json()})
    return result.id
