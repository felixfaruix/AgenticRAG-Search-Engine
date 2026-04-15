"""Read scratchpad: retrieve an agent's private retrieval history from Supermemory."""

from supermemory import Supermemory
from src.models.agent_contracts import ScratchpadEntry
from src.session import scratchpad_container


def read_scratchpad(agent_type: str, session_id: str, sm_client: Supermemory, query: str | None = None) -> list[ScratchpadEntry]:
    """Read ScratchpadEntry entries from an agent's private container."""
    container: str = scratchpad_container(agent_type, session_id)
    results = sm_client.search.execute(q=query or f"retrieval attempts by {agent_type}", container_tags=[container], limit=50)
    entries: list[ScratchpadEntry] = []
    for r in results.results:
        entry_json: str | None = (r.metadata or {}).get("entry_json")
        if not entry_json or not isinstance(entry_json, str):
            continue
        try:
            entries.append(ScratchpadEntry.model_validate_json(entry_json))
        except Exception:
            continue
    return entries
