"""Read agent results: retrieve another agent's shared results from Supermemory."""

from supermemory import Supermemory
from src.models.agent_contracts import SharedResultEntry
from src.session import results_container


def read_agent_results(agent_type: str, session_id: str, sm_client: Supermemory, query: str | None = None) -> list[SharedResultEntry]:
    """Read SharedResultEntry entries from another agent's shared container."""
    container: str = results_container(agent_type, session_id)
    results = sm_client.search.execute(q=query or "retrieval results", container_tags=[container], limit=50)
    entries: list[SharedResultEntry] = []
    for r in results.results:
        entry_json: str | None = (r.metadata or {}).get("entry_json")
        if not entry_json or not isinstance(entry_json, str):
            continue
        try:
            entries.append(SharedResultEntry.model_validate_json(entry_json))
        except Exception:
            continue
    return entries
