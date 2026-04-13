"""Read agent results: retrieve another agent's shared results from Supermemory.
Searches the shared results container by semantic relevance (query provided) or
lists all entries (no query). Results are reconstructed from the entry_json stored
in metadata by write_results.
"""
from typing import Any
from src.models.agent_contracts import SharedResultEntry
from src.session import results_container

def read_agent_results(agent_type: str, session_id: str, sm_client: Any, query: str | None = None) -> list[SharedResultEntry]:
    """Read SharedResultEntry entries from another agent's shared container.
    If query is provided, search by semantic relevance against passage texts.
    Otherwise list all entries for that session.
    """
    container: str = results_container(agent_type, session_id)

    if query:
        results = sm_client.memory.search(query=query, container=container, top_k=50)
    else:
        results = sm_client.memory.list(container=container)

    entries: list[SharedResultEntry] = []
    for mem in results.memories:
        meta: dict = mem.metadata or {}
        entry_json: str | None = meta.get("entry_json")
        if not entry_json:
            continue
        try:
            entries.append(SharedResultEntry.model_validate_json(entry_json))
        except Exception:
            continue
    return entries
