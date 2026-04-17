"""graph search: multi-hop traversal over supermemory's memory graph.

uses supermemory's search.memories endpoint with a metadata filter on the
canonical name (entity_from) so every hop is a true edge lookup, not a text
search over a container. the earlier implementation searched by text content
and tried to chain by the raw canonical name against a slugified start id,
which never aligned. here the bfs seeds and chains by canonical_name
consistently — the key supermemory already stores on every triple.
"""

from typing import Any
from supermemory import Supermemory
from src.config.supermemory_client import book_container
from src.models.agent_contracts import Passage


def graph_search(start_node_names: list[str], sm_client: Supermemory, book_id: str,
                 relationship_type: str | None = None, max_hops: int = 2, top_k: int = 10,
                 query: str | None = None) -> list[Passage]:
    """bfs from start_node_names over the supermemory memory graph within one book.
    each hop pulls edges whose metadata.entity_from matches the frontier; the
    source chunk attached to the edge becomes a passage, and entity_to seeds
    the next frontier. when query is provided, supermemory ranks the filtered
    set by similarity to the user question instead of to the entity name, and
    the passage score reflects that same similarity (not the extraction-time
    confidence, which is near-constant and useless for cross-book ranking).
    """
    if not start_node_names or not book_id:
        return []

    hops: int = max(1, min(max_hops, 6))
    container: str = book_container(book_id)
    visited: set[str] = set()
    frontier: list[tuple[str, list[str]]] = [(name, [name]) for name in start_node_names]
    passages: list[Passage] = []

    for _ in range(hops):
        pending: list[tuple[str, list[str]]] = [(name, path) for name, path in frontier if name not in visited]
        if not pending:
            break

        for name, _ in pending:
            visited.add(name)

        next_frontier: list[tuple[str, list[str]]] = []

        for start_name, path in pending:
            edges: list[dict[str, Any]] = _fetch_edges(sm_client, container, start_name, relationship_type, top_k, query)

            for edge in edges:
                meta: dict = edge["metadata"]
                predicate: str = str(meta.get("relationship", ""))
                target_name: str = str(meta.get("entity_to", ""))
                source_name: str = str(meta.get("entity_from", ""))
                new_path: list[str] = path + [predicate, target_name]

                passages.append(Passage(
                    book_id=book_id, book_title=str(meta.get("book_title", "")),
                    chapter_number=int(float(meta.get("chapter_number", 0) or 0)),
                    chapter_title=meta.get("chapter_title"),
                    chunk_index=int(float(meta.get("chunk_index", 0) or 0)),
                    text=edge["content"],
                    score=edge["score"],
                    retrieval_method="graph_traversal", retrieval_agent="graph_rag",
                    graph_path=new_path,
                    source_triple=f"{source_name} -{predicate}-> {target_name}"))

                if target_name and target_name not in visited:
                    next_frontier.append((target_name, new_path))

        frontier = next_frontier
        if not frontier:
            break

    seen: dict[str, Passage] = {}
    for p in passages:
        key: str = f"{p.chapter_number}:{p.chunk_index}:{p.source_triple}"
        if key not in seen or p.score > seen[key].score:
            seen[key] = p

    return sorted(seen.values(), key=lambda p: p.score, reverse=True)[:top_k]


def _fetch_edges(sm_client: Supermemory, container: str, entity_from: str,
                 relationship_type: str | None, top_k: int, query: str | None) -> list[dict[str, Any]]:
    """return all edges whose entity_from matches the given canonical name.
    metadata filter gates which edges are eligible; the q string controls
    intra-set ranking. when a user query is available we pass it so the
    returned passages are the ones most relevant to the question, not merely
    the ones closest in text to the entity name.
    """
    conditions: list[dict] = [
        {"key": "entity_from", "value": entity_from, "filterType": "metadata"}]

    if relationship_type:
        conditions.append({"key": "relationship", "value": relationship_type, "filterType": "metadata"})

    filters: dict = {"AND": conditions}

    response = sm_client.search.memories(
        q=query or entity_from, container_tag=container, filters=filters,
        search_mode="memories", limit=max(top_k * 3, 20), rerank=False)

    edges: list[dict[str, Any]] = []

    for r in response.results:
        meta: dict = dict(r.metadata or {})
        content: str = r.memory or (r.chunk or "")
        if not content:
            continue

        edges.append({"metadata": meta, "content": content, "score": float(r.similarity or 0.0)})

    return edges
