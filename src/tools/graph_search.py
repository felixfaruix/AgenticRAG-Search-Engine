"""Graph search tool: multi-hop traversal over Supermemory knowledge graph."""

from supermemory import Supermemory
from src.models.agent_contracts import Passage


def graph_search(start_node_ids: list[str], sm_client: Supermemory, book_id: str,
                 relationship_type: str | None = None, max_hops: int = 2, top_k: int = 10) -> list[Passage]:
    """Traverse Supermemory memory graph from start nodes, collecting source passages."""
    container: str = f"book_{book_id}"
    visited: set[str] = set()
    passages: list[Passage] = []
    frontier: list[tuple[str, list[str]]] = [(nid, [nid]) for nid in start_node_ids]

    for hop in range(max_hops):
        next_frontier: list[tuple[str, list[str]]] = []
        for node_id, path in frontier:
            if node_id in visited:
                continue
            visited.add(node_id)
            results = sm_client.search.execute(q=node_id, container_tags=[container], limit=top_k)
            for r in results.results:
                meta: dict = r.metadata or {}
                rel: str = str(meta.get("relationship", ""))
                if relationship_type and rel != relationship_type:
                    continue
                entity_to: str = str(meta.get("entity_to", ""))
                new_path: list[str] = path + [rel, entity_to]
                passages.append(Passage(
                    book_id=book_id, book_title=str(meta.get("book_title", "")),
                    chapter_number=int(meta.get("chapter_number", 0)),
                    chapter_title=meta.get("chapter_title"), chunk_index=int(meta.get("chunk_index", 0)),
                    text=r.content or "", score=r.score, retrieval_method="graph_traversal",
                    retrieval_agent="graph_rag", graph_path=new_path,
                    source_triple=f"{meta.get('entity_from', '')} -{rel}-> {entity_to}"))
                if entity_to and entity_to not in visited:
                    next_frontier.append((entity_to, new_path))
        frontier = next_frontier
        if not frontier:
            break

    seen: dict[str, Passage] = {}
    for p in passages:
        key: str = f"{p.chapter_number}:{p.chunk_index}"
        if key not in seen or p.score > seen[key].score:
            seen[key] = p
    return sorted(seen.values(), key=lambda p: p.score, reverse=True)[:top_k]
