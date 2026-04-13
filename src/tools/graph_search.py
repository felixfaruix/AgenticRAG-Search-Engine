"""Graph search tool: multi-hop traversal over Supermemory knowledge graph."""

from typing import Any
from src.models.agent_contracts import Passage

def graph_search(start_node_ids: list[str], sm_client: Any, book_id: str, relationship_type: str | None = None, 
                max_hops: int = 2, top_k: int = 10) -> list[Passage]:
    """This tool traverses the Supermemory memory graph from start nodes, collecting source passages.
    start_node_ids: canonical entity IDs from ResolvedEntity.canonical_id.
    sm_client: Supermemory client instance.
    book_id: determines the container (book_{book_id}).
    relationship_type: filter edges to this type only (e.g. OCCURS_BEFORE for temporal).
    Each stored memory is a triple edge whose content is the source chunk text.
    """
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

            results = sm_client.memory.search(query=node_id, container=container, top_k=top_k)

            for mem in results.memories:
                meta: dict = mem.metadata or {}
                rel: str = meta.get("relationship", "")

                if relationship_type and rel != relationship_type:
                    continue

                entity_to: str = meta.get("entity_to", "")
                triple_str: str = f"{meta.get('entity_from', '')} -{rel}-> {entity_to}"
                new_path: list[str] = path + [rel, entity_to]
                
                passages.append(Passage(book_id=book_id, book_title=meta.get("book_title", ""),
                                        chapter_number=meta.get("chapter_number", 0),
                                        chapter_title=meta.get("chapter_title", None),
                                        chunk_index=meta.get("chunk_index", 0),
                                        text=mem.content,
                                        score=mem.score,
                                        retrieval_method="graph_traversal",
                                        retrieval_agent="graph_rag",
                                        graph_path=new_path,
                                        source_triple=triple_str))

                if entity_to and entity_to not in visited:
                    next_frontier.append((entity_to, new_path))

        frontier = next_frontier

        if not frontier:
            break
    # deduplicate by (chapter, chunk), keep highest score
    seen: dict[str, Passage] = {}
    for p in passages:
        key: str = f"{p.chapter_number}:{p.chunk_index}"
        if key not in seen or p.score > seen[key].score:
            seen[key] = p

    deduped: list[Passage] = sorted(seen.values(), key=lambda p: p.score, reverse=True)

    return deduped[:top_k]
