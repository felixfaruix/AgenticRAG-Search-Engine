"""Route query: deterministic mapping from intent classification to agent type."""

from src.models.agent_contracts import QueryUnderstanding

routing_table: dict[tuple[str, str], str] = {("single", "passage"): "vector_rag", ("single", "book"): "vector_rag",
                                            ("single", "cross_book"): "comparative", ("single", "exploratory"): "thematic",
                                            ("multi", "passage"): "graph_rag", ("multi", "book"): "graph_rag",
                                            ("multi", "cross_book"): "comparative", ("multi", "exploratory"): "thematic",
                                            ("unknown", "passage"): "vector_rag", ("unknown", "book"): "vector_rag",
                                            ("unknown", "cross_book"): "comparative", ("unknown", "exploratory"): "thematic"}

def route_query(understanding: QueryUnderstanding) -> str:
    """Deterministic routing from hop_count and scope to agent type.
    Returns one of: vector_rag, graph_rag, thematic, comparative.
    Falls back to vector_rag for unrecognized combinations.
    """
    return routing_table.get((understanding.hop_count, understanding.scope), "vector_rag")
