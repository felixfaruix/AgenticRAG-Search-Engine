"""Resolve entities: fuzzy match raw query mentions against the alias index."""
from typing import Any
from rapidfuzz import fuzz
from src.models.agent_contracts import ResolvedEntity

def resolve_entities(raw_mentions: list[str], alias_index: dict[str, list[dict[str, Any]]], book_ids_hint: list[str] | None = None,
                    threshold: float = 0.7) -> list[ResolvedEntity]:
    """Matches raw query mentions to canonical entities via RapidFuzz weighted ratio.
    alias_index: {book_id: [{canonical_name, canonical_id, entity_type, aliases: [str]}]}.
    book_ids_hint: narrow search to these books when co-occurrence resolves scope.
    threshold: minimum normalized score (0.0-1.0) to accept a match.
    """
    books: list[str] = book_ids_hint if book_ids_hint else list(alias_index.keys())
    resolved: list[ResolvedEntity] = []

    for mention in raw_mentions:
        best_score: float = 0.0
        best_match: ResolvedEntity | None = None

        for bid in books:
            for entity in alias_index.get(bid, []):
                candidates: list[str] = [entity["canonical_name"]] + entity.get("aliases", [])
                for alias in candidates:
                    score: float = fuzz.WRatio(mention.lower(), alias.lower()) / 100.0
                    if score > best_score and score >= threshold:
                        best_score = score
                        best_match = ResolvedEntity(raw_mention=mention, canonical_name=entity["canonical_name"],
                                                    canonical_id=entity["canonical_id"], entity_type=entity["entity_type"],
                                                    book_id=bid, confidence=score)
        if best_match:
            resolved.append(best_match)

    resolved.sort(key=lambda e: e.confidence, reverse=True)
    return resolved



