"""
upload raw triples directly to supermemory, skipping validation.
run with: python -m scripts.upload_raw_triples
"""

import json
import time
from pathlib import Path

from dotenv import load_dotenv
from supermemory import Supermemory

from src.config.supermemory_client import get_client, book_container

load_dotenv(Path(".env"))

# graph_search reads book_title from metadata but raw triples only have book_id
# this lookup fills that gap so the passage objects render correctly downstream
book_titles = {
    "alice_in_wonderland": "Alice's Adventures in Wonderland",
    "beowulf": "Beowulf",
    "count_of_monte_cristo": "The Count of Monte Cristo",
    "dracula": "Dracula",
    "frankenstein": "Frankenstein",
    "great_gatsby": "The Great Gatsby",
    "pride_and_prejudice": "Pride and Prejudice",
    "prince": "The Prince",
    "shakespeare_complete": "Shakespeare Complete Works",
}


def store_triple(triple: dict, sm_client: Supermemory, container: str) -> str:
    """
    store a single raw triple as a memory entry in supermemory.
    """
    content = triple["source_text"]
    book_id = triple.get("book_id", "")
    metadata = {
        "entity_from": triple["subject"], "entity_from_type": triple["subject_type"],
        "relationship": triple["predicate"],
        "entity_to": triple["object"], "entity_to_type": triple["object_type"],
        "book_id": book_id, "book_title": book_titles.get(book_id, book_id),
        "chapter_number": float(triple.get("chapter_number", 0)),
        "chunk_index": float(triple.get("chunk_index", 0)),
        "confidence": float(triple.get("confidence", 0.0)),
    }
    result = sm_client.add(content=content, metadata=metadata, container_tags=[container])
    return result.id


def upload_book(book_id: str, sm_client: Supermemory) -> int:
    """
    load raw triples for one book and push them all to supermemory.
    """
    path = Path(f"data/triples/raw/raw_triples_{book_id}.json")
    if not path.exists():
        print(f"  skipping {book_id}: no raw triples file")
        return 0
    triples = json.load(open(path))
    container = book_container(book_id)
    print(f"  {book_id}: uploading {len(triples)} triples to {container}")
    for i, t in enumerate(triples):
        store_triple(t, sm_client, container)
        if (i + 1) % 200 == 0:
            print(f"    {i + 1}/{len(triples)}")
    return len(triples)


def main():
    """
    upload raw triples for all books.
    """
    sm = get_client()
    total = 0
    t0 = time.time()
    for book_id in sorted(book_titles):
        n = upload_book(book_id, sm)
        total += n
    elapsed = time.time() - t0
    print(f"\ndone: {total} triples uploaded in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
