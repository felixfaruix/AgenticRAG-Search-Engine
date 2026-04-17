"""upload contextual dense embeddings to qdrant.

bm25 keyword search runs in-memory via rank_bm25 (see src/tools/vector_search.py).
qdrant stores only the dense contextual embeddings for semantic search.
"""

import json
import os
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

project_root: Path = Path(__file__).resolve().parent.parent.parent
data_dir: Path = project_root / "data"
collection_name: str = "literary_chunks"
batch_size: int = 100


def load_env() -> dict[str, str]:
    """load qdrant credentials from .env."""
    load_dotenv(project_root / ".env")
    return {"url": os.environ["qdrant_url"], "api_key": os.environ["qdrant_api_key"]}


def load_chunks() -> list[dict]:
    """load recursive chunks aligned with the embedding matrix."""
    with open(data_dir / "chunks" / "recursive_chunks.json") as f:
        return json.load(f)


def load_embeddings() -> np.ndarray:
    """load contextual embeddings (n_chunks x 768, float32)."""
    return np.load(data_dir / "embeddings" / "contextual_embeddings.npy")


def upload(client: QdrantClient, chunks: list[dict], embeddings: np.ndarray) -> None:
    """upload all points to qdrant in batches. dense vectors only."""
    total: int = len(chunks)

    for start in range(0, total, batch_size):
        end: int = min(start + batch_size, total)
        points: list[models.PointStruct] = []

        for i in range(start, end):
            c: dict = chunks[i]
            points.append(models.PointStruct(
                id=i,
                vector={"contextual": embeddings[i].tolist()},
                payload={k: c[k] for k in ("book_id", "book_title", "author", "chapter_number",
                                           "chapter_title", "chunk_index", "total_chunks_in_chapter", "text")}))

        client.upsert(collection_name=collection_name, points=points)


def main() -> None:
    """load data, upload to qdrant."""
    env: dict[str, str] = load_env()
    client: QdrantClient = QdrantClient(url=env["url"], api_key=env["api_key"])
    chunks: list[dict] = load_chunks()
    embeddings: np.ndarray = load_embeddings()

    assert len(chunks) == embeddings.shape[0], f"mismatch: {len(chunks)} chunks vs {embeddings.shape[0]} embeddings"
    upload(client, chunks, embeddings)


if __name__ == "__main__":
    main()
