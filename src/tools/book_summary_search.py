"""Book summary search: dense search over book-level summaries for thematic queries."""
from typing import Any
import numpy as np
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from src.models.agent_contracts import Passage

def book_summary_search(query: str, summaries: list[dict[str, Any]], summary_embeddings: np.ndarray, 
                        embedding_model: TextEmbeddingModel, top_k: int = 3) -> list[Passage]:
    """Dense cosine similarity search over book-level summaries.
    summaries: loaded from data/book_summaries.json, each dict has book_key, title, summary.
    summary_embeddings: numpy array of shape (n_books, embedding_dim).
    """
    q_input: TextEmbeddingInput = TextEmbeddingInput(text=query, task_type="RETRIEVAL_QUERY")
    q_vec: np.ndarray = np.array(embedding_model.get_embeddings([q_input])[0].values, dtype=np.float32)
    norms: np.ndarray = np.linalg.norm(summary_embeddings, axis=1) * np.linalg.norm(q_vec)
    scores: np.ndarray = (summary_embeddings @ q_vec) / np.where(norms > 0, norms, 1.0)
    ranked: np.ndarray = np.argsort(scores)[::-1][:top_k]

    return [Passage(book_id=summaries[i]["book_key"], book_title=summaries[i]["title"], chapter_number=0,
                    chunk_index=0, text=summaries[i]["summary"], score=float(scores[i]), 
                    retrieval_method="dense", retrieval_agent="thematic") for i in ranked]
