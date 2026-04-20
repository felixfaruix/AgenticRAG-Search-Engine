"""cross-encoder reranker: second-stage precision reranking over retriever output.

bi-encoder retrieval (dense vectors, bm25) scores query and passage from
independently computed representations, which is fast but blind to fine-grained
joint signal. the cross-encoder reads [query, passage] together through a single
transformer pass, so self-attention runs across both token streams and the output
score reflects 'does this passage actually contain the answer' rather than
'is this passage semantically adjacent to the question'.

used as a langgraph node between each retrieval agent and the synthesis agent.
the agent returns a broad top-k candidate set; this module narrows it to the
top_n highest-precision passages.

contextual header parity. the dense index was embedded from
contextual_header + body but qdrant stores body only, so by default the
reranker sees strictly less signal than the embedder did. passing chunks_index
restores that signal: the header is prepended for the cross-encoder input
only, while the returned Passage.text stays body-only so downstream synthesis
and grounding read the same text they always have.
"""

from sentence_transformers import CrossEncoder
from src.models.agent_contracts import Passage

reranker_model: str = "BAAI/bge-reranker-base"
model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    """lazy-load the cross-encoder so import stays cheap and tests that never
    call rerank_passages don't pay the model-load cost.
    """
    global model
    if model is None:
        model = CrossEncoder(reranker_model)
    return model


def _rerank_text(passage: Passage, chunks_index: dict | None) -> str:
    """text fed to the cross-encoder. when chunks_index carries a contextual_header
    for this (book_id, chapter_number, chunk_index), the header is prepended so
    the reranker sees the same context the embedder did. otherwise falls back to
    passage body only — same behavior as before.
    """
    if not chunks_index:
        return passage.text
    chunk: dict | None = chunks_index.get((passage.book_id, passage.chapter_number, passage.chunk_index))
    header: str = (chunk or {}).get("contextual_header", "") if chunk else ""
    return f"{header}\n\n{passage.text}" if header else passage.text


def rerank_passages(query: str, passages: list[Passage], top_n: int = 3,
                    chunks_index: dict | None = None) -> list[Passage]:
    """score each (query, passage) pair with the cross-encoder and return the top_n.
    when the retriever already returned <= top_n passages the model load is
    skipped: there's nothing to filter, and ordering a handful of items is not
    worth loading a 278m-parameter model for.
    chunks_index: optional {(book_id, chapter_number, chunk_index): chunk_dict}
    whose entries may carry a 'contextual_header' field. when provided, the
    header is prepended to the cross-encoder input to close the dense-index
    information gap. Passage.text on the returned passages is unchanged.
    """
    if len(passages) <= top_n:
        return passages

    model: CrossEncoder = _get_model()
    pairs: list[list[str]] = [[query, _rerank_text(p, chunks_index)] for p in passages]
    scores: list[float] = model.predict(pairs).tolist()
    ranked: list[tuple[float, Passage]] = sorted(zip(scores, passages), key=lambda x: x[0], reverse=True)

    return [p.model_copy(update={"score": float(s)}) for s, p in ranked[:top_n]]
