"""streamlit demo for the full agentic rag pipeline.
run with: streamlit run app.py

tab 1 runs the orchestrator end-to-end: classify, resolve entities, route, retrieve,
synthesize with grounding verification. tabs 2 and 3 keep the retrieval ablation
views (single query, batch evaluation) that compare bm25, dense, and hybrid search
against the competency questions.

set LANGSMITH_TRACING=true and LANGSMITH_API_KEY in .env to stream langgraph spans
to langsmith automatically.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import streamlit as st
import vertexai
import instructor
from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from supermemory import Supermemory
from vertexai.language_models import TextEmbeddingModel

from src.agents.orchestrator import build_graph, initial_state
from src.config import get_classifier_model, synthesis_model as default_synthesis_model
from src.models.agent_contracts import Passage, SynthesizedAnswer
from src.tools.vector_search import bm25_search, dense_search, hybrid_search

load_dotenv(Path(".env"))


class GenaiInstructorAdapter:
    """wrap instructor.from_genai so temperature and max_retries behave like openai-style kwargs."""

    def __init__(self, inner: instructor.Instructor) -> None:
        self.inner: instructor.Instructor = inner
        self.chat: Any = type("Chat", (), {"completions": self})()

    def create(self, **kwargs: Any) -> Any:
        """move temperature into genai config and forward."""
        temperature: float | None = kwargs.pop("temperature", None)
        kwargs.pop("max_retries", None)
        if temperature is not None:
            kwargs["config"] = types.GenerateContentConfig(temperature=temperature)
        return self.inner.chat.completions.create(**kwargs)


class ResilientSupermemory:
    """pass-through wrapper that skips writes on rate-limit / auth errors."""

    def __init__(self, inner: Supermemory) -> None:
        self.inner: Supermemory = inner
        self.search: Any = inner.search

    def add(self, **kwargs: Any) -> Any:
        try:
            return self.inner.add(**kwargs)
        except Exception as e:
            stub: Any = type("StubResult", (), {"id": f"skipped_{type(e).__name__}"})()
            return stub


@st.cache_resource
def init_clients() -> tuple[TextEmbeddingModel, QdrantClient, ResilientSupermemory, Any]:
    """initialize vertex, qdrant, supermemory, and the instructor adapter once."""
    vertexai.init(project=os.environ["gcp_project"], location=os.environ["gcp_location"])
    emb: TextEmbeddingModel = TextEmbeddingModel.from_pretrained("text-embedding-005")
    qd: QdrantClient = QdrantClient(url=os.environ["qdrant_url"], api_key=os.environ["qdrant_api_key"], timeout=60)
    sm: ResilientSupermemory = ResilientSupermemory(Supermemory(api_key=os.environ["sup_key"]))

    genai_client: genai.Client = genai.Client(
        vertexai=True, project=os.environ["gcp_project"], location=os.environ["gcp_location"])
    instr: GenaiInstructorAdapter = GenaiInstructorAdapter(instructor.from_genai(genai_client))

    return emb, qd, sm, instr


@st.cache_resource
def load_bm25_index() -> tuple[BM25Okapi, list[dict]]:
    """load recursive chunks and build the bm25 index in memory."""
    with open("data/chunks/recursive_chunks.json") as f:
        c: list[dict] = json.load(f)
    tokens: list[list[str]] = [chunk["text"].lower().split() for chunk in c]
    return BM25Okapi(tokens), c


@st.cache_resource
def load_summaries() -> tuple[list[dict], np.ndarray]:
    """load book summaries and their precomputed embeddings for the thematic agent."""
    with open("data/book_summaries.json") as f:
        s: list[dict] = json.load(f)
    embs: np.ndarray = np.array([x["embedding"] for x in s], dtype=np.float32)
    return s, embs


@st.cache_resource
def load_alias_index() -> dict[str, list[dict[str, Any]]]:
    """load per-book alias files, slugifying canonical_ids to match upload/query convention."""
    alias_dir: Path = Path("data/aliases")
    index: dict[str, list[dict[str, Any]]] = {}

    for path in sorted(alias_dir.glob("aliases_*.json")):
        book_id: str = path.stem.replace("aliases_", "")
        with open(path) as f:
            entities: list[dict] = json.load(f)
        for entity in entities:
            if "canonical_id" not in entity:
                slug: str = re.sub(r"[^a-z0-9]+", "_", entity["canonical_name"].lower()).strip("_")
                entity["canonical_id"] = f"{book_id}_{slug}"
        index[book_id] = entities
    return index


@st.cache_data
def load_competency_questions() -> list[dict]:
    """load the competency question set, excluding books without aliases uploaded."""
    with open("data/competency_questions.json") as f:
        cqs: list[dict] = json.load(f)
    excluded: set[str] = {"pride_and_prejudice"}
    return [cq for cq in cqs if not any(b in excluded for b in cq["example_books"])]


@st.cache_resource
def build_orchestrator() -> Any:
    """compile the langgraph orchestrator once per session."""
    emb, qd, sm, instr = init_clients()
    bm25_index, chunks = load_bm25_index()
    summaries, summary_embeddings = load_summaries()
    alias_index: dict = load_alias_index()
    model_name: str = os.environ.get("SYNTHESIS_MODEL", default_synthesis_model)

    return build_graph(
        qdrant_client=qd, collection_name="literary_chunks",
        embedding_model=emb, bm25_index=bm25_index, chunks=chunks,
        summaries=summaries, summary_embeddings=summary_embeddings,
        sm_client=sm, alias_index=alias_index,
        classifier_model=get_classifier_model(), classifier_client=instr,
        synthesis_model=model_name, synthesis_client=instr)


def run_orchestrator(query: str) -> dict:
    """invoke the compiled orchestrator on one query and return the final state."""
    graph = build_orchestrator()
    state: dict = initial_state(query)
    config: dict = {"configurable": {"thread_id": f"ui_{int(time.time() * 1000)}"}}
    return graph.invoke(state, config)


def passages_to_df(passages: list[Passage], method: str) -> pd.DataFrame:
    """flatten passages into a table for display."""
    return pd.DataFrame([
        {"rank": i + 1, "method": method, "book": p.book_id, "ch": p.chapter_number,
         "score": round(p.score, 4), "text": p.text[:200].replace("\n", " ")}
        for i, p in enumerate(passages)])


# initialize cached resources
embedding_model, qdrant, _supermemory, _instr_client = init_clients()
bm25_index, chunks = load_bm25_index()
book_summaries, summary_embeddings = load_summaries()
competency_questions: list[dict] = load_competency_questions()
collection: str = "literary_chunks"


def run_bm25(query: str, book_id: str | None = None, top_k: int = 5) -> list[Passage]:
    """in-memory bm25 keyword search."""
    return bm25_search(query, bm25_index, chunks, book_id=book_id, top_k=top_k)


def run_dense(query: str, book_id: str | None = None, top_k: int = 5) -> list[Passage]:
    """dense cosine similarity via qdrant."""
    return dense_search(query, qdrant, collection, embedding_model, book_id=book_id, top_k=top_k)


def run_hybrid(query: str, book_id: str | None = None, top_k: int = 5) -> list[Passage]:
    """bm25 + dense fused via reciprocal rank fusion."""
    return hybrid_search(query, bm25_index, chunks, qdrant, collection, embedding_model, book_id=book_id, top_k=top_k)


search_fns: dict[str, Any] = {"bm25": run_bm25, "dense": run_dense, "hybrid": run_hybrid}

st.set_page_config(page_title="agentic rag", layout="wide")
st.title("agentic rag over 9 literary texts")
st.caption("tab 1 runs the full langgraph orchestrator. tabs 2 and 3 are the retrieval ablation views.")

tab_agent, tab_retrieval, tab_batch = st.tabs(["agentic pipeline", "retrieval comparison", "batch evaluation"])


with tab_agent:
    st.subheader("full pipeline: classify -> resolve -> route -> retrieve -> synthesize")

    default_query: str = "Who served as a mentor to Edmond Dantès during his time in prison?"
    agent_query: str = st.text_input("query", value=default_query, key="agent_query")

    if st.button("run pipeline", key="run_agent"):
        with st.spinner("orchestrator running..."):
            t0: float = time.time()
            result: dict = run_orchestrator(agent_query)
            elapsed_ms: int = round((time.time() - t0) * 1000)

        answer: SynthesizedAnswer | None = result.get("synthesized_answer")

        st.markdown(f"**latency:** {elapsed_ms} ms  |  **routed agent:** `{result.get('routed_agent')}`"
                    f"  |  **attempts:** {answer.attempt_number if answer else 0}"
                    f"  |  **grounded:** {answer.grounding_passed if answer else False}")

        col_ans, col_meta = st.columns([3, 2])

        with col_ans:
            st.markdown("### answer")
            st.write(answer.answer_text if answer else "no answer produced")

            if answer and answer.cited_passages:
                st.markdown("### cited passages")
                st.dataframe(passages_to_df(answer.cited_passages, "cited"), use_container_width=True, hide_index=True)

        with col_meta:
            st.markdown("### query understanding")
            understanding = result.get("understanding")
            if understanding:
                st.json({
                    "hop_count": understanding.hop_count, "scope": understanding.scope,
                    "query_type": understanding.query_type,
                    "sub_classification": understanding.sub_classification,
                    "extracted_entities": understanding.extracted_entities,
                    "confidence": round(understanding.confidence, 3)})

            st.markdown("### resolved entities")
            entities: list = result.get("resolved_entities", [])
            if entities:
                st.dataframe(pd.DataFrame([{
                    "raw": e.raw_mention, "canonical": e.canonical_name,
                    "id": e.canonical_id, "book": e.book_id,
                    "confidence": round(e.confidence, 3)} for e in entities]),
                    use_container_width=True, hide_index=True)
            else:
                st.info("no entities extracted")

            st.markdown("### agents used")
            if answer:
                st.write(", ".join(answer.agents_used) or "—")


with tab_retrieval:
    st.subheader("compare bm25, dense, and hybrid on a single query")

    col_q, col_opts = st.columns([3, 1])
    with col_q:
        query: str = st.text_input("query", value="Victor Frankenstein creates the creature in his laboratory", key="ret_query")
    with col_opts:
        top_k: int = st.slider("top k", min_value=1, max_value=20, value=5)

    if st.button("search", key="single_search"):
        results: dict = {}
        timings: dict = {}
        for name, fn in search_fns.items():
            t0 = time.time()
            results[name] = fn(query, top_k=top_k)
            timings[name] = round((time.time() - t0) * 1000)

        timing_text: str = " | ".join([f"**{k}**: {v}ms" for k, v in timings.items()])
        st.markdown(timing_text)

        col_bm25, col_dense, col_hybrid = st.columns(3)
        for col, name in zip([col_bm25, col_dense, col_hybrid], ["bm25", "dense", "hybrid"]):
            with col:
                st.markdown(f"### {name}")
                if results[name]:
                    st.dataframe(passages_to_df(results[name], name), use_container_width=True, hide_index=True)
                else:
                    st.info("no results")


with tab_batch:
    st.subheader("batch evaluation on competency questions")

    categories: list[str] = sorted(set(cq["category"] for cq in competency_questions))
    selected_cats: list[str] = st.multiselect("categories", categories, default=categories)
    sample_n: int = st.number_input("sample per category (0 = all)", min_value=0, max_value=28, value=3)

    if st.button("run evaluation", key="batch_eval"):
        filtered: list[dict] = [cq for cq in competency_questions if cq["category"] in selected_cats]
        if sample_n > 0:
            import random
            random.seed(42)
            by_cat: dict[str, list[dict]] = {}
            for cq in filtered:
                by_cat.setdefault(cq["category"], []).append(cq)
            eval_set: list[dict] = []
            for cat in sorted(by_cat):
                eval_set.extend(random.sample(by_cat[cat], min(sample_n, len(by_cat[cat]))))
        else:
            eval_set = filtered

        progress = st.progress(0, text="evaluating...")
        rows: list[dict] = []

        for i, cq in enumerate(eval_set):
            q: str = cq["question"]
            expected: list[str] = cq["example_books"]
            row: dict = {"id": cq["id"], "category": cq["category"], "question": q[:80], "expected_book": expected[0]}

            for name, fn in search_fns.items():
                t0 = time.time()
                passages: list[Passage] = fn(q, top_k=5)
                ms: int = round((time.time() - t0) * 1000)
                row[f"{name}_hit"] = passages[0].book_id in expected if passages else False
                row[f"{name}_score"] = round(passages[0].score, 4) if passages else 0.0
                row[f"{name}_ms"] = ms

            rows.append(row)
            progress.progress((i + 1) / len(eval_set), text=f"evaluated {i + 1}/{len(eval_set)}")

        progress.empty()
        df: pd.DataFrame = pd.DataFrame(rows)

        st.markdown("### per query results")
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("### aggregate metrics")
        agg_rows: list[dict] = []
        for method in ["bm25", "dense", "hybrid"]:
            agg_rows.append({
                "method": method,
                "book_accuracy": f"{df[f'{method}_hit'].mean():.0%}",
                "mean_score": round(df[f"{method}_score"].mean(), 4),
                "mean_latency_ms": round(df[f"{method}_ms"].mean())})
        st.dataframe(pd.DataFrame(agg_rows), use_container_width=True, hide_index=True)
