"""
streamlit demo for retrieval strategy evaluation.
run with: streamlit run app.py
"""

import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import vertexai
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from vertexai.language_models import TextEmbeddingModel
from src.tools.vector_search import bm25_search, dense_search, hybrid_search
from src.tools.book_summary_search import book_summary_search
from src.models.agent_contracts import Passage
load_dotenv(Path(".env"))

@st.cache_resource
def init_clients():
    """
    initialize vertex ai, qdrant, and the embedding model once per session.
    """
    vertexai.init(project=os.environ["gcp_project"], location=os.environ["gcp_location"])
    emb = TextEmbeddingModel.from_pretrained("text-embedding-005")
    qd = QdrantClient(url=os.environ["qdrant_url"], api_key=os.environ["qdrant_api_key"])
    return emb, qd

@st.cache_resource
def load_bm25_index():
    """
    load recursive chunks and build the bm25 index in memory.
    """
    with open("data/chunks/recursive_chunks.json") as f:
        c = json.load(f)
    tokens = [chunk["text"].lower().split() for chunk in c]
    return BM25Okapi(tokens), c

@st.cache_resource
def load_summaries():
    """
    load book summaries and their precomputed embeddings for thematic search.
    """
    with open("data/book_summaries.json") as f:
        s = json.load(f)
    embs = np.array([x["embedding"] for x in s], dtype=np.float32)
    return s, embs

@st.cache_data
def load_competency_questions():
    """
    load the 196 competency questions used as the evaluation set.
    """
    with open("data/competency_questions.json") as f:
        cqs = json.load(f)
    excluded_books = {"pride_and_prejudice"}
    return [cq for cq in cqs if not any(b in excluded_books for b in cq["example_books"])]

embedding_model, qdrant = init_clients()
bm25_index, chunks = load_bm25_index()
summaries, summary_embeddings = load_summaries()
all_cqs = load_competency_questions()
collection = "literary_chunks"


def run_bm25(query, book_id=None, top_k=5):
    """
    run bm25 keyword search over the in memory index.
    """
    return bm25_search(query, bm25_index, chunks, book_id=book_id, top_k=top_k)

def run_dense(query, book_id=None, top_k=5):
    """
    run dense cosine similarity search via qdrant.
    """
    return dense_search(query, qdrant, collection, embedding_model, book_id=book_id, top_k=top_k)

def run_hybrid(query, book_id=None, top_k=5):
    """
    run hybrid search fusing bm25 and dense via reciprocal rank fusion.
    """
    return hybrid_search(query, bm25_index, chunks, qdrant, collection, embedding_model, book_id=book_id, top_k=top_k)

def run_thematic(query, top_k=3):
    """
    run dense search over book level summaries for broad thematic queries.
    """
    return book_summary_search(query, summaries, summary_embeddings, embedding_model, top_k=top_k)

search_fns = {"bm25": run_bm25, "dense": run_dense, "hybrid": run_hybrid}

def passages_to_df(passages, method):
    """
    convert a list of passage objects to a dataframe for display.
    """
    return pd.DataFrame([{"rank": i + 1, "method": method, "book": p.book_id, "ch": p.chapter_number,
                        "score": round(p.score, 4), "text": p.text[:200].replace("\n", " ")} for i, p in enumerate(passages)])

st.set_page_config(page_title="retrieval evaluation", layout="wide")
st.title("retrieval strategy evaluation")
st.caption("compare bm25, dense, hybrid, and thematic search across the literary corpus")

tab_interactive, tab_batch, tab_head_to_head = st.tabs(["interactive search", "batch evaluation", "head to head"])

with tab_interactive:
    st.subheader("try a single query across all methods")

    col_q, col_opts = st.columns([3, 1])
    with col_q:
        query = st.text_input("query", value="Victor Frankenstein creates the creature in his laboratory")
    with col_opts:
        top_k = st.slider("top k", min_value=1, max_value=20, value=5)

    if st.button("search", key="single_search"):
        # running all four methods and show results side by side
        # timing each one so the user can see the latency difference
        results = {}
        timings = {}
        for name, fn in search_fns.items():
            t0 = time.time()
            results[name] = fn(query, top_k=top_k)
            timings[name] = round((time.time() - t0) * 1000)

        t0 = time.time()
        results["thematic"] = run_thematic(query, top_k=min(top_k, 9))
        timings["thematic"] = round((time.time() - t0) * 1000)

        timing_text = " | ".join([f"**{k}**: {v}ms" for k, v in timings.items()])
        st.markdown(timing_text)

        col_bm25, col_dense, col_hybrid = st.columns(3)
        for col, name in zip([col_bm25, col_dense, col_hybrid], ["bm25", "dense", "hybrid"]):
            with col:
                st.markdown(f"### {name}")
                if results[name]:
                    st.dataframe(passages_to_df(results[name], name), use_container_width=True, hide_index=True)
                else:
                    st.info("no results")

        st.markdown("### thematic (book level)")
        if results["thematic"]:
            st.dataframe(passages_to_df(results["thematic"], "thematic"), use_container_width=True, hide_index=True)

with tab_batch:
    st.subheader("batch evaluation on competency questions")

    categories = sorted(set(cq["category"] for cq in all_cqs))
    selected_cats = st.multiselect("categories", categories, default=categories)
    sample_n = st.number_input("sample per category (0 = all)", min_value=0, max_value=28, value=3)

    if st.button("run evaluation", key="batch_eval"):
        # filter and optionally sample the query set
        filtered = [cq for cq in all_cqs if cq["category"] in selected_cats]
        if sample_n > 0:
            import random
            random.seed(42)
            by_cat = {}
            for cq in filtered:
                by_cat.setdefault(cq["category"], []).append(cq)
            eval_set = []
            for cat in sorted(by_cat):
                eval_set.extend(random.sample(by_cat[cat], min(sample_n, len(by_cat[cat]))))
        else:
            eval_set = filtered

        progress = st.progress(0, text="evaluating...")
        rows = []

        for i, cq in enumerate(eval_set):
            q = cq["question"]
            expected = cq["example_books"]
            row = {"id": cq["id"], "category": cq["category"], "question": q[:80], "expected_book": expected[0]}

            for name, fn in search_fns.items():
                t0 = time.time()
                passages = fn(q, top_k=5)
                ms = round((time.time() - t0) * 1000)
                row[f"{name}_hit"] = passages[0].book_id in expected if passages else False
                row[f"{name}_score"] = round(passages[0].score, 4) if passages else 0.0
                row[f"{name}_ms"] = ms

            rows.append(row)
            progress.progress((i + 1) / len(eval_set), text=f"evaluated {i + 1}/{len(eval_set)}")

        progress.empty()
        df = pd.DataFrame(rows)

        # show the per query table with color coded hits
        st.markdown("### per query results")
        st.dataframe(df, use_container_width=True, hide_index=True)

        # aggregate metrics
        st.markdown("### aggregate metrics")
        agg_rows = []
        for method in ["bm25", "dense", "hybrid"]:
            agg_rows.append({
                "method": method,
                "book_accuracy": f"{df[f'{method}_hit'].mean():.0%}",
                "mean_score": round(df[f"{method}_score"].mean(), 4),
                "mean_latency_ms": round(df[f"{method}_ms"].mean()),
            })
        st.dataframe(pd.DataFrame(agg_rows), use_container_width=True, hide_index=True)

        # per category breakdown
        st.markdown("### by category")
        cat_rows = []
        for cat in sorted(df["category"].unique()):
            cat_df = df[df["category"] == cat]
            for method in ["bm25", "dense", "hybrid"]:
                cat_rows.append({
                    "category": cat, "n": len(cat_df), "method": method,
                    "book_accuracy": f"{cat_df[f'{method}_hit'].mean():.0%}",
                    "mean_score": round(cat_df[f"{method}_score"].mean(), 4),
                })
        st.dataframe(pd.DataFrame(cat_rows), use_container_width=True, hide_index=True)

with tab_head_to_head:
    st.subheader("head to head on curated queries")
    st.caption("hand picked queries that exercise each method's strengths and weaknesses")

    curated = [
        {"label": "factual (entity heavy)", "query": "Who is the narrator of The Great Gatsby?", "expected": "great_gatsby"},
        {"label": "fuzzy (no entity names)", "query": "a scientist who creates something monstrous and then suffers for it", "expected": "frankenstein"},
        {"label": "relational", "query": "Who served as a mentor to Edmond Dantes during his time in prison?", "expected": "count_of_monte_cristo"},
        {"label": "temporal", "query": "What happens after Dracula arrives in England on the ship Demeter?", "expected": "dracula"},
        {"label": "mixed (entity + semantic)", "query": "Beowulf fights Grendel in the great hall", "expected": "beowulf"}]

    if st.button("run head to head", key="h2h"):
        h2h_rows = []
        for item in curated:
            q = item["query"]
            for name, fn in search_fns.items():
                passages = fn(q, top_k=1)
                h2h_rows.append({
                    "label": item["label"], "query": q[:70], "method": name,
                    "hit": passages[0].book_id == item["expected"] if passages else False,
                    "top_book": passages[0].book_id if passages else "",
                    "score": round(passages[0].score, 4) if passages else 0.0,
                    "snippet": passages[0].text[:120].replace("\n", " ") if passages else ""})

            passages = run_thematic(q, top_k=1)
            h2h_rows.append({
                "label": item["label"], "query": q[:70], "method": "thematic",
                "hit": passages[0].book_id == item["expected"] if passages else False,
                "top_book": passages[0].book_id if passages else "",
                "score": round(passages[0].score, 4) if passages else 0.0,
                "snippet": passages[0].text[:120].replace("\n", " ") if passages else ""})

        st.dataframe(pd.DataFrame(h2h_rows), use_container_width=True, hide_index=True)
