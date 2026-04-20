# Literary Search and Discovery Engine

This document describes the design of an agentic retrieval system for literary texts. The system accepts natural language queries about a corpus of 9 public-domain books and returns grounded, cited answers by routing each query to a specialized retrieval agent. The agents differ not by query topic but by retrieval mechanism, because no single retrieval approach performs best across all query types. Single-hop factual queries are best served by vector-based retrieval with keyword matching. Multi-hop relational queries are best served by knowledge graph traversal. Thematic and exploratory queries are best served by book-level summaries. Cross-book queries are best served by a parallel search that covers each relevant book. A system that uses only one approach will underperform on the query types that approach is weakest at. The orchestrator exists to prevent this: it classifies the query, resolves the entities once, selects a single specialized agent based on the classification, and passes the result to a synthesis agent that generates a grounded, cited answer.

The system is built on three foundational decisions. First, the knowledge graph serves as an index, not as content: the output sent to the language model is the original source text attached to each graph edge, not bare triple labels, because retrieval systems using only extracted triples achieve roughly 65% entity coverage, losing 35% of answer-relevant information during extraction. Second, entity extraction and resolution happen once at the orchestrator level, so every agent receives pre-resolved entity identifiers rather than re-parsing the raw query. Triples are extracted by a high-quality model (Gemini 2.5 Pro) and stored directly in Supermemory without an offline validation step, because the extraction model produces sufficiently accurate triples that validation overhead is not justified. Third, LangGraph manages all orchestration state: checkpointing, pause/resume for human-in-the-loop disambiguation, inter-agent transitions, retry policies, and subgraph composition. Agents do not manage their own control flow or communicate state via manual flags. Supermemory handles persistent memory across sessions; LangGraph handles in-flight state within a session.

The corpus consists of 9 Gutenberg books spanning diverse genres (gothic horror, comedy of manners, epic poetry, adventure, fantasy), diverse lengths, and diverse narrative styles.

---

## System Design

```
                                          +------------------+
                                          |   Supermemory    |
                                          |   (graph store,  |
                  User Query              |    agent memory) |
                       |                  +------------------+
                       v                     |  accessed by
          +---------------------------+      |  all agents
          |    Intent Classifier      |      |  and orchestrator
          |    (lightweight model,    |      |
          |     structured output)    |      |
          +---------------------------+      |
                       |                     |
                       v                     |
          +---------------------------+      |
          |      Orchestrator         |------+
          |  entity resolution,       |      |
          |  routing                  |      |
          |  (LangGraph state graph)  |      |
          +---------------------------+      |
          |           |          |           |
   +------+    +------+    +--------+  +----+-----+
   |Vector|    |Graph |    |Thematic|  |Comparat. |
   | RAG  |    | RAG  |    | Agent  |  |  Agent   |
   |Agent |    |Agent |    |        |  |          |
   +------+    +------+    +--------+  +----------+
          |           |          |           |
          v           v          v           v
          +---------------------------+
          |   Cross-Encoder Rerank    |
          |   (top-k candidates ->    |
          |    top-3 high precision)  |
          +---------------------------+
                        |
                        v
          +---------------------------+
          |   Synthesis Agent         |
          |   (grounding loop with    |
          |    retry mechanism)       |
          +---------------------------+
```

The intent classifier produces structured output validated through Instructor, determining hop complexity, scope, sub-classification, and raw entity mentions. It does not determine which agent to use: routing is a deterministic mapping from hop count and scope that lives in the orchestrator. This separation keeps the classifier focused on fewer fields, improving its reliability.

The orchestrator receives the classifier's output, performs entity resolution via RapidFuzz, applies the routing table to select the target agent, and dispatches.

**Agents at a glance:**

- **Vector RAG Agent:** single-hop queries. Uses BM25 keyword search or dense semantic search depending on entity presence and specificity. Operates on chunked text indexes.
- **Graph RAG Agent:** multi-hop and relational queries. Traverses ontology-aware edges in Supermemory's memory graph, returning source chunks attached to each edge.
- **Thematic Agent:** broad thematic and exploratory queries. Searches book-level summaries ranked by cosine similarity.
- **Comparative Agent:** cross-book comparison queries. Parallel search across multiple book containers using both graph traversal and vector search.
- **Cross-Encoder Rerank:** second-stage precision stage that sits between every agent and synthesis. Reads the agent's top-k candidates, scores each (query, passage) pair jointly with a cross-attention model, and keeps the top-3 for downstream use.
- **Synthesis Agent:** receives the reranked top-3 passages from the routed agent, generates a grounded answer with citations, runs a grounding verification check, and triggers a retry of the same agent via scratchpad feedback when the answer does not meet quality thresholds.

| Store | Contents | Accessed By |
|---|---|---|
| Supermemory memory graph | Extracted triples with source chunks, ontology edges, per-book containers | Graph RAG Agent, Comparative Agent |
| Qdrant Cloud | Dense vectors (768d, contextual chunks) with payload metadata for filtered search | Vector RAG Agent, Comparative Agent |
| In-memory BM25 index | Recursive chunks tokenized for keyword search, built at process start | Vector RAG Agent, Comparative Agent |
| Book-level summary index | ~500-word book summaries with embeddings | Thematic Agent |
| Supermemory scratchpad containers | Per-agent per-session retrieval history: tool calls, passage counts, top scores, grounding feedback | Each agent on retry (own container only) |

---

## Taxonomy and Ontology

The knowledge graph requires a schema constraining what entities and relationships extraction is allowed to produce. Without it, the same character might appear as "Person" in one chunk and "Character" in another, preventing proper node merging.

The taxonomy defines six entity types: Character, Location, Event, Object, Theme, and Chapter. The ontology defines seven relationship types: INTERACTS_WITH (between characters), LOCATED_IN (event to location), PARTICIPATES_IN (character to event), OCCURS_BEFORE (between events), MENTIONED_IN (any entity to chapter), HAS_THEME (event to theme), and LOCATED_AT (character to location).

Six entity types and seven relationship types represent the maximum an extraction model can reliably distinguish in literary text. Additional granularity degrades extraction consistency because the model must make subjective literary judgments that vary across chunks. The schema is generated by a powerful model using competency questions and validated by a human reviewer. Storage uses property graphs rather than RDF, because property graphs optimize for traversal speed and all retrieval-augmented generation tooling integrates natively with them.

Documented in `notebooks/taxonomy_ontology.ipynb`.

---

## Entity Resolution

Entity resolution merges different surface forms of the same entity into a single canonical node. During ingestion, a lightweight model generates aliases for each entity ("the creature," "the monster," "the wretch," "Frankenstein's creation"). The canonical name and aliases are stored in a flat entity index scoped per book container.

At query time, the orchestrator extracts raw entity mentions and runs RapidFuzz weighted ratio matching against the entity index. Disambiguation confidence comes from the RapidFuzz similarity scores, not from a language model: if the top match scores 0.92 and the second scores 0.75, the gap is large and resolution is confident. If scores are 0.87 and 0.85, the gap is small and the orchestrator triggers clarification. This is deterministic and reliable.

When the top candidate's confidence falls in the 0.5–0.85 ambiguous band, the orchestrator triggers a clarification question via LangGraph's human-in-the-loop interrupt: the graph pauses, waits for the user's response, and resumes from the same point. The resolver accepts an optional `book_ids_hint` to narrow the search to specific books when the caller already knows scope, though the orchestrator currently invokes it without a hint and lets RapidFuzz pick the global best match per mention.

When the query contains no entity mentions ("something dark and philosophical"), entity resolution is skipped entirely. The classifier outputs hop_count="unknown" and scope="exploratory", the routing table maps this to the Thematic Agent, and dispatch proceeds without resolution.

RapidFuzz performs string similarity only. If the alias generation step misses a variation, that variation becomes an orphan node in the graph. Resolved entity node identifiers are passed to whichever agent the orchestrator activates.

---

## Supermemory Storage

Supermemory serves two roles: knowledge graph storage for the graph retrieval agent, and session memory for all agents.

Each resolved triple is stored as a memory entry. The content field holds the full source chunk text. The metadata contains the structured triple, entity types, relationship type, book identifier, chapter, chunk index, and confidence score. Each book occupies its own container (`book_frankenstein`, `book_dracula`) to prevent cross-book entity confusion.

Supermemory's memory graph tracks relationships between entries using ontology-aware edges. When the graph retrieval agent traverses from one node to another, Supermemory returns the edge type, target metadata, and the source chunk text attached to that edge.

```
              Supermemory Memory Graph
                    
[Victor]----INTERACTS_WITH----[Creature]
   |                              |
PARTICIPATES_IN             PARTICIPATES_IN
   |                              |
[Creation]----PRECEDES-------[Glacier Scene]
   |                              |
LOCATED_IN                   LOCATED_IN
   |                              |
[Ingolstadt]               [Mer de Glace]
```

Each node carries its source chunk text. Traversal collects these chunks as the retrieval output.

### Agent Memory (one scratchpad container per agent-session)

Each agent writes its retrieval attempts to a private Supermemory container scoped by agent type and session. On retry, the same agent reads its own scratchpad to see what it already tried and escalate strategy (broaden top_k, drop the book filter, relax the relationship filter) rather than blindly incrementing an attempt counter.

| Container | Scope | Contents |
|---|---|---|
| `agent_{type}_{session}_scratchpad` | Private to the agent, per session | Full retrieval history per attempt: tool name, parameters, passage count, top score, success flag, and any grounding feedback the synthesis agent wrote on the previous attempt. |

Inter-agent communication goes through LangGraph state, not through Supermemory: `AgentResult` objects are written to the state graph, which the synthesis node reads directly. Supermemory stores only the scratchpad (per-session) and the knowledge graph (persistent across sessions).

LangGraph manages transitions between agents, retries, and human-in-the-loop pauses. Supermemory manages the persistent graph and the per-session scratchpads agents consume on retry.

---

## Ingestion

Ingestion produces the following artifacts from the same base chunks:

1. Recursive splitting of raw text into 512-token chunks with chapter-aware boundaries
2. Each chunk carries metadata: book_id, book_title, author, chapter_number, chapter_title, chunk_index (position within chapter), total_chunks_in_chapter
3. Dense vectors (768d) computed from contextually enriched chunks and uploaded to a single Qdrant Cloud collection with payload metadata for filtered search
4. BM25 index built in-process at startup from the recursive chunks via `rank_bm25`; not persisted, rebuilt each time the app boots
5. Book-level summaries (~500 words each) embedded separately for the thematic agent
6. Triple extraction from each chunk, constrained to the ontology schema
7. Source chunk text attached to every extracted triple
8. Entity resolution: alias generation per entity via lightweight model, then RapidFuzz merging within each book
9. Resolved triples stored in Supermemory per-book containers

The chunk metadata enables filtered retrieval: when the orchestrator resolves entities to a specific book, the vector retrieval agent scopes search to only chunks from that book via Qdrant payload filtering. Chapter number and chunk position within chapter enable relative location of a passage in the narrative, which supports temporal reasoning even outside the graph retrieval path.

Hybrid retrieval is handled in Python via weighted Reciprocal Rank Fusion between the dense Qdrant results and the in-memory BM25 results, not by Qdrant. The dense weight is tuned above the BM25 weight because dense drives semantic precision while BM25 contributes rank signal for entity-rich queries. Keeping BM25 in-memory avoids maintaining a second index representation in Qdrant and keeps the corpus small enough that in-process scoring is negligible.

Supermemory container layout:

| Container | Contents |
|---|---|
| `book_{key}` (one per book) | Resolved triples with source chunks and ontology-typed edges |
| `agent_{type}_{session}_scratchpad` (one per agent per session) | Retrieval attempt history used only by that agent on retry |

---

## Evaluation Dataset

The evaluation set is 140 hand-written queries spread evenly across seven categories (factual, relational, temporal, structural, thematic, comparative, spatial), 20 queries per category. The set lives in `data/eval_queries/queries.json` as a flat JSON list with schema `{category, query, expected}` where `expected` is a list of book identifiers (single book for within-book categories, two books for comparative). Each query is grounded: the answer is retrievable from the ingested corpus, verified during authoring against the per-book entity alias indexes, the extracted triples, and the detected chapter metadata of the seven ingested books. Pride and Prejudice and Shakespeare's Complete Works are excluded from the expected sets because their triples were never uploaded to Supermemory, and Shakespeare's size biases BM25 toward surfacing it for unrelated queries.

Each query is chosen to stress one retrieval mechanism. Factual queries test entity-rich keyword matching. Relational queries test multi-hop graph traversal. Temporal queries test ordering edges (PRECEDES) and narrative sequence. Structural queries test chapter-level metadata and textual structure. Thematic queries test book-level summary search. Comparative queries test cross-book parallel search. Spatial queries test location entities and TRAVELS_TO / SET_IN edges. Within each category the 20 queries mix easy anchors (headline entities, titled chapters, core themes) with harder ones (subsidiary characters, less-prominent chapters, subsidiary motifs) so the delta between methods is not dominated by a single difficulty mode. The benchmark is deliberately small enough for the full query list to be inspected by hand; a larger auto-generated dataset would increase statistical power but would also introduce the well-known bias of questions written by the same model family that is being evaluated.

Documented in `notebooks/retrieval_evaluation.ipynb`.

---

## Agents

### Vector Retrieval Agent

The vector retrieval agent handles single-hop queries. It receives pre-resolved entity identifiers and a sub-classification from the orchestrator, then calls a single `vector_search` tool with the method selected from that sub-classification.

| Sub-classification from classifier | `vector_search` method | Index used |
|---|---|---|
| `factual` | `bm25` | In-memory BM25 over recursive chunks |
| `fuzzy` | `dense` | Qdrant dense vectors |
| `mixed` | `hybrid` | Weighted RRF over BM25 and dense |

On retry, the agent reads its scratchpad. If the prior top score was below 0.3 or the attempt count has reached 3, it broadens by switching to hybrid, raising top_k, and dropping the book_id filter.

### Graph Retrieval Agent

The graph retrieval agent handles multi-hop, relational, and temporal queries. It receives pre-resolved canonical entity names from the orchestrator and runs a parallel breadth-first traversal in Supermemory's memory graph via a single `graph_search` tool. At each hop, the frontier is capped to the top-N most relevant targets by edge similarity so fan-out stays bounded; per-hop requests run concurrently via a thread pool. The tool accepts an optional `relationship_type` filter. For temporal queries the agent sets this to `PRECEDES` so only chronology edges are traversed; for all other graph queries the filter is left open. Temporal ordering itself comes from chapter number and chunk position captured at extraction time. The filter value must match the predicate string the extraction pipeline actually wrote to Supermemory: the extracted ontology uses `PRECEDES` for event ordering.

On retry, the agent reads its scratchpad. If the last attempt returned zero passages, or the attempt count has reached 3, it drops the relationship filter and raises max_hops and top_k. In production mode, if graph retrieval exhausts its retries, the orchestrator falls back to a hybrid vector search over the full corpus rather than returning an empty result.

### Thematic Agent

The thematic agent handles broad thematic and exploratory queries at book-level granularity.

| Tool | Function |
|---|---|
| book_summary_search | Dense search over book-level summaries |

### Comparative Agent

The comparative agent activates when the classifier's scope is `cross_book` or entities resolve into more than one book. For each distinct book in the resolved entity set, it runs a hybrid vector search scoped to that book and, if that book has resolved entities, a graph traversal seeded by those entities. Results across books are deduplicated by (book_id, chapter, chunk_index) and ranked by retrieval score. No separate "comparison dimension" is extracted: the book set itself drives the parallel search.

---

## Reranking

The retrieval agents operate as bi-encoders. Dense search encodes the query and each chunk independently into 768-d vectors and ranks by cosine similarity; BM25 scores by term frequency and inverse document frequency; graph traversal ranks by edge similarity. These scores are fast to compute but coarse, because the query representation and passage representation are produced in isolation and can only interact through a scalar distance. Within the top-k returned, rank order is noisy: the passage that actually contains the answer is often not at rank 1.

A cross-encoder closes this gap. It reads `[CLS] query [SEP] passage [SEP]` in a single transformer pass, so self-attention runs jointly across query tokens and passage tokens and the model can condition "is this passage answer-bearing" on fine-grained lexical and semantic interaction (`occur before` vs `occur after`, entity alignment, predicate polarity). The output is a single scalar relevance score. This is the standard two-stage IR pattern from MS MARCO and BEIR: cheap recall-oriented retrieval to pull a candidate set, then expensive precision-oriented rerank over that set.

The rerank node sits between every agent and the synthesis agent in the LangGraph orchestrator. Each agent returns its top-k candidates (10 for vector_rag and graph_rag, 15 for the production fallback, per-book for comparative, summaries for thematic); the rerank node scores all (query, passage) pairs with the cross-encoder and passes the top-3 to synthesis. When the agent returns 3 or fewer candidates the rerank node no-ops, so the thematic agent's first attempt (3 book summaries) passes through unchanged.

The chosen model is `BAAI/bge-reranker-base` (278M parameters, XLM-RoBERTa backbone). It was trained on MS MARCO plus Natural Questions, HotpotQA, and FEVER, so its training distribution covers the multi-hop and entity-rich patterns literary queries exhibit. The backbone handles the 512-token chunk length cleanly without the attention saturation that smaller distilled baselines (for example, `cross-encoder/ms-marco-MiniLM-L-6-v2`) exhibit past ~256 tokens. Latency is ~80ms per pair on CPU, so reranking a top-10 candidate set adds under one second of end-to-end latency before synthesis.

**Contextual header parity.** The dense index is embedded from `contextual_header + body`, but the Qdrant payload stores body only. If the reranker were fed body alone it would see strictly less signal than the embedder did, and an earlier run of the evaluation showed this degrading dense/hybrid passage specificity by 3–18 points on categories whose relevance lives in scene framing (temporal, factual, thematic). The rerank node closes that gap by looking up each passage's contextual header from a `(book_id, chapter_number, chunk_index)` index built from `contextual_chunks.json` and prepending it to the cross-encoder input only — the returned `Passage.text` stays body-only so synthesis, grounding, and citations read the same text they always have.

The model is lazy-loaded on first call. Tests and retrieval paths that return a candidate set already smaller than the target top-n never trigger the load.

---

## Disambiguation and Clarification

RapidFuzz returns a ranked list of candidate canonical entities above its own lower floor (0.5). The orchestrator treats 0.5–0.85 as the ambiguous band: if the top match's confidence falls in that range, it pauses the graph via a LangGraph `interrupt`, surfacing the closest match and the book it belongs to, and asks the user to clarify. The user's response (a list of confirmed mentions) is fed back through the same resolution path.

```
Disambiguation:

  RapidFuzz scores candidates per raw mention
              |
              v
  Top match confidence >= 0.85 ?
              |
        yes   |   no
         |         \
         v          v
     continue   interrupt with closest match
                + book, ask user to clarify
                      |
                      v
                  resume with confirmed mentions,
                  re-run resolution
```

The pause uses LangGraph's human-in-the-loop interrupt: the graph checkpoints its full state, waits for the user, and resumes from exactly the same point without re-running classification or prior resolution. There is no multi-round clarification loop, no contrastive attribute selection, and no LLM involved in disambiguation — the confidence gap alone decides whether to pause.

---

## Synthesis and Grounding Loop

The synthesis agent generates a grounded answer from retrieved passages, then runs a grounding check: is the answer supported by the passages and does it address the query?

On failure, the synthesis node writes a failed `ScratchpadEntry` to the originating agent's scratchpad container, LangGraph transitions back to that agent, and the agent reads its scratchpad before retrying. The retry uses the prior entry to escalate strategy: the vector agent switches to hybrid and drops the book filter when the prior top score was below 0.3, and the graph agent drops the relationship filter and raises max_hops when the prior attempt returned zero passages. Maximum three total attempts. On final failure in production mode, when the routed agent was `graph_rag`, the orchestrator falls back once to a hybrid vector search over the full corpus rather than returning an empty result.

```
Synthesis Loop:

  Agent retrieves top-k candidates,
  writes scratchpad entry
                    |
                    v
    Cross-encoder reranks to top-3
                    |
                    v
        Synthesis generates answer
                    |
                    v
             Grounding check
                    |
             pass   |   fail
              |         |
              v         v
          Return    Synthesis writes failed
          answer    scratchpad entry; LangGraph
                    transitions back to agent
                            |
                            v
                    Agent reads scratchpad,
                    escalates strategy,
                    retries (max 3 attempts).
                    If graph_rag exhausts retries
                    in production mode, fallback
                    to hybrid vector search.
```

Passages retrieved by each agent flow to synthesis through LangGraph state (`AgentResult`), not through Supermemory. The scratchpad is only for retry signalling within a single agent's own recovery path.

---

## Passage Provenance

Every passage retrieved by any agent carries provenance metadata: book identifier, book title, chapter number, chunk index, passage text, retrieval score, retrieval method (BM25, dense, hybrid, or graph traversal), and the agent that retrieved it. Graph-retrieved passages also carry the traversal path and source triple. This provenance enables citation in the final answer and passage-level evaluation.

---

## Evaluation

All metrics evaluate at the book level because the corpus has no passage-level ground-truth labels. A method scores well by retrieving any passage from the right book; whether the specific passage answers the question is a separate check handled by the LLM judge.

### Retrieval metrics (mechanical, no LLM)

| Metric | What it measures |
|---|---|
| **hit@1** | Is the top-1 passage from one of the expected books? Binary, averaged across queries. |
| **MRR** | Reciprocal rank of the first correctly-booked passage in the top 5. Rewards getting the right book high. |
| **book_recall@3** | Fraction of expected books covered by the top-3 results. The primary metric for book-level retrievers (thematic) and for cross-book queries. |
| **book_correctness** | Fraction of the top-5 passages that come from any expected book. Catches methods that get the right book at rank 1 but fill the rest with noise. |

### Passage specificity (LLM judge)

For each query and method, Gemini Flash judges each of the top-5 passages as relevant or not relevant to the question, and the passage_specificity score is the fraction judged relevant. This catches the case where a method finds the right book but returns paragraphs that do not actually answer the query. The judge is throttled and retries on 429s with exponential backoff; failed calls return a sentinel so nan-aware averaging excludes them.

Thematic's passage_specificity is structurally penalized because it returns book-level summaries rather than chapter passages; read it on book_recall@3 and book_correctness instead.

### Methods compared

Five retrieval paths are benchmarked on the same 140 queries: BM25 (in-memory), dense (Qdrant), hybrid (weighted RRF of BM25 and dense), thematic (book-summary cosine search), and supermemory (graph traversal seeded by entities the classifier extracts from each query). Every method except thematic runs as a two-stage pipeline matching the production orchestrator: top-10 bi-encoder/graph retrieval followed by a `BAAI/bge-reranker-base` rerank to top-3, with the passage's contextual header prepended for the cross-encoder input only. This removes the earlier "bare vs reranked" variant split — the reranker is part of every backbone, so each row's passage_specificity already reflects the two-stage pipeline. Thematic is excluded from reranking because the book-level index has only 9 summaries and there is no meaningful pool to reorder.

---

## Multi-Model Routing

Different tasks require different model capabilities. Schema generation, triple extraction, and final answer synthesis use a powerful model (Gemini 2.5 Pro) because quality is critical and volume is moderate. Intent classification uses a lightweight model (Gemini 2.5 Flash) because it runs on every query and its schema is narrow enough to be reliable there. The LLM judge for passage specificity also runs on Gemini Flash. Entity resolution at query time uses RapidFuzz with no language model. Graph traversal at query time uses the Supermemory API with no language model. Passage reranking uses a local cross-encoder (`BAAI/bge-reranker-base`, 278M parameters) with no API call. The principle is to use the least expensive model that meets the quality threshold for each task, with that threshold determined by evaluation.
