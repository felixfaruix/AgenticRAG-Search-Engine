# Literary Search and Discovery Engine

This document describes the design of an agentic retrieval system for literary texts. The system accepts natural language queries about a corpus of 9 public-domain books and returns grounded, cited answers by routing each query to a specialized retrieval agent. The agents differ not by query topic but by retrieval mechanism, because no single retrieval approach performs best across all query types. Single-hop factual queries are best served by vector-based retrieval with keyword matching. Multi-hop relational queries are best served by knowledge graph traversal. Thematic and exploratory queries are best served by book-level summaries. A system that uses only one approach will underperform on the query types that approach is weakest at. The orchestrator exists to prevent this: it classifies the query, selects the appropriate agent or combination of agents, and combines their results into a single grounded answer

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
          v           v          v           |
          +---------------------------+      |
          |   Synthesis Agent         |------+
          |   (grounding loop with    |
          |    retry mechanism)       |
          +---------------------------+
```

The intent classifier produces structured output validated through Instructor, determining hop complexity, scope, sub-classification, and raw entity mentions. It does not determine which agent to use: routing is a deterministic mapping from hop count and scope that lives in the orchestrator. This separation keeps the classifier focused on fewer fields, improving its reliability. The classifier can be fine-tuned on approximately 1000 synthetic queries, each labeled with query type, hop count, and scope.

The orchestrator receives the classifier's output, performs entity resolution via RapidFuzz, applies the routing table to select the target agent, and dispatches.

**Agents at a glance:**

- **Vector RAG Agent:** single-hop queries. Uses BM25 keyword search or dense semantic search depending on entity presence and specificity. Operates on chunked text indexes.
- **Graph RAG Agent:** multi-hop and relational queries. Traverses ontology-aware edges in Supermemory's memory graph, returning source chunks attached to each edge.
- **Thematic Agent:** broad thematic and exploratory queries. Searches book-level summaries ranked by cosine similarity.
- **Comparative Agent:** cross-book comparison queries. Parallel search across multiple book containers using both graph traversal and vector search.
- **Synthesis Agent:** receives retrieved passages from all active agents, generates a grounded answer with citations, runs a grounding verification check, and triggers agent retries via feedback when the answer does not meet quality thresholds.

| Store | Contents | Accessed By |
|---|---|---|
| Supermemory memory graph | Extracted triples with source chunks, ontology edges, per-book containers | Graph RAG Agent |
| Qdrant Cloud (single collection, hybrid) | Dense vectors (768d, contextual chunks) + sparse vectors (BM25 tokens, recursive chunks) + payload metadata | Vector RAG Agent, Comparative Agent |
| Book-level vector index | ~500-word book summaries with embeddings | Thematic Agent |
| Supermemory scratchpad containers | Private per-agent retrieval history, failed attempts, grounding feedback | Each agent (own container only) |
| Supermemory results containers | Shared per-agent structured output, passages, confidence | All agents, orchestrator (read) |

---

## Taxonomy and Ontology

The knowledge graph requires a schema constraining what entities and relationships extraction is allowed to produce. Without it, the same character might appear as "Person" in one chunk and "Character" in another, preventing proper node merging.

The taxonomy defines six entity types: Character, Location, Event, Object, Theme, and Chapter. The ontology defines seven relationship types: INTERACTS_WITH (between characters), LOCATED_IN (event to location), PARTICIPATES_IN (character to event), OCCURS_BEFORE (between events), MENTIONED_IN (any entity to chapter), HAS_THEME (event to theme), and LOCATED_AT (character to location).

Six entity types and seven relationship types represent the maximum an extraction model can reliably distinguish in literary text. Additional granularity degrades extraction consistency because the model must make subjective literary judgments that vary across chunks. The schema is generated by a powerful model using competency questions and validated by a human reviewer. Storage uses property graphs rather than RDF, because property graphs optimize for traversal speed and all retrieval-augmented generation tooling integrates natively with them.

Documented in `notebooks/01_taxonomy_ontology.ipynb`.

---

## Entity Resolution

Entity resolution merges different surface forms of the same entity into a single canonical node. During ingestion, a lightweight model generates aliases for each entity ("the creature," "the monster," "the wretch," "Frankenstein's creation"). The canonical name and aliases are stored in a flat entity index scoped per book container.

At query time, the orchestrator extracts raw entity mentions and runs RapidFuzz weighted ratio matching against the entity index. Disambiguation confidence comes from the RapidFuzz similarity scores, not from a language model: if the top match scores 0.92 and the second scores 0.75, the gap is large and resolution is confident. If scores are 0.87 and 0.85, the gap is small and the orchestrator triggers clarification. This is deterministic and reliable.

When multiple candidates match, container scoping narrows the lookup to a specific book when other query entities have already resolved there. Co-occurrence disambiguation selects the candidate sharing the most resolved entities with the rest of the query. When neither mechanism resolves the ambiguity, the orchestrator triggers a clarification question via LangGraph's human-in-the-loop interrupt: the graph pauses, waits for the user's response, and resumes from the same point.

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
[Creation]--OCCURS_BEFORE--[Glacier Scene]
   |                              |
LOCATED_IN                   LOCATED_IN
   |                              |
[Ingolstadt]               [Mer de Glace]
```

Each node carries its source chunk text. Traversal collects these chunks as the retrieval output.

### Agent Memory (two containers per agent)

Each agent has two Supermemory containers per session:

| Container | Scope | Contents |
|---|---|---|
| `agent_{type}_{session}_scratchpad` | Private to the agent | Full retrieval history per attempt: tool name, parameters, passage count, top score, success/failure, grounding feedback from synthesis on retries. The agent reads this to know what it already tried. No other agent reads it. |
| `agent_{type}_{session}_results` | Shared, read by other agents | Clean structured output: retrieved passages, confidence, attempt number, grounding status. Other agents and the orchestrator read this via a dedicated read tool. Fixed structure so consumers know what to expect. |

Each agent has two Supermemory tools: `write_scratchpad` for its private container and `write_results` for the shared container. Other agents read shared results via `read_agent_results(agent_type, session_id)`.

LangGraph manages the transitions between agents (which agent runs next, when to retry, when to pause for user input). Supermemory manages the persistent data that agents produce and consume.

---

## Ingestion

Ingestion produces two parallel data structures from the same base chunks:

1. Recursive splitting of raw text into 512-token chunks with chapter-aware boundaries
2. Each chunk carries metadata: book_id, book_title, author, chapter_number, chapter_title, chunk_index (position within chapter), total_chunks_in_chapter
3. Both chunk representations uploaded to a single Qdrant Cloud collection with hybrid search: dense vectors (768d, from contextual enriched chunks) and sparse vectors (BM25 token weights with IDF, from recursive unenriched chunks) stored per point, with payload metadata for filtered search. Qdrant handles RRF fusion internally for hybrid queries. Book summaries stored separately (one per book, embedded for the thematic agent).
4. Triple extraction from each chunk, constrained to the ontology schema
5. Source chunk text attached to every extracted triple
6. Entity resolution: alias generation per entity via lightweight model, then RapidFuzz merging within each book
7. Resolved triples stored in Supermemory per-book containers

The chunk metadata enables filtered retrieval: when the orchestrator resolves entities to a specific book, the vector retrieval agent scopes search to only chunks from that book via Qdrant payload filtering. Chapter number and chunk position within chapter enable relative location of a passage in the narrative, which supports temporal reasoning even outside the graph retrieval path.

The sparse vectors are computed from unenriched recursive chunks to preserve clean term frequencies: no injected context words that would dilute the inverse document frequency signal. The dense vectors are computed from contextual enriched chunks with resolved pronouns and references. Both vectors live on the same Qdrant point, searched independently or fused depending on the query method. Payload indexes on book_id (keyword), chapter_number (integer), and author (keyword) enable filtered search without scanning the full collection.

Supermemory container layout:

| Container | Contents |
|---|---|
| `book_{key}` (one per book) | Resolved triples with source chunks, ontology-aware edges |
| `agent_{type}_{session}` (per agent per session) | Intermediate results, retry history, synthesis feedback |

---

## Synthetic Evaluation Dataset

A powerful model reads each book end-to-end and generates diverse queries with ground truth answers, organized by retrieval mechanism: specific factual (BM25), fuzzy recall (dense search), relational (graph traversal), temporal (temporal edge filtering), comparative (parallel cross-book search), and thematic (book summaries).

Each query is tagged with expected books, passages, query type, hop count, expected agent, ground truth answer, and specificity score. Approximately 1000 queries are generated, providing sufficient volume to fine-tune the intent classifier.

Documented in `notebooks/02_synthetic_eval_dataset.ipynb`.

---

## Agents

### Vector Retrieval Agent

The vector retrieval agent handles single-hop queries. It receives pre-resolved entity identifiers and a sub-classification from the orchestrator.

| Sub-classification | Tool | Index | Condition |
|---|---|---|---|
| Factual | keyword_search | Recursive index (BM25) | Entities present, high specificity |
| Fuzzy recall | semantic_search | Contextual index (dense) | No entities, low specificity |
| Mixed | hybrid_search | Both (RRF fusion) | Uncertain sub-classification |

### Graph Retrieval Agent

The graph retrieval agent handles multi-hop, relational, and temporal queries. It receives pre-resolved entity node identifiers from the orchestrator and traverses typed edges in Supermemory's memory graph, collecting source chunk text from each traversed node.

| Tool | Function |
|---|---|
| graph_traverse | Follow typed edges from start nodes, return source chunks per hop |
| temporal_chain | Follow only OCCURS_BEFORE/AFTER edges, return ordered event sequence |
| subgraph_context | Retrieve all edges between a given set of nodes |

For temporal queries, traversal restricts to OCCURS_BEFORE and OCCURS_AFTER edges. Temporal ordering comes from chapter number and position within chapter during extraction. If initial traversal returns low-relevance results, the agent reads its scratchpad container for previous attempts and tries alternative paths or falls back to the vector retrieval agent's tools.

### Thematic Agent

The thematic agent handles broad thematic and exploratory queries at book-level granularity.

| Tool | Function |
|---|---|
| book_summary_search | Dense search over book-level summaries |

### Comparative Agent

The comparative agent searches multiple books in parallel using both graph traversal across book containers and vector search within each book. The orchestrator passes the comparison dimension from the query.

---

## Disambiguation and Clarification

Disambiguation triggers when entity resolution produces multiple candidates with close confidence scores at the orchestrator level, or when an agent's retrieval returns a flat score distribution (high entropy across top-k results).

The system examines the one-hop graph neighborhood of each candidate node in Supermemory, performs contrastive feature extraction to identify attributes that differ between candidates, and selects the attribute with highest discriminative power (the one that most evenly splits candidates). A lightweight model converts this into a clarification question. The user's response is a soft re-ranking signal.

```
Disambiguation Loop:

  Multiple candidates with close scores
              |
              v
  Fetch one-hop neighborhoods per candidate
              |
              v
  Contrastive extraction: find attributes
  that differ between neighborhoods
              |
              v
  Select highest info-gain attribute,
  generate clarification question
              |
              v
  User responds (soft signal)
              |
              v
  Re-rank candidates
              |
              v
  Exit conditions:
    score gap exceeds threshold    --> answer
    max 3 rounds reached           --> present top candidates
    user confirms                  --> answer
    none met                       --> loop with next attribute
```

Each iteration's candidates, question, and response are stored in LangGraph's checkpointed state so the loop never repeats an attribute. The pause for user input uses LangGraph's human-in-the-loop interrupt: the graph checkpoints its full state, waits for the user's response, and resumes from exactly the same point without re-running previous steps.

---

## Synthesis and Grounding Loop

The synthesis agent generates a grounded answer from retrieved passages, then runs a grounding check: is the answer supported by the passages and does it address the query?

On failure, the synthesis agent writes feedback to the originating agent's scratchpad container via `write_scratchpad`, describing what was missing or incorrect. LangGraph transitions back to the originating agent node, which reads its scratchpad, adjusts its retrieval strategy, writes new results to its shared results container, and LangGraph transitions back to synthesis. Maximum two retries (three total attempts), enforced by LangGraph's retry policy. A wallclock timeout per agent prevents hanging calls from blocking the pipeline. On final failure, the system returns the best available answer with a low-confidence flag.

```
Synthesis Loop:

  Agent retrieves passages
          |
          v
  Synthesis generates answer
          |
          v
  Grounding check
          |
     pass | fail
      |       |
      v       v
  Write      Write feedback to agent's
  to shared  scratchpad container,
  results    LangGraph transitions
      |      back to agent node
      v              |
  Return        Agent reads scratchpad,
  answer        retries with adjusted
                strategy (max 2 retries)
```

The grounding check and the container writes are tools available to the synthesis agent. Scratchpad writes record attempt number, feedback text, passages used, and confidence. Shared results writes record the final output for downstream consumers.

---

## Passage Provenance

Every passage retrieved by any agent carries provenance metadata: book identifier, book title, chapter number, chunk index, passage text, retrieval score, retrieval method (BM25, dense, hybrid, or graph traversal), and the agent that retrieved it. Graph-retrieved passages also carry the traversal path and source triple. This provenance enables citation in the final answer and passage-level evaluation.

---

## Evaluation

### Primary metric

Answer accuracy: did the system return the correct answer, grounded in the correct source passage?

### Unbiased evaluation protocol

Position-swapped scoring (evaluate both orderings, declare tie if results differ), length-aligned answers (normalize before comparison), multiple evaluation rounds (report median and percentiles). Questions generated from graph structures and source text, not vague corpus summaries.

### Per-agent metrics

| Agent | Metric | Measured By |
|---|---|---|
| Vector RAG | Precision@1, MRR | Correct passage in top result |
| Graph RAG | Multi-hop accuracy | Correct answer requiring 2 or more hops |
| Thematic | Recommendation relevance | LLM-as-judge on book relevance |
| Comparative | Cross-book coverage | Relevant passages found in all relevant books |
| Orchestrator | Routing accuracy | Routed strategy matches optimal strategy per ablation |

### Ablation comparisons

| Comparison | Expected Result |
|---|---|
| Vector RAG vs Graph RAG on single-hop | Vector RAG wins |
| Vector RAG vs Graph RAG on multi-hop | Graph RAG wins |
| Hybrid orchestrated vs single strategy | Hybrid wins |
| Constrained ontology vs unconstrained extraction | Constrained wins |
| Text-augmented triples vs bare triples | Augmented wins |

---

## Multi-Model Routing

Different tasks require different model capabilities. Schema generation, synthetic query generation, and final answer synthesis require a powerful model because quality is critical. Triple extraction, contextual chunk headers, book-level summaries, and intent classification use a lightweight model because volume is high and the tasks are structured and constrained. Entity resolution at query time uses RapidFuzz with no language model. Graph traversal at query time uses the Supermemory API with no language model. The principle is to use the least expensive model that meets the quality threshold for each task, with that threshold determined by evaluation.
