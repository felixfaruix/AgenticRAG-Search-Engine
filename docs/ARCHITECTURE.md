# Literary Search and Discovery Engine

This document describes the design of an agentic retrieval system for literary texts. The system accepts natural language queries about a corpus of 9 public-domain books and returns grounded, cited answers by routing each query to a specialized retrieval agent. The agents differ not by query topic but by retrieval mechanism, because no single retrieval approach performs best across all query types. Single-hop factual queries are best served by vector-based retrieval with keyword matching. Multi-hop relational queries are best served by knowledge graph traversal. Thematic and exploratory queries are best served by book-level summaries. A system that uses only one approach will underperform on the query types that approach is weakest at. The orchestrator exists to prevent this: it classifies the query, selects the appropriate agent or combination of agents, and combines their results into a single grounded answer.

The system is built on three foundational decisions. First, the knowledge graph serves as an index, not as content: the output sent to the language model is the original source text attached to each graph edge, not bare triple labels, because retrieval systems using only extracted triples achieve roughly 65% entity coverage, losing 35% of answer-relevant information during extraction. Second, entity extraction and resolution happen once at the orchestrator level, so every agent receives pre-resolved entity identifiers rather than re-parsing the raw query. Third, all extraction into the knowledge graph passes through an offline validation step before storage, because current memory systems exhibit extraction accuracy below 62%.

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
          |  routing, decomposition   |      |
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
          |    retry mechanism)       |      +------------------+
          +---------------------------+      |   PostgreSQL     |
                                             |   (tool call     |
                                             |    logging,      |
                                             |    all agents)   |
                                             +------------------+
```

The intent classifier produces structured output validated through Instructor, determining hop complexity, scope, and raw entity mentions. It can be fine-tuned on approximately 1000 synthetic queries, each labeled with query type, hop count, and expected agent.

The orchestrator receives the classifier's output, performs entity resolution via RapidFuzz, and routes to the appropriate agent or agents. When confidence is below threshold, it decomposes the query into sub-queries dispatched in dependency order.

**Agents at a glance:**

- **Vector RAG Agent:** single-hop queries. Uses BM25 keyword search or dense semantic search depending on entity presence and specificity. Operates on chunked text indexes.
- **Graph RAG Agent:** multi-hop and relational queries. Traverses ontology-aware edges in Supermemory's memory graph, returning source chunks attached to each edge.
- **Thematic Agent:** broad thematic and exploratory queries. Searches book-level summaries and runs the convergence engine for recommendation narrowing.
- **Comparative Agent:** cross-book comparison queries. Parallel search across multiple book containers using both graph traversal and vector search.
- **Synthesis Agent:** receives retrieved passages from all active agents, generates a grounded answer with citations, runs a grounding verification check, and triggers agent retries via feedback when the answer does not meet quality thresholds.

| Store | Contents | Accessed By |
|---|---|---|
| Supermemory memory graph | Validated triples with source chunks, ontology edges | Graph RAG Agent |
| Recursive vector index | 512-token chunks, no enrichment | Vector RAG Agent (keyword search) |
| Contextual vector index | 512-token chunks with contextual headers | Vector RAG Agent (semantic search) |
| Book-level vector index | ~500-word book summaries | Thematic Agent |
| Supermemory agent containers | Per-agent session state, retry history | All agents |

---

## Taxonomy and Ontology

The knowledge graph requires a schema constraining what entities and relationships extraction is allowed to produce. Without it, the same character might appear as "Person" in one chunk and "Character" in another, preventing proper node merging.

The taxonomy defines six entity types: Character, Location, Event, Object, Theme, and Chapter. The ontology defines seven relationship types: INTERACTS_WITH (between characters), LOCATED_IN (event to location), PARTICIPATES_IN (character to event), OCCURS_BEFORE (between events), MENTIONED_IN (any entity to chapter), HAS_THEME (event to theme), and LOCATED_AT (character to location).

Six entity types and seven relationship types represent the maximum an extraction model can reliably distinguish in literary text. Additional granularity degrades extraction consistency because the model must make subjective literary judgments that vary across chunks. The schema is generated by a powerful model using competency questions and validated by a human reviewer. Storage uses property graphs rather than RDF, because property graphs optimize for traversal speed and all retrieval-augmented generation tooling integrates natively with them.

Documented in `notebooks/01_taxonomy_ontology.ipynb`.

---

## Entity Resolution

Entity resolution merges different surface forms of the same entity into a single canonical node. During ingestion, a lightweight model generates aliases for each entity ("the creature," "the monster," "the wretch," "Frankenstein's creation"). The canonical name and aliases are stored in a flat entity index scoped per book container.

At query time, the orchestrator extracts raw entity mentions and runs RapidFuzz weighted ratio matching against the entity index. When multiple candidates match, container scoping narrows the lookup to a specific book when other query entities have already resolved there. Co-occurrence disambiguation selects the candidate sharing the most resolved entities with the rest of the query. When neither mechanism resolves the ambiguity, the orchestrator triggers a clarification question.

RapidFuzz performs string similarity only. If the alias generation step misses a variation, that variation becomes an orphan node in the graph. Resolved entity node identifiers are passed to whichever agent the orchestrator activates.

---

## Supermemory Storage

Supermemory serves two roles: knowledge graph storage for the graph retrieval agent, and session memory for all agents.

Each validated triple is stored as a memory entry. The content field holds the full source chunk text. The metadata contains the structured triple, entity types, relationship type, book identifier, chapter, chunk index, confidence score, and validation status. Each book occupies its own container (`book_frankenstein`, `book_dracula`) to prevent cross-book entity confusion. Rejected triples go to per-book review containers.

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

Agent session containers (`agent_graph_rag_{session}`, `agent_vector_rag_{session}`) store intermediate results, retry history, and synthesis feedback per session. Agents review their own previous attempts before trying alternative strategies. Cross-agent context passing works through the orchestrator extracting structured context from one agent's container and injecting it into the next agent's input.

---

## Ingestion

Ingestion produces two parallel data structures from the same base chunks:

1. Recursive splitting of raw text into 512-token chunks with chapter-aware boundaries
2. Three vector indexes built from these chunks: recursive (no enrichment, for BM25), contextual (with model-generated headers, for dense search), book-level (one summary per book)
3. Triple extraction from each chunk, constrained to the ontology schema
4. Source chunk text attached to every extracted triple
5. Entity resolution via RapidFuzz and model-generated aliases
6. Validation agent scores each triple against its source text (Instructor, up to three retries)
7. Validated triples stored in Supermemory per-book containers; rejected triples stored in review containers

The recursive index serves BM25 keyword search with clean term frequencies: no injected context words that would dilute the inverse document frequency signal. The contextual index serves dense semantic search with resolved pronouns and references. Validation is offline; runtime agents operate on already-validated data.

---

## Synthetic Evaluation Dataset

A powerful model reads each book end-to-end and generates diverse queries with ground truth answers, organized by retrieval mechanism: specific factual (BM25), fuzzy recall (dense search), relational (graph traversal), temporal (temporal edge filtering), comparative (parallel cross-book search), thematic (book summaries), and exploratory (convergence engine).

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

For temporal queries, traversal restricts to OCCURS_BEFORE and OCCURS_AFTER edges. Temporal ordering comes from chapter number and position within chapter during extraction. If initial traversal returns low-relevance results, the agent checks its session container for previous attempts and tries alternative paths or falls back to the vector retrieval agent's tools.

### Thematic Agent

The thematic agent handles broad thematic and exploratory queries at book-level granularity.

| Tool | Function |
|---|---|
| book_summary_search | Dense search over book-level summaries |
| browse_features | Expose structured book ratings for convergence |
| ask_preference | Generate convergence question from highest-variance feature |

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

Each iteration's candidates, question, and response are stored in session state so the loop never repeats an attribute.

---

## Synthesis and Grounding Loop

The synthesis agent generates a grounded answer from retrieved passages, then runs a grounding check: is the answer supported by the passages and does it address the query?

On failure, the synthesis agent writes the result and specific feedback to the originating agent's session container via a dedicated tool. The agent reads this feedback, adjusts its retrieval strategy, and retries. Maximum two retries (three total attempts). On final failure, the system returns the best available answer with a low-confidence flag.

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
  "pass"     session container
      |              |
      v              v
  Return        Agent retries with
  answer        adjusted strategy
                (max 2 retries)
```

The grounding check and the status write are tools available to the synthesis agent. Each write records attempt status, feedback text, passages used, and confidence score, creating a traceable retry history in Supermemory.

---

## Passage Provenance

Every passage retrieved by any agent carries provenance metadata: book identifier, book title, chapter number, chunk index, passage text, retrieval score, retrieval method (BM25, dense, hybrid, or graph traversal), and the agent that retrieved it. Graph-retrieved passages also carry the traversal path and source triple. This provenance enables citation in the final answer and passage-level evaluation.

---

## PostgreSQL Observability

Every tool call is logged to PostgreSQL with a decorator requiring zero code changes to tool functions.

```python
class ToolCallLog(BaseModel):
    id: int
    timestamp: datetime
    session_id: str
    agent_type: str
    tool_name: str
    input_params: dict
    output_summary: str
    output_passage_count: int
    top_score: float | None
    latency_ms: int
    tokens_used: int | None
    success: bool
    error: str | None
    retry_attempt: int
```

PostgreSQL is queryable at demo time for aggregating latency per agent, counting tool calls per session, and computing retry rates. This observability layer provides operational instrumentation for monitoring and debugging alongside the live demo.

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

Different tasks require different model capabilities. Schema generation, synthetic query generation, and final answer synthesis require a powerful model because quality is critical. Triple extraction, contextual chunk headers, book-level summaries, intent classification, and triple validation use a lightweight model because volume is high and the tasks are structured and constrained. Entity resolution at query time uses RapidFuzz with no language model. Graph traversal at query time uses the Supermemory API with no language model. The principle is to use the least expensive model that meets the quality threshold for each task, with that threshold determined by evaluation.
