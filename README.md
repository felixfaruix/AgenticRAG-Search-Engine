# Agentic RAG 

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging_Face-FFD21E?logo=huggingface&logoColor=black)
![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?logo=langchain&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?logo=pydantic&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?logo=postgresql&logoColor=white)
![Supermemory](https://img.shields.io/badge/Supermemory-000000)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Google Cloud](https://img.shields.io/badge/Google_Cloud-4285F4?logo=googlecloud&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-8E75B2?logo=googlegemini&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-154F3F?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)

Dear reader/s, this is a project tha reflects the thinking processes that I like to apply to my projects. Evaluation driven, everything quantifiable - if I can't quantify the results, I haven't built anything. A demo is a demo, but building reliable AI systems requires much more effort and A LOT of end-to-end testings. Isolation is unfortunately not easy when it comes to AI application. Thus, we need to have a good baseline evaluation we can refer to. 

I suggest you to start reading this repo from the notebooks: first, you want to inspect the taxonomy and ontology one, then the extraction notebook. 
Some of the concepts are new to me, but this is what I like about addressing a new project - a good balance between new and old stuff. 

The entire idea starts from one simple assumption: not every question is the same, so they need different retrieval strategies. Every choice is grounded on the most revent research paper about this topic. 

----

A multi-agent retrieval system that answers natural language queries over a corpus of 9 public-domain literary texts. The core idea is that different query types need different retrieval mechanisms: factual lookups work best with vector search, relational questions need graph traversal, thematic exploration needs book-level summaries. Instead of forcing one approach, an orchestrator classifies the query and routes it to the right agent.

The ontology schema, extraction pipeline, and evaluation methodology are grounded in recent literature on CQ-driven ontology engineering (Ontogenia, ESWC 2025) and follow current state-of-the-art practices for knowledge graph construction and retrieval-augmented generation.

```
          User Query
               |
        Intent Classifier
               |
          Orchestrator  ----------------  PostgreSQL
       (entity resolution,            (tool call logging,
            routing)                    latency, retries)
         /     |      \        \
   Vector    Graph   Thematic  Comparative
    RAG       RAG     Agent      Agent
     |         |       |          |
     |    Supermemory  |          |
     |    (KG store,   |          |
     |     agent mem)  |          |
      \        |       /          /
       Synthesis Agent           /
       (grounding + citations)  /
```

## Agents

Each agent handles a different retrieval pattern. They share the same orchestrator and entity resolution layer but differ in how they find and return relevant passages.

- **Vector RAG Agent**: single-hop factual queries. Switches between BM25 keyword search (high entity specificity) and dense semantic search (fuzzy recall) over chunked text indexes. Uses reciprocal rank fusion when the sub-classification is uncertain.
- **Graph RAG Agent**: multi-hop relational and temporal queries. Traverses typed edges in the knowledge graph, collecting source text attached to each node. For temporal queries, restricts traversal to PRECEDES edges and returns ordered event chains.
- **Thematic Agent**: broad thematic and exploratory queries. Searches over book-level summary embeddings, then runs a convergence loop that narrows recommendations by asking discriminative preference questions.
- **Comparative Agent**: cross-book comparisons. Runs parallel search across multiple book containers using both graph traversal and vector search, then aligns results along the comparison dimension extracted from the query.
- **Synthesis Agent**: not a retrieval agent. Takes passages from all active agents, generates a grounded answer with citations, and runs a verification check. On failure, it writes structured feedback back to the originating agent for retry (max 2 retries).

## Entity Taxonomy and Ontology - Knowledge Graph

The knowledge graph is built through a two-stage LLM-driven pipeline documented in `notebooks/taxonomy_ontology.ipynb`. The diagram below shows the resulting ontology schema: 8 entity types as nodes and 12 typed relationships as directed edges, each constrained to specific source and target types.

<p align="center">
  <img src="docs/images/relationships.png" alt="Ontology Schema" width="600">
</p>

**Competency questions.** Following the CQ-by-CQ methodology (Chevignard et al., ESWC 2025), we generate ~200 competency questions that define what the ontology must be able to answer. These questions — not sample passages — drive the schema design, avoiding noise injection from raw text.

**Taxonomy.** The LLM produces an entity type hierarchy from the competency questions alone: 8 entity types (Character, Location, Event, Object, Theme, TextualComponent, LiteraryWork, Author) organized to cover factual, relational, temporal, structural, thematic, comparative, and spatial query patterns.

**Ontology.** The schema adds 12 typed relationship types (WROTE, APPEARS_IN, SET_IN, RELATED_TO, PARTICIPATES_IN, PRECEDES, IS_PART_OF, IS_SUBCOMPONENT_OF, OCCURS_IN, MANIFESTS_THEME, USES, TRAVELS_TO) with constrained source/target entity pairs. This follows the principle from Ontogenia that competency questions are sufficient input for schema generation: additional text samples introduce extraction bias without improving schema coverage.

**Validation.** Schema quality is tested by extracting triples from random 800-word passages across the corpus and measuring per-entity-type hit rates. This acts as a proxy for schema completeness, confirming the type system captures the structural patterns present in the texts.

## Memory and Storage

Supermemory serves as the unified storage layer, handling both the knowledge graph and per-agent session memory. It natively supports ontology-aware edges, container-scoped storage, and memory graph traversal — which maps directly onto the multi-agent retrieval needs without building a custom graph layer.

Different retrieval patterns need different index structures. Keyword search needs clean chunks with undiluted term frequencies. Dense semantic search needs contextually enriched chunks with resolved pronouns. Thematic exploration needs book-level representations. Graph traversal needs typed edges with source text attached. The system maintains four separate indexes to serve these patterns:

| Index | Contents | Used By |
|---|---|---|
| Recursive vector index | 512-token chunks, no enrichment | Vector RAG Agent (BM25 keyword search) |
| Contextual vector index | 512-token chunks with LLM-generated contextual headers | Vector RAG Agent (dense semantic search) |
| Book-level vector index | ~800-word book summaries | Thematic Agent |
| Supermemory memory graph | Validated triples with source chunks, ontology-aware edges | Graph RAG Agent, Comparative Agent |

## Project Structure

- **`docs/ARCHITECTURE.md`**: full system design: agent specifications, storage layout, ingestion pipeline, disambiguation logic, evaluation protocol, and multi-model routing strategy
- **`notebooks/taxonomy_ontology.ipynb`**: ontology pipeline: corpus loading, LLM-based chapter detection, competency question generation, taxonomy and schema construction, schema validation, extraction testing, and book summary embeddings

## Corpus

9 Gutenberg books: Alice in Wonderland, Beowulf, The Count of Monte Cristo, Dracula, Frankenstein, The Great Gatsby, Pride and Prejudice, The Prince, The Complete Works of Shakespeare.

## Folders

| Path | Contents |
|---|---|
| `src/agents/` | Intent classifier, orchestrator, retrieval agents, synthesis agent |
| `src/models/` | Pydantic contracts shared across agents and tools |
| `src/tools/` | Vector search, graph search, grounding check, scratchpad/results I/O |
| `src/config/` | Model routing and client initialization |
| `notebooks/` | Taxonomy/ontology pipeline, extraction pipeline |
| `data/aliases/` | Per-book entity alias indexes for RapidFuzz resolution |
| `data/books/` | Raw Gutenberg text files |
| `data/triples/` | Extracted and validated knowledge graph triples |
| `data/fine_tuning/` | 1000-query intent classifier dataset (train/eval, Vertex AI format) |
| `docs/` | Architecture document and images |
| `scripts/` | Utility scripts |
