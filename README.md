# Literary Search and Discovery Engine

A multi-agent retrieval system that answers natural language queries over a corpus of 9 public-domain literary texts. The core idea is that different query types need different retrieval mechanisms -- factual lookups work best with vector search, relational questions need graph traversal, thematic exploration needs book-level summaries. Instead of forcing one approach, an orchestrator classifies the query and routes it to the right agent.

The ontology schema, extraction pipeline, and evaluation methodology are grounded in recent literature on CQ-driven ontology engineering (Ontogenia, ESWC 2025) and follow current state-of-the-art practices for knowledge graph construction and retrieval-augmented generation.

```
          User Query
               |
        Intent Classifier
               |
          Orchestrator
          (entity resolution, routing)
         /     |      \        \
   Vector    Graph   Thematic  Comparative
    RAG       RAG     Agent      Agent
     \        |       /          /
      Synthesis Agent
      (grounding + citations)
```

## What's in here

- **`docs/ARCHITECTURE.md`** -- full system design: agent specifications, storage layout, ingestion pipeline, disambiguation logic, evaluation protocol, and multi-model routing strategy
- **`notebooks/taxonomy_ontology.ipynb`** -- ontology pipeline: corpus loading, LLM-based chapter detection, competency question generation, taxonomy and schema construction, schema validation, extraction testing, and book summary embeddings

## Stack

Gemini 2.5 Pro (Vertex AI) for generation, Instructor + Pydantic for structured output, Supermemory for knowledge graph storage, Vertex AI text-embedding-005 for embeddings, LangGraph for orchestration, PostgreSQL for observability.

## Corpus

9 Gutenberg books: Alice in Wonderland, Beowulf, The Count of Monte Cristo, Dracula, Frankenstein, The Great Gatsby, Pride and Prejudice, The Prince, The Complete Works of Shakespeare.
