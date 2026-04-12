# Literary Search and Discovery Engine

An agentic retrieval system for literary texts. The system accepts natural language queries about a corpus of 9 public-domain books (from Project Gutenberg) and returns grounded, cited answers by routing each query to a specialized retrieval agent.

## Motivation

No single retrieval approach performs best across all query types. Single-hop factual queries are best served by vector search with keyword matching. Multi-hop relational queries require knowledge graph traversal. Thematic and exploratory queries benefit from book-level summaries. This system uses an orchestrator to classify the query, select the appropriate agent, and combine results into a single grounded answer.

## Architecture

The system routes queries through four specialized agents, each backed by a different retrieval mechanism:

- **Vector RAG Agent** -- single-hop factual queries via BM25 or dense semantic search over chunked text
- **Graph RAG Agent** -- multi-hop relational queries via knowledge graph traversal (Supermemory)
- **Thematic Agent** -- broad thematic queries via book-level summary search
- **Comparative Agent** -- cross-book comparisons via parallel search across book containers

A synthesis agent generates grounded answers with citations and triggers retries when quality thresholds are not met.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full system design.

## Corpus

| Book | Author | Genre |
|---|---|---|
| Alice's Adventures in Wonderland | Lewis Carroll | Fantasy |
| Beowulf | Anonymous | Epic Poetry |
| The Count of Monte Cristo | Alexandre Dumas | Adventure |
| Dracula | Bram Stoker | Gothic Horror |
| Frankenstein | Mary Shelley | Gothic Horror |
| The Great Gatsby | F. Scott Fitzgerald | Modernist Fiction |
| Pride and Prejudice | Jane Austen | Comedy of Manners |
| The Prince | Niccolo Machiavelli | Political Philosophy |
| The Complete Works of Shakespeare | William Shakespeare | Drama |

## Pipeline

1. **Ontology design** (`notebooks/taxonomy_ontology.ipynb`) -- LLM-generated competency questions drive taxonomy and ontology schema creation, validated against the corpus
2. **Entity extraction** -- schema-constrained triple extraction from chunked text
3. **Entity resolution** -- alias generation and RapidFuzz matching for canonical node merging
4. **Ingestion** -- parallel construction of vector indexes and knowledge graph in Supermemory
5. **Evaluation** -- synthetic query dataset with per-agent metrics and ablation comparisons

## Stack

- **LLM**: Gemini 2.5 Pro via Vertex AI (OpenAI-compatible endpoint)
- **Structured output**: Instructor with Pydantic models
- **Knowledge graph**: Supermemory
- **Embeddings**: Vertex AI text-embedding-005
- **Observability**: PostgreSQL tool call logging
- **Orchestration**: LangGraph

## Setup

```bash
# Clone and install dependencies
git clone https://github.com/felixfaruix/literary-search-engine.git
cd literary-search-engine
uv sync

# Set environment variables
cp .env.example .env
# Fill in GCP_PROJECT, GCP_LOCATION, GOOGLE_APPLICATION_CREDENTIALS, DATABASE_URL
```

Requires Python 3.12+ and a GCP project with Vertex AI enabled.
