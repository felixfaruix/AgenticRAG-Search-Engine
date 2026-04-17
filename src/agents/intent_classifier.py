"""Intent classifier: single LLM call to produce structured query understanding.
Does not perform entity resolution — the orchestrator handles that separately.
"""
import instructor
from src.models.agent_contracts import QueryUnderstanding

classify_prompt: str = """\
You are an intent classifier for a literary search engine covering 9 public-domain books \
(gothic horror, comedy of manners, epic poetry, adventure, fantasy).

Given a user query, determine:
- hop_count: "single" if answerable from one passage, "multi" if connecting multiple passages, "unknown" if unclear
- scope: "passage" for specific facts, "book" for book-level, "cross_book" for comparisons, "exploratory" for open-ended
- query_type: factual | fuzzy_recall | relational | temporal | comparative | thematic | exploratory
- sub_classification: "factual" if specific entities/events named, "fuzzy" if vague/paraphrased, "mixed" if both
- extracted_entities: raw entity mentions (characters, locations, events) exactly as written in the query
- confidence: your confidence in this classification (0.0 to 1.0)

User query: {query}
"""

def classify_query(query: str, model: str, client: instructor.Instructor) -> QueryUnderstanding:
    """Single LLM call via Instructor to classify a user query into structured intent fields.
    Activated at the start of every query before routing or entity resolution.
    """
    result: QueryUnderstanding = client.chat.completions.create(
        model=model, response_model=QueryUnderstanding, temperature=0.0, max_retries=2,
        messages=[{"role": "user", "content": classify_prompt.format(query=query)}])
    result.original_query = query
    return result
