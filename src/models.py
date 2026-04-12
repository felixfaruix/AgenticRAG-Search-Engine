from __future__ import annotations

from pydantic import BaseModel, Field


# ── Chapter Detection ──

class ChapterDetectionResult(BaseModel):
    """
    LLM-detected chapter headings for a book.
    """
    headings: list[str] = Field(description="Exact heading texts from candidate list, in appearance order")


# ── Competency Questions ──

class CompetencyQuestion(BaseModel):
    """
    A single competency question the ontology has to answer.
    """
    id: str = Field(description="Unique ID like CQ001")
    category: str = Field(description="One of: Factual, Relational, Temporal, Structural, Thematic, Comparative, Spatial")
    question: str = Field(description="Natural language question")
    example_books: list[str] = Field(description="1-2 book keys where this question applies")


class CompetencyQuestionsResponse(BaseModel):
    """
    LLM response containing generated competency questions.
    """
    questions: list[CompetencyQuestion]


# ── Taxonomy ──

class EntityType(BaseModel):
    """
    A single entity type in the taxonomy.
    """
    name: str = Field(description="PascalCase type name, e.g. 'Character'")
    description: str = Field(description="What this type represents")
    examples: list[str] = Field(description="3-5 example instances from the corpus")
    parent: str | None = Field(default=None, description="Parent type if hierarchical, else null")


class TaxonomyResponse(BaseModel):
    """
    Complete taxonomy returned by the LLM.
    """
    entity_types: list[EntityType]
    reasoning: str = Field(description="Why these types were chosen")


# ── Ontology ──

class RelationshipType(BaseModel):
    """
    A directed relationship type between two entity types.
    """
    name: str = Field(description="UPPER_SNAKE_CASE relationship name")
    description: str = Field(description="What this relationship represents")
    source_type: str = Field(description="Source entity type name")
    target_type: str = Field(description="Target entity type name")
    examples: list[str] = Field(description="2-3 example triples as 'Subject -> Object'")


class OntologySchema(BaseModel):
    """
    Complete ontology schema: entity types + relationship types.
    """
    entity_types: list[EntityType]
    relationship_types: list[RelationshipType]

    def entity_type_names(self) -> list[str]: return [et.name for et in self.entity_types]
    def relationship_type_names(self) -> list[str]: return [rt.name for rt in self.relationship_types]


class OntologyResponse(BaseModel):
    """
    LLM response for ontology generation.
    """
    relationship_types: list[RelationshipType]
    reasoning: str = Field(description="Why these relationships were chosen")


# ── Extraction ──

class ExtractedEntity(BaseModel):
    """
    A single entity extracted from a passage.
    """
    name: str = Field(description="Entity name as it appears in text")
    entity_type: str = Field(description="One of the allowed entity types")
    confidence: float = Field(ge=0.0, le=1.0, description="Extraction confidence")


class PassageExtractionResult(BaseModel):
    """
    All entities extracted from one passage.
    """
    entities: list[ExtractedEntity]


class Triple(BaseModel):
    """
    A single (subject, predicate, object) triple.
    """
    subject: str = Field(description="Subject entity name")
    subject_type: str = Field(description="Subject entity type from schema")
    predicate: str = Field(description="Relationship type from schema")
    object: str = Field(description="Object entity name")
    object_type: str = Field(description="Object entity type from schema")
    confidence: float = Field(ge=0.0, le=1.0)
    source_text: str = Field(description="Exact sentence(s) supporting this triple")


class ExtractionResponse(BaseModel):
    """
    All triples extracted from one passage.
    """
    triples: list[Triple]


# ── Book Summaries ──

class BookSummary(BaseModel):
    """
    A book-level summary for the thematic agent.
    """
    book_key: str = Field(description="Key from the books dict")
    title: str = Field(description="Full book title")
    summary: str = Field(description="800 word cohesive summary")
