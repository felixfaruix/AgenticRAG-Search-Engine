"""configuration: client initialization, model routing, and naming conventions.

multi-model routing:
  - intent classification uses a fine-tuned lightweight model (high volume, structured output).
  - synthesis and grounding use a powerful model (quality-critical, low volume).
  - retrieval agents (vector_rag, graph_rag, thematic, comparative) use no llm;
    they operate through qdrant, kuzu, and embedding models only.
"""
import os

classifier_model: str = os.environ.get(
    "CLASSIFIER_MODEL",
    "gemini-2.5-flash-lite",
)

classifier_model_tuned: str | None = os.environ.get("CLASSIFIER_MODEL_TUNED")

synthesis_model: str = os.environ.get(
    "SYNTHESIS_MODEL",
    "gemini-2.5-flash",
)


def get_classifier_model() -> str:
    """return the fine-tuned model endpoint when available, else the base model."""
    return classifier_model_tuned or classifier_model
