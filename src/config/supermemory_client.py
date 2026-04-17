"""supermemory client configuration and container naming helpers."""

import os
from pathlib import Path
from dotenv import load_dotenv
from supermemory import Supermemory

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

client: Supermemory | None = None


def get_client() -> Supermemory:
    """return a configured supermemory client (singleton)."""
    global client

    if client is None:
        client = Supermemory(api_key=os.environ["sup_key"])
    return client


def book_container(book_id: str) -> str:
    """container for validated triples: book_{book_id}."""
    return f"book_{book_id}"


def review_container(book_id: str) -> str:
    """container for rejected triples: review_{book_id}."""
    return f"review_{book_id}"


def scratchpad_container(agent_type: str, session_id: str) -> str:
    """private container for an agent's retrieval history."""
    return f"agent_{agent_type}_{session_id}_scratchpad"


def results_container(agent_type: str, session_id: str) -> str:
    """shared container for an agent's final results."""
    return f"agent_{agent_type}_{session_id}_results"
