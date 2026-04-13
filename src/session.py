"""Session management: ID generation and container naming.
The orchestrator calls create_session_id() once per incoming query. That ID flows
through LangGraph state into every agent and every Supermemory write. Container
names are deterministic from (agent_type, session_id) so any component can
reconstruct them without passing container references around.
Supermemory auto-creates containers on first write — no explicit creation step needed.
"""

import uuid

def create_session_id() -> str:
    """Generate a unique session ID. Called once per query by the orchestrator.
    """
    return uuid.uuid4().hex

def scratchpad_container(agent_type: str, session_id: str) -> str:
    """Private container name for an agent's retrieval attempt history.
    """
    return f"agent_{agent_type}_{session_id}_scratchpad"

def results_container(agent_type: str, session_id: str) -> str:
    """Shared container name for an agent's final results, read by orchestrator and other agents.
    """
    return f"agent_{agent_type}_{session_id}_results"
