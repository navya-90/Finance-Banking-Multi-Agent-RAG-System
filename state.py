from typing import Annotated, Optional
from pydantic import BaseModel
from langgraph.graph.message import add_messages


class BankingState(BaseModel):
    """Shared state passed between all agents in the graph."""

    # Conversation
    messages: Annotated[list, add_messages] = []
    session_id: str = ""

    # Routing
    intent: Optional[str] = None
    routed_to: Optional[str] = None

    # Agent output
    agent_response: Optional[str] = None
    confidence: float = 1.0

    # Escalation
    escalate: bool = False
    escalation_reason: Optional[str] = None

    # Context passed to tools via system prompt
    customer_id: Optional[str] = None
    account_id: Optional[str] = None
