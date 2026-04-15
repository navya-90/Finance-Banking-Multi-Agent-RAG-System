from langgraph.graph import StateGraph, END
from state import BankingState
from supervisor import supervisor_node, route_after_supervisor
from agents.specialist_agents import (
    fraud_node, loans_node, accounts_node,
    markets_node, compliance_node, general_node,
)


def escalation_node(state: BankingState) -> BankingState:
    """Human-in-the-loop placeholder — in production, trigger a ticket/Slack alert."""
    msg = (
        f"Your query has been escalated to a human agent. "
        f"Reason: {state.escalation_reason or 'Complex query'}. "
        f"You will be contacted within 2 business hours."
    )
    from langchain_core.messages import AIMessage
    return state.model_copy(update={
        "agent_response": msg,
        "routed_to": "escalation",
        "messages": state.messages + [AIMessage(content=msg)],
    })


def build_graph() -> StateGraph:
    g = StateGraph(BankingState)

    # Nodes
    g.add_node("supervisor",  supervisor_node)
    g.add_node("fraud",       fraud_node)
    g.add_node("loans",       loans_node)
    g.add_node("accounts",    accounts_node)
    g.add_node("markets",     markets_node)
    g.add_node("compliance",  compliance_node)
    g.add_node("general",     general_node)
    g.add_node("escalation",  escalation_node)

    # Entry point
    g.set_entry_point("supervisor")

    # Conditional routing from supervisor
    g.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "fraud":      "fraud",
            "loans":      "loans",
            "accounts":   "accounts",
            "markets":    "markets",
            "compliance": "compliance",
            "general":    "general",
            "escalation": "escalation",
        },
    )

    # All specialist nodes → END
    for node in ["fraud", "loans", "accounts", "markets", "compliance", "general", "escalation"]:
        g.add_edge(node, END)

    return g.compile()


# Singleton — import this wherever you need to run the graph
banking_graph = build_graph()
