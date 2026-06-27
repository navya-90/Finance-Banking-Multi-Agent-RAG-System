"""
Supervisor agent — classifies intent and routes to specialist agents.
Uses CHEAP_LLM from config (fast, low-cost model for classification).
"""
import json
import re
from langchain_core.messages import SystemMessage, HumanMessage
from state import BankingState
from config import CHEAP_LLM, ESCALATION_CONFIDENCE_THRESHOLD, make_llm

ROUTING_PROMPT = """
You are a banking AI supervisor. Classify the user query into exactly ONE intent:

- fraud       : suspicious activity, blocked card, unauthorised transaction
- loans       : loan application, EMI, interest rates, eligibility, repayment
- accounts    : balance, statement, account details, transfers, beneficiaries
- markets     : stock prices, mutual funds, FD rates, portfolio advice
- compliance  : KYC, AML, regulatory queries, document submission
- general     : anything else, greetings, bank branch info

Respond with JSON only:
{
  "intent": "<one of the six labels>",
  "confidence": <0.0 to 1.0>,
  "customer_id": "<if mentioned, else null>",
  "account_id": "<if mentioned, else null>"
}
"""


def supervisor_node(state: BankingState) -> dict:
    llm = make_llm(CHEAP_LLM, temperature=0)   # no tools needed for classification

    last_user_msg = next(
        (m.content for m in reversed(state.messages) if m.type == "human"), ""
    )

    response = llm.invoke([
        SystemMessage(content=ROUTING_PROMPT),
        HumanMessage(content=last_user_msg),
    ])

    response_content = response.content
    if isinstance(response_content, list):
        response_content = "".join(
            part if isinstance(part, str) else (part.get("text", "") if isinstance(part, dict) else "")
            for part in response_content
        )

    parsed = {}
    try:
        match = re.search(r"\{.*\}", response_content, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
    except Exception:
        pass

    confidence = parsed.get("confidence", 0.0) if parsed else 0.0
    intent = parsed.get("intent", "general")
    escalate = (not parsed) or (confidence < ESCALATION_CONFIDENCE_THRESHOLD)
    reason = "Invalid routing response format" if not parsed else ("Low classification confidence" if escalate else None)

    return {
        "intent": intent,
        "confidence": confidence,
        "customer_id": parsed.get("customer_id") or state.customer_id,
        "account_id":  parsed.get("account_id")  or state.account_id,
        "escalate": escalate,
        "escalation_reason": reason,
    }
def route_after_supervisor(state: BankingState) -> str:
    if state.escalate:
        return "escalation"
    return state.intent or "general"
