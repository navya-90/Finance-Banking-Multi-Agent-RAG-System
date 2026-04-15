"""
FastAPI entrypoint.
Run:  uvicorn main:app --reload
"""
import uuid
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from db.database import init_db
from graph import banking_graph
from state import BankingState
from observability.tracing import setup_tracing

app = FastAPI(title="Finance Banking Multi-Agent AI")


@app.on_event("startup")
def startup():
    init_db()
    setup_tracing()
    print("DB initialised. Tracing active.")


class ChatRequest(BaseModel):
    message: str
    session_id: str = ""
    customer_id: str = ""
    account_id: str = ""


class ChatResponse(BaseModel):
    reply: str
    routed_to: str
    confidence: float
    escalated: bool
    session_id: str


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())

    initial_state = BankingState(
        messages=[HumanMessage(content=req.message)],
        session_id=session_id,
        customer_id=req.customer_id,
        account_id=req.account_id,
    )

    result: BankingState = banking_graph.invoke(initial_state)

    return ChatResponse(
        reply=result.agent_response or "Sorry, I could not process your request.",
        routed_to=result.routed_to or "unknown",
        confidence=result.confidence,
        escalated=result.escalate,
        session_id=session_id,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


# ── CLI quick-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    queries = [
        ("I see an unknown transaction of ₹15,000 on my account", "C001", "A001"),
        ("What is the interest rate for a home loan?", "C001", ""),
        ("Show me my account balance", "C001", "A001"),
        ("Should I invest in index funds?", "", ""),
        ("My KYC is pending, what documents do I need?", "C001", ""),
    ]
    for q, cid, aid in queries:
        state = BankingState(
            messages=[HumanMessage(content=q)],
            customer_id=cid,
            account_id=aid,
        )
        result = banking_graph.invoke(state)
        print(f"\nQ: {q}")
        print(f"→ Routed to : {result.routed_to}")
        print(f"→ Confidence: {result.confidence:.2f}")
        print(f"→ Reply     : {result.agent_response[:120]}...")
