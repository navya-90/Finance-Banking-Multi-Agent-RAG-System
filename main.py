"""
FastAPI entrypoint.
Run:  uvicorn main:app --reload
"""
import sys
import uuid
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

# Force stdout/stderr to use UTF-8 to prevent encoding errors on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

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

    config = {"configurable": {"thread_id": session_id}}

    initial_state = {
        "messages": [HumanMessage(content=req.message)],
        "session_id": session_id,
        "customer_id": req.customer_id or None,
        "account_id": req.account_id or None,
    }

    result = banking_graph.invoke(initial_state, config)

    return ChatResponse(
        reply=result.get("agent_response") or "Sorry, I could not process your request.",
        routed_to=result.get("routed_to") or "unknown",
        confidence=result.get("confidence", 1.0),
        escalated=result.get("escalate", False),
        session_id=session_id,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


# ── CLI quick-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    queries = [
        ("I see an unknown transaction of Rs. 15,000 on my account", "C001", "A001"),
        ("What is the interest rate for a home loan?", "C001", ""),
        ("Show me my account balance", "C001", "A001"),
        ("Should I invest in index funds?", "", ""),
        ("My KYC is pending, what documents do I need?", "C001", ""),
    ]
    for i, (q, cid, aid) in enumerate(queries):
        state = {
            "messages": [HumanMessage(content=q)],
            "customer_id": cid,
            "account_id": aid,
        }
        config = {"configurable": {"thread_id": f"test-thread-{i}"}}
        result = banking_graph.invoke(state, config)
        print(f"\nQ: {q}")
        print(f"-> Routed to : {result.get('routed_to')}")
        print(f"-> Confidence: {result.get('confidence', 1.0):.2f}")
        print(f"-> Reply     : {(result.get('agent_response') or '')[:120]}...")
