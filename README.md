# Finance Banking Multi-Agent RAG System

Multi-agent AI for banking support — LangGraph orchestration, Google Gemini + GPT-4o, ChromaDB RAG, SQLite structured data, and Phoenix/OpenTelemetry observability.

---

## Architecture

```
User query
    │
    ▼
Supervisor (Gemini Flash — cheap, fast intent classification)
    │
    ├── fraud       → FraudAgent
    ├── loans       → LoanAgent
    ├── accounts    → AccountsAgent
    ├── markets     → MarketsAgent
    ├── compliance  → ComplianceAgent
    ├── general     → GeneralAgent
    └── escalation  → EscalationNode (human handoff)

Each agent:
  ChromaDB (RAG) + SQLite (structured) → Gemini Pro → response
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables
cp .env.example .env
# Edit .env with your API keys

# 3. Seed the database
python db/seed.py

# 4. Ingest documents into ChromaDB
python rag/ingest.py

# 5. Run the API server
uvicorn main:app --reload

# 6. Or run CLI quick-test
python main.py
```

Phoenix tracing UI will be available at **http://localhost:6006**

---

## Project Structure

```
finance_agent/
├── main.py                    # FastAPI entrypoint
├── graph.py                   # LangGraph graph assembly
├── state.py                   # Shared BankingState (Pydantic)
├── supervisor.py              # Intent classifier + router
├── config.py                  # Config & thresholds
├── requirements.txt
│
├── agents/
│   ├── base_agent.py          # RAG + SQL + LLM base class
│   └── specialist_agents.py   # All 5 domain agents + node fns
│
├── rag/
│   ├── retriever.py           # ChromaDB read/write
│   └── ingest.py              # Seed docs into ChromaDB
│
├── db/
│   ├── database.py            # SQLite schema + query helpers
│   └── seed.py                # Sample banking data
│
└── observability/
    └── tracing.py             # OpenTelemetry + Phoenix setup
```

---

## API

### POST /chat
```json
{
  "message": "I see an unknown ₹15,000 transaction",
  "session_id": "abc123",
  "customer_id": "C001",
  "account_id": "A001"
}
```
Response:
```json
{
  "reply": "We've detected a suspicious transaction...",
  "routed_to": "fraud",
  "confidence": 0.9,
  "escalated": false,
  "session_id": "abc123"
}
```

---

## Extending

**Add a new agent**: subclass `BaseAgent`, set `domain` and `system_prompt`, override `_fetch_sql_context()`, add a node in `graph.py`.

**Add new documents**: add entries to `rag/ingest.py` SAMPLE_DOCS and re-run.

**Switch LLM**: change `PRIMARY_LLM` in `config.py`.
