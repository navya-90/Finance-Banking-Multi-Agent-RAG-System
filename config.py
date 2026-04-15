import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "./data/banking.db")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
PHOENIX_PORT = int(os.getenv("PHOENIX_PORT", 6006))

# Routing thresholds
FRAUD_RISK_THRESHOLD = 0.75
ESCALATION_CONFIDENCE_THRESHOLD = 0.4

# ── LLM config — swap primary/fallback here, nothing else needs to change ─────
#
# OpenAI models  : "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"
# Gemini models  : "gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash"
#
PRIMARY_LLM  = "gpt-4o"             # used by all specialist agents
FALLBACK_LLM = "gemini-1.5-pro"     # used if primary raises an exception
CHEAP_LLM    = "gpt-4o-mini"        # used by supervisor (fast classification)

# RAG settings
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_DEVICE          = "cpu"
CHUNK_SIZE         = 500
CHUNK_OVERLAP      = 50
RAG_TOP_K          = 4


# ── LLM factory — returns the right provider class for any model name ─────────

def make_llm(model: str, temperature: float = 0.2, tools: list = None):
    """
    Return a LangChain chat LLM for the given model name.
    Detects provider from the model string — no hardcoding elsewhere needed.
    If tools is provided, calls .bind_tools() before returning.

    Supported prefixes:
      gpt-*, o1-*, o3-*          → ChatOpenAI
      gemini-*                   → ChatGoogleGenerativeAI
      mistral-*, open-mistral-*  → ChatMistralAI  (install langchain-mistralai)
    """
    model_lower = model.lower()

    if model_lower.startswith(("gpt-", "o1-", "o3-")):
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=model, temperature=temperature)

    elif model_lower.startswith("gemini-"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)

    elif model_lower.startswith(("mistral-", "open-mistral-")):
        from langchain_mistralai import ChatMistralAI
        llm = ChatMistralAI(model=model, temperature=temperature)

    else:
        raise ValueError(
            f"Unknown model '{model}'. Add a provider branch in config.make_llm()."
        )

    return llm.bind_tools(tools) if tools else llm
