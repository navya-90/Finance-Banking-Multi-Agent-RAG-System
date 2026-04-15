import json
from langchain_core.tools import tool
from db.database import (
    get_account_summary,
    get_recent_transactions,
    get_loan_details,
    get_open_fraud_flags,
    insert_fraud_flag,
)
from rag.retriever import retrieve


# ── Account tools ─────────────────────────────────────────────────────────────

@tool
def tool_get_accounts(customer_id: str) -> str:
    """
    Fetch all bank accounts for a customer.
    Returns account IDs, types (SAVINGS/CURRENT/FD), balances, and status.
    Use this when the user asks about their balance, account details, or account list.
    """
    rows = get_account_summary(customer_id)
    if not rows:
        return f"No accounts found for customer {customer_id}."
    # Mask full account number — show last 4 only
    for r in rows:
        aid = r.get("account_id", "")
        r["account_id_display"] = f"****{aid[-4:]}" if len(aid) >= 4 else aid
    return json.dumps(rows, indent=2)


@tool
def tool_get_transactions(account_id: str, limit: int = 10) -> str:
    """
    Fetch recent transactions for a bank account.
    Returns txn_id, type (DEBIT/CREDIT), amount, merchant, category, status, and timestamp.
    Use this when the user asks about recent activity, spending, or a specific transaction.
    limit: number of transactions to return (default 10, max 50).
    """
    limit = min(limit, 50)
    rows = get_recent_transactions(account_id, limit)
    if not rows:
        return f"No transactions found for account {account_id}."
    return json.dumps(rows, indent=2)


# ── Fraud tools ───────────────────────────────────────────────────────────────

@tool
def tool_get_fraud_flags(account_id: str) -> str:
    """
    Fetch open (unresolved) fraud flags for an account.
    Returns flag type, risk_score (0–1), and when it was flagged.
    Use this to check if suspicious activity has already been detected on an account.
    """
    rows = get_open_fraud_flags(account_id)
    if not rows:
        return f"No open fraud flags for account {account_id}."
    return json.dumps(rows, indent=2)


@tool
def tool_flag_transaction(txn_id: str, flag_type: str, risk_score: float) -> str:
    """
    Raise a fraud flag on a transaction.
    flag_type: one of UNUSUAL_HOUR | LARGE_CASH | UNKNOWN_MERCHANT | VELOCITY | FOREIGN
    risk_score: float between 0.0 and 1.0
    Use this when you identify a transaction that warrants investigation.
    """
    if not (0.0 <= risk_score <= 1.0):
        return "Error: risk_score must be between 0.0 and 1.0"
    insert_fraud_flag(txn_id, flag_type, risk_score)
    return f"Fraud flag raised on transaction {txn_id} (type={flag_type}, score={risk_score:.2f})."


# ── Loan tools ────────────────────────────────────────────────────────────────

@tool
def tool_get_loans(customer_id: str) -> str:
    """
    Fetch active loan details for a customer.
    Returns loan type, principal, interest rate, tenure, status, and disbursement date.
    Use this when the user asks about their existing loans, EMI, or repayment schedule.
    """
    rows = get_loan_details(customer_id)
    if not rows:
        return f"No loans found for customer {customer_id}."
    return json.dumps(rows, indent=2)


@tool
def tool_calculate_emi(principal: float, annual_rate: float, tenure_months: int) -> str:
    """
    Calculate the monthly EMI for a loan.
    principal: loan amount in INR
    annual_rate: annual interest rate as a percentage (e.g. 8.5 for 8.5%)
    tenure_months: repayment period in months
    Returns EMI amount, total payment, and total interest.
    """
    r = (annual_rate / 100) / 12
    if r == 0:
        emi = principal / tenure_months
    else:
        emi = principal * r * (1 + r) ** tenure_months / ((1 + r) ** tenure_months - 1)
    total = emi * tenure_months
    interest = total - principal
    return json.dumps({
        "monthly_emi":    round(emi, 2),
        "total_payment":  round(total, 2),
        "total_interest": round(interest, 2),
        "principal":      principal,
        "rate_pa":        annual_rate,
        "tenure_months":  tenure_months,
    }, indent=2)


# ── RAG tools ─────────────────────────────────────────────────────────────────

@tool
def tool_search_fraud_policy(query: str) -> str:
    """
    Search internal fraud detection policies and dispute procedures.
    Use this to answer questions about what actions to take on suspicious transactions,
    how to block a card, or how to raise a dispute.
    """
    docs = retrieve("fraud", query, n_results=3)
    return "\n---\n".join(docs) if docs else "No relevant fraud policy documents found."


@tool
def tool_search_loan_products(query: str) -> str:
    """
    Search the bank's loan product catalogue — interest rates, eligibility, tenure options.
    Use this when the user asks about loan types, rates, or eligibility criteria.
    """
    docs = retrieve("loans", query, n_results=3)
    return "\n---\n".join(docs) if docs else "No relevant loan product documents found."


@tool
def tool_search_account_faqs(query: str) -> str:
    """
    Search account-related FAQs — minimum balance, transfer limits, charges.
    Use this when the user asks about account rules or fees.
    """
    docs = retrieve("accounts", query, n_results=3)
    return "\n---\n".join(docs) if docs else "No relevant account FAQs found."


@tool
def tool_search_market_reports(query: str) -> str:
    """
    Search market intelligence reports — FD rates, mutual funds, investment products.
    Use this when the user asks about investment options, returns, or market rates.
    """
    docs = retrieve("markets", query, n_results=3)
    return "\n---\n".join(docs) if docs else "No relevant market reports found."


@tool
def tool_search_compliance_docs(query: str) -> str:
    """
    Search RBI/SEBI regulatory documents, KYC requirements, and AML guidelines.
    Use this when the user asks about compliance, KYC documents, or regulatory obligations.
    """
    docs = retrieve("compliance", query, n_results=3)
    return "\n---\n".join(docs) if docs else "No relevant compliance documents found."


@tool
def tool_search_general_faqs(query: str) -> str:
    """
    Search general banking FAQs — branch info, contact numbers, working hours.
    Use this for general queries that don't fit a specific domain.
    """
    docs = retrieve("general", query, n_results=3)
    return "\n---\n".join(docs) if docs else "No relevant FAQs found."


# ── Tool registry per agent domain ────────────────────────────────────────────

AGENT_TOOLS = {
    "fraud": [
        tool_get_transactions,
        tool_get_fraud_flags,
        tool_flag_transaction,
        tool_search_fraud_policy,
    ],
    "loans": [
        tool_get_loans,
        tool_calculate_emi,
        tool_search_loan_products,
    ],
    "accounts": [
        tool_get_accounts,
        tool_get_transactions,
        tool_search_account_faqs,
    ],
    "markets": [
        tool_search_market_reports,
    ],
    "compliance": [
        tool_search_compliance_docs,
    ],
    "general": [
        tool_search_general_faqs,
        tool_get_accounts,
    ],
}
