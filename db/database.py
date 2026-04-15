import sqlite3
from contextlib import contextmanager
from config import SQLITE_DB_PATH


@contextmanager
def get_conn():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create all tables if they don't exist."""
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id  TEXT PRIMARY KEY,
            full_name    TEXT NOT NULL,
            email        TEXT,
            phone        TEXT,
            kyc_status   TEXT DEFAULT 'PENDING',
            risk_profile TEXT DEFAULT 'LOW',
            created_at   DATE DEFAULT CURRENT_DATE
        );

        CREATE TABLE IF NOT EXISTS accounts (
            account_id   TEXT PRIMARY KEY,
            customer_id  TEXT NOT NULL REFERENCES customers(customer_id),
            account_type TEXT NOT NULL,
            balance      REAL DEFAULT 0.0,
            currency     TEXT DEFAULT 'INR',
            status       TEXT DEFAULT 'ACTIVE',
            opened_at    DATE DEFAULT CURRENT_DATE
        );

        CREATE TABLE IF NOT EXISTS transactions (
            txn_id      TEXT PRIMARY KEY,
            account_id  TEXT NOT NULL REFERENCES accounts(account_id),
            txn_type    TEXT NOT NULL,
            amount      REAL NOT NULL,
            merchant    TEXT,
            category    TEXT,
            status      TEXT DEFAULT 'SUCCESS',
            txn_at      DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS fraud_flags (
            flag_id     TEXT PRIMARY KEY,
            txn_id      TEXT NOT NULL REFERENCES transactions(txn_id),
            flag_type   TEXT,
            risk_score  REAL,
            resolution  TEXT DEFAULT 'OPEN',
            flagged_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS loans (
            loan_id       TEXT PRIMARY KEY,
            customer_id   TEXT NOT NULL REFERENCES customers(customer_id),
            loan_type     TEXT NOT NULL,
            principal     REAL NOT NULL,
            interest_rate REAL NOT NULL,
            tenure_months INTEGER NOT NULL,
            status        TEXT DEFAULT 'ACTIVE',
            disbursed_at  DATE DEFAULT CURRENT_DATE
        );

        CREATE TABLE IF NOT EXISTS loan_payments (
            payment_id     TEXT PRIMARY KEY,
            loan_id        TEXT NOT NULL REFERENCES loans(loan_id),
            amount_paid    REAL,
            payment_status TEXT DEFAULT 'PENDING',
            due_date       DATE,
            paid_date      DATE
        );

        CREATE TABLE IF NOT EXISTS compliance_cases (
            case_id     TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL REFERENCES customers(customer_id),
            case_type   TEXT,
            severity    TEXT DEFAULT 'LOW',
            status      TEXT DEFAULT 'OPEN',
            assigned_to TEXT,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """)


# ── Query helpers (used as agent tools) ──────────────────────────────────────

def get_account_summary(customer_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM accounts WHERE customer_id = ?", (customer_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_recent_transactions(account_id: str, limit: int = 10) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT * FROM transactions
               WHERE account_id = ?
               ORDER BY txn_at DESC LIMIT ?""",
            (account_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_loan_details(customer_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM loans WHERE customer_id = ?", (customer_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_open_fraud_flags(account_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT ff.* FROM fraud_flags ff
               JOIN transactions t ON ff.txn_id = t.txn_id
               WHERE t.account_id = ? AND ff.resolution = 'OPEN'""",
            (account_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def insert_fraud_flag(txn_id: str, flag_type: str, risk_score: float):
    import uuid
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO fraud_flags (flag_id, txn_id, flag_type, risk_score) VALUES (?,?,?,?)",
            (str(uuid.uuid4()), txn_id, flag_type, risk_score),
        )
