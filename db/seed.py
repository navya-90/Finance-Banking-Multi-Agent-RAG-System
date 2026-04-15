"""
Seed the SQLite database with realistic sample data.
Run once:  python db/seed.py
"""
import sys, uuid
sys.path.insert(0, "..")

from database import init_db, get_conn

def seed():
    init_db()
    with get_conn() as conn:
        # Customers
        conn.executemany(
            "INSERT OR IGNORE INTO customers VALUES (?,?,?,?,?,?,?)",
            [
                ("C001", "Priya Sharma",  "priya@email.com",  "9876543210", "VERIFIED",  "LOW",    "2022-01-15"),
                ("C002", "Rahul Mehta",   "rahul@email.com",  "9123456780", "VERIFIED",  "MEDIUM", "2021-06-20"),
                ("C003", "Anita Desai",   "anita@email.com",  "9988776655", "PENDING",   "HIGH",   "2023-03-10"),
            ],
        )

        # Accounts
        conn.executemany(
            "INSERT OR IGNORE INTO accounts VALUES (?,?,?,?,?,?,?)",
            [
                ("A001", "C001", "SAVINGS",  125000.00, "INR", "ACTIVE", "2022-01-16"),
                ("A002", "C001", "CURRENT",   45000.00, "INR", "ACTIVE", "2022-01-16"),
                ("A003", "C002", "SAVINGS",   88500.50, "INR", "ACTIVE", "2021-06-21"),
                ("A004", "C003", "SAVINGS",    3200.00, "INR", "FROZEN", "2023-03-11"),
            ],
        )

        # Transactions
        txns = [
            ("T001","A001","DEBIT",  1200.00,"Swiggy",       "FOOD",     "SUCCESS","2024-03-10 12:00"),
            ("T002","A001","DEBIT", 15000.00,"Unknown Vendor","UNKNOWN",  "SUCCESS","2024-03-11 03:22"),
            ("T003","A001","CREDIT",50000.00","Salary",       "INCOME",   "SUCCESS","2024-03-01 09:00"),
            ("T004","A002","DEBIT",  4500.00,"Amazon",        "SHOPPING", "SUCCESS","2024-03-09 18:30"),
            ("T005","A003","DEBIT",  2200.00","Zomato",        "FOOD",     "SUCCESS","2024-03-10 20:00"),
            ("T006","A004","DEBIT", 50000.00","Cash Withdrawal","CASH",   "SUCCESS","2024-03-08 11:15"),
        ]
        conn.executemany(
            "INSERT OR IGNORE INTO transactions VALUES (?,?,?,?,?,?,?,?)", txns
        )

        # Fraud flag on suspicious T002 and T006
        conn.executemany(
            "INSERT OR IGNORE INTO fraud_flags VALUES (?,?,?,?,?,?)",
            [
                ("FF001","T002","UNUSUAL_HOUR",  0.88,"OPEN","2024-03-11 03:23"),
                ("FF002","T006","LARGE_CASH",    0.72,"OPEN","2024-03-08 11:16"),
            ],
        )

        # Loans
        conn.executemany(
            "INSERT OR IGNORE INTO loans VALUES (?,?,?,?,?,?,?,?)",
            [
                ("L001","C001","HOME_LOAN",  3000000.00, 8.5, 240,"ACTIVE","2023-06-01"),
                ("L002","C002","PERSONAL",    500000.00,12.0,  36,"ACTIVE","2024-01-15"),
            ],
        )

        # Loan payments
        conn.executemany(
            "INSERT OR IGNORE INTO loan_payments VALUES (?,?,?,?,?,?)",
            [
                ("LP001","L001",26500.00,"PAID",   "2024-03-01","2024-03-01"),
                ("LP002","L001",26500.00,"OVERDUE","2024-04-01", None),
                ("LP003","L002",16600.00,"PAID",   "2024-03-15","2024-03-14"),
            ],
        )

        # Compliance case for high-risk customer
        conn.execute(
            "INSERT OR IGNORE INTO compliance_cases VALUES (?,?,?,?,?,?,?)",
            ("CC001","C003","KYC_EXPIRY","HIGH","OPEN","compliance_team","2024-03-12 10:00"),
        )

    print("Seed complete.")

if __name__ == "__main__":
    seed()
