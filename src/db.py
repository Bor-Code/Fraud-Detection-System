import sqlite3
from typing import Dict, Any

DB_PATH = "transactions.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time REAL,
            amount REAL,
            probability REAL,
            decision INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_prediction(tx: Dict[str, Any], prob: float, pred: int):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO predictions (time, amount, probability, decision) VALUES (?, ?, ?, ?)",
        (tx.get("Time", 0), tx.get("Amount", 0), prob, pred)
    )
    conn.commit()
    conn.close()

init_db()
