import sqlite3
from datetime import datetime


def init_db():
    conn = sqlite3.connect("escalations.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS escalations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        phone TEXT,
        email TEXT,
        issue TEXT,
        created_at TEXT
    )
    """)

    conn.commit()
    conn.close()


def save_escalation(name, phone, email, issue):
    print("Saving user details which are followed up by human agent")
    conn = sqlite3.connect("escalations.db")
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO escalations (name, phone, email, issue, created_at)
    VALUES (?, ?, ?, ?, ?)
    """, (name, phone, email, issue, datetime.now().isoformat()))

    conn.commit()
    conn.close()