import sqlite3
from contextlib import contextmanager

DB_PATH = "support_tickets.db"

@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.commit()
        conn.close()


def initialize_conversation_table():
    with get_connection() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS conversation_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone TEXT NOT NULL,
            session_id TEXT NOT NULL,
            query TEXT,
            response TEXT,
            turn_type TEXT,
            department TEXT,
            sentiment TEXT,
            emotion TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)


def append_message(
    phone,
    session_id,
    query,
    response,
    turn_type=None,
    department=None,
    sentiment=None,
    emotion=None
):
    with get_connection() as conn:
        conn.execute("""
        INSERT INTO conversation_messages
        (phone, session_id, query, response, turn_type, department, sentiment, emotion)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            phone,
            session_id,
            query,
            response,
            turn_type,
            department,
            sentiment,
            emotion
        ))


def fetch_session_history(session_id, limit=20):
    with get_connection() as conn:
        rows = conn.execute("""
        SELECT query, response, turn_type
        FROM conversation_messages
        WHERE session_id = ?
        ORDER BY created_at ASC
        LIMIT ?
        """, (session_id, limit)).fetchall()

        return [dict(row) for row in rows]
    
def fetch_phone_history(phone, limit=50):
    with get_connection() as conn:
        rows = conn.execute("""
        SELECT query, response, turn_type
        FROM conversation_messages
        WHERE phone = ?
        ORDER BY created_at ASC
        LIMIT ?
        """, (phone, limit)).fetchall()

        return [dict(row) for row in rows]    