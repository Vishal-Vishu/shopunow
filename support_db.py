import sqlite3

import uuid
import time
from datetime import datetime

def save_support_ticket(data):
    print("Inserting user details into the system")
    conn = sqlite3.connect("support_tickets.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone TEXT,
            name TEXT,
            email TEXT,
            issue TEXT,
            original_query TEXT
        )
    """)

    try:
        cursor.execute("""
            INSERT INTO tickets (phone, name, email, issue, original_query)
            VALUES (?, ?, ?, ?, ?)
        """, (
            data["phone"],
            data["name"],
            data["email"],
            data["issue"],
            data["original_query"]
        ))
        print("User details successfully inserted")
    except Exception as e:
        import traceback
        traceback.print_exc()
    conn.commit()
    conn.close()



# ==========================================================
# SESSION TABLE INITIALIZATION
# ==========================================================

def initialize_session_table():
    conn = sqlite3.connect("support_tickets.db")
    cursor = conn.cursor()
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE,
                phone TEXT,
                session_start TEXT,
                session_end TEXT,
                last_activity TEXT,
                message_count INTEGER DEFAULT 0,
                is_active INTEGER DEFAULT 1
            )
        """)
    except:
        import traceback
        traceback.print_exc()

    conn.commit()
    conn.close()


# ==========================================================
# CREATE SESSION
# ==========================================================

def create_user_session(phone):
    initialize_session_table()

    session_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()

    conn = sqlite3.connect("support_tickets.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO user_sessions
        (session_id, phone, session_start, last_activity, message_count, is_active)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (session_id, phone, now, now, 0, 1))

    conn.commit()
    conn.close()

    return session_id


# ==========================================================
# UPDATE ACTIVITY
# ==========================================================

def update_session_activity(session_id):
    now = datetime.utcnow().isoformat()

    conn = sqlite3.connect("support_tickets.db")
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE user_sessions
        SET last_activity = ?,
            message_count = message_count + 1
        WHERE session_id = ?
    """, (now, session_id))

    conn.commit()
    conn.close()


# ==========================================================
# CLOSE SESSION
# ==========================================================

def close_user_session(session_id):
    now = datetime.utcnow().isoformat()

    conn = sqlite3.connect("support_tickets.db")
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE user_sessions
        SET session_end = ?,
            is_active = 0
        WHERE session_id = ?
    """, (now, session_id))

    conn.commit()
    conn.close()


# ==========================================================
# CHECK INACTIVITY
# ==========================================================

def is_session_expired(session_id, timeout_minutes=30):
    conn = sqlite3.connect("support_tickets.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT last_activity FROM user_sessions
        WHERE session_id = ? AND is_active = 1
    """, (session_id,))

    result = cursor.fetchone()
    conn.close()

    if not result:
        return True

    last_activity = datetime.fromisoformat(result[0])
    now = datetime.utcnow()

    inactivity = (now - last_activity).total_seconds() / 60

    return inactivity > timeout_minutes

def cleanup_expired_sessions(timeout_minutes=1):
    conn = sqlite3.connect("support_tickets.db")
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE user_sessions
        SET 
            is_active = 0,
            session_end = datetime('now')
        WHERE 
            is_active = 1
        AND 
            (strftime('%s','now') - strftime('%s', last_activity)) / 60 > ?
    """, (timeout_minutes,))

    conn.commit()
    conn.close()