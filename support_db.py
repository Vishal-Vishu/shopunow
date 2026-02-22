import sqlite3

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
