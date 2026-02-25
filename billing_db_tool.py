import sqlite3
import re

DB_PATH = "support_tickets.db"

def extract_order_id(text: str):
    match = re.search(r"ORD-\d+", text)
    return match.group(0) if match else None


def fetch_order_details(order_id: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get order
    cursor.execute("""
        SELECT order_id, customer_id, order_date, payment_mode, subtotal, tax, discount, total
        FROM orders
        WHERE order_id = ?
    """, (order_id,))
    order = cursor.fetchone()

    if not order:
        conn.close()
        return None

    # Get items
    cursor.execute("""
        SELECT product_name, category, quantity, unit_price, warranty_months
        FROM order_items
        WHERE order_id = ?
    """, (order_id,))
    items = cursor.fetchall()

    conn.close()

    return {
        "order": order,
        "items": items
    }