import sqlite3
from datetime import datetime

DB_FILE = "expenses.db"

def setup_database():
    """Initializes the database and creates tables if they don't exist."""
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()

    # Create the main invoices table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS invoices (
        transaction_id TEXT PRIMARY KEY,
        invoice_date TEXT,
        store_name TEXT,
        store_location TEXT,
        total_amount REAL
    )
    """)

    # Create a table for items, linked to the invoices table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS items (
        item_id INTEGER PRIMARY KEY AUTOINCREMENT,
        invoice_id TEXT NOT NULL,
        name TEXT,
        quantity REAL,
        unit_price REAL,
        total_price REAL,
        FOREIGN KEY (invoice_id) REFERENCES invoices (transaction_id)
    )
    """)

    con.commit()
    con.close()
    print("Database initialized successfully.")


def save_invoice(invoice_data: dict):
    """Saves a single invoice and its items to the database."""
    transaction_id = invoice_data.get("transaction_id")
    if not transaction_id:
        print("Error: Invoice data missing transaction_id.")
        return

    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()

    # Check if invoice already exists to prevent duplicates
    cur.execute("SELECT transaction_id FROM invoices WHERE transaction_id = ?", (transaction_id,))
    if cur.fetchone():
        print(f"Invoice {transaction_id} already exists in the database. Skipping.")
        con.close()
        return

    try:
        # Insert main invoice details
        cur.execute("""
        INSERT INTO invoices (
            transaction_id, invoice_date, store_name, store_location,
            total_amount
        ) VALUES (?, ?, ?, ?, ?)
        """, (
            transaction_id,
            invoice_data.get("invoice_date"),
            invoice_data.get("store_name"),
            invoice_data.get("store_location"),
            invoice_data.get("total_amount")
        ))

        # Insert each item linked to the invoice
        items = invoice_data.get("items", [])
        for item in items:
            cur.execute("""
                INSERT INTO items (invoice_id, name, quantity, unit_price, total_price)
                VALUES (?, ?, ?, ?, ?)
            """, (
                transaction_id,
                item.get("name"),
                item.get("quantity"),
                item.get("unit_price"),
                item.get("total_price")
            ))

        con.commit()
        print(f"Successfully saved invoice {transaction_id} to the database.")
    except sqlite3.Error as e:
        con.rollback()  # Roll back changes if any error occurs
        print(f"Database error: {e}")
    finally:
        con.close()
