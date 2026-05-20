import sqlite3
def create_database():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT UNIQUE,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        age INTEGER NOT NULL,
        gender INTEGER NOT NULL,
        password TEXT NOT NULL
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id INTEGER,
        category TEXT,
        brand TEXT,
        price REAL,
        purchase_frequency INTEGER,
        customer_satisfaction INTEGER,
        purchase_intent INTEGER
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS purchases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        product_id INTEGER,
        category TEXT,
        brand TEXT,
        price REAL
    )
    """)
    conn.commit()
    conn.close()
    print("Database tables created successfully!")
if __name__ == "__main__":
    create_database()
