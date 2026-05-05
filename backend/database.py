import sqlite3
import os

DB_PATH = "backend/feedback.db"

def init_db():
    os.makedirs("backend", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Bảng chứa phản hồi của người dùng
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedbacks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT,
            recency REAL,
            frequency REAL,
            monetary REAL,
            predicted_cluster INTEGER,
            correct BOOLEAN,
            user_suggested_cluster INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Bảng ghi lại lịch sử Retrain
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS retrain_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trigger TEXT,
            status TEXT,
            note TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    print("✅ Database SQLite đã khởi tạo thành công.")

def insert_feedback(customer_id, r, f, m, predicted_cluster, correct, user_suggested_cluster):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO feedbacks (customer_id, recency, frequency, monetary, predicted_cluster, correct, user_suggested_cluster)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (customer_id, r, f, m, predicted_cluster, correct, user_suggested_cluster))
    conn.commit()
    conn.close()

def get_feedbacks():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM feedbacks")
    rows = cursor.fetchall()
    conn.close()
    return rows

def count_wrong_feedbacks():
    """Đếm số feedback báo sai cụm (chưa được dùng để retrain)."""
    if not os.path.exists(DB_PATH):
        return 0
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM feedbacks WHERE correct = 0")
    count = cursor.fetchone()[0]
    conn.close()
    return count

def insert_retrain_log(trigger: str, status: str, note: str = ""):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO retrain_logs (trigger, status, note) VALUES (?, ?, ?)",
        (trigger, status, note)
    )
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
