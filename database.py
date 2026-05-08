import sqlite3
import os
import hashlib
from datetime import datetime

DB_PATH = os.path.join("data", "hallurag.db")

def get_connection():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    c = conn.cursor()
    
    # Create users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user'
        )
    ''')
    
    # Create documents table
    c.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE NOT NULL,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT NOT NULL DEFAULT 'active',
            visibility TEXT NOT NULL DEFAULT 'public'
        )
    ''')
    
    # Try to add visibility column for existing DBs
    try:
        c.execute("ALTER TABLE documents ADD COLUMN visibility TEXT NOT NULL DEFAULT 'public'")
    except sqlite3.OperationalError:
        pass
    
    # Create chat history table
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

# --- Users ---
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password, role='user'):
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)", 
                  (username, hash_password(password), role))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def authenticate_user(username, password):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, role FROM users WHERE username = ? AND password_hash = ?", 
              (username, hash_password(password)))
    user = c.fetchone()
    conn.close()
    return dict(user) if user else None

def seed_admin_if_needed(admin_password):
    """Seed the admin user from .env if no users exist"""
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users")
    if c.fetchone()[0] == 0:
        create_user("admin", admin_password, "admin")
    conn.close()

# --- Documents ---
def add_document(filename, status='active', visibility='public'):
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO documents (filename, status, visibility) VALUES (?, ?, ?)", (filename, status, visibility))
        conn.commit()
    except sqlite3.IntegrityError:
        pass # Already exists
    finally:
        conn.close()

def remove_document(filename):
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM documents WHERE filename = ?", (filename,))
    conn.commit()
    conn.close()

def get_all_documents():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM documents ORDER BY upload_date DESC")
    docs = c.fetchall()
    conn.close()
    return [dict(d) for d in docs]

# --- Chat History ---
def add_chat_message(session_id, role, content):
    conn = get_connection()
    c = conn.cursor()
    c.execute("INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)", 
              (session_id, role, content))
    conn.commit()
    conn.close()

def get_chat_history(session_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT role, content FROM chat_history WHERE session_id = ? ORDER BY timestamp ASC", (session_id,))
    history = c.fetchall()
    conn.close()
    return [dict(row) for row in history]

def delete_chat_session(session_id):
    """
    Deletes a chat session and all associated messages.
    Removes all messages linked to the session_id and commits changes safely.
    """
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
        conn.commit()
    finally:
        conn.close()

def get_all_sessions():
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        SELECT ch1.session_id, MIN(ch1.timestamp) as start_time, 
        (SELECT content FROM chat_history ch2 WHERE ch2.session_id = ch1.session_id AND role='user' ORDER BY timestamp ASC LIMIT 1) as title
        FROM chat_history ch1
        GROUP BY ch1.session_id
        ORDER BY start_time DESC
    ''')
    sessions = c.fetchall()
    conn.close()
    return [dict(row) for row in sessions]
