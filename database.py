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


def _table_has_column(cursor, table_name, column_name):
    cursor.execute(f"PRAGMA table_info({table_name})")
    return any(row[1] == column_name for row in cursor.fetchall())


def _documents_has_unique_filename(cursor, table_name):
    cursor.execute(f"PRAGMA index_list({table_name})")
    for row in cursor.fetchall():
        index_name = row[1]
        unique = row[2]
        if unique:
            cursor.execute(f"PRAGMA index_info({index_name})")
            columns = [index_row[2] for index_row in cursor.fetchall()]
            if columns == ["filename"]:
                return True
    return False


def init_db():
    conn = get_connection()
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user'
        )
    ''')

    # Ensure documents table schema supports owner isolation.
    c.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT NOT NULL DEFAULT 'active',
            visibility TEXT NOT NULL DEFAULT 'public',
            owner_id INTEGER NOT NULL DEFAULT 1,
            FOREIGN KEY(owner_id) REFERENCES users(id),
            UNIQUE(filename, owner_id)
        )
    ''')

    # Migrate legacy documents table schema to include owner_id and composite uniqueness.
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='documents_old'")
    legacy_exists = c.fetchone() is not None
    if not legacy_exists:
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='documents'")
        if c.fetchone() is not None:
            cursor = c
            if not _table_has_column(cursor, 'documents', 'owner_id') or _documents_has_unique_filename(cursor, 'documents'):
                c.execute('ALTER TABLE documents RENAME TO documents_old')
                c.execute('''
                    CREATE TABLE documents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT NOT NULL,
                        upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        status TEXT NOT NULL DEFAULT 'active',
                        visibility TEXT NOT NULL DEFAULT 'public',
                        owner_id INTEGER NOT NULL DEFAULT 1,
                        FOREIGN KEY(owner_id) REFERENCES users(id),
                        UNIQUE(filename, owner_id)
                    )
                ''')
                columns = [row[1] for row in cursor.execute("PRAGMA table_info(documents_old)").fetchall()]
                select_columns = [col for col in ["filename", "upload_date", "status", "visibility"] if col in columns]
                if "owner_id" in columns:
                    select_columns.append("owner_id")
                insert_columns = ", ".join(["filename", "upload_date", "status", "visibility", "owner_id"])
                if "owner_id" in columns:
                    cursor.execute(f"INSERT INTO documents ({insert_columns}) SELECT filename, upload_date, status, visibility, owner_id FROM documents_old")
                else:
                    cursor.execute(f"INSERT INTO documents ({insert_columns}) SELECT filename, upload_date, status, visibility, 1 FROM documents_old")
                c.execute("DROP TABLE documents_old")

    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(session_id) REFERENCES chat_sessions(session_id),
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')

    # Add missing owner_id column to legacy chat_history table if needed.
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chat_history_old'")
    legacy_chat_exists = c.fetchone() is not None
    if not legacy_chat_exists:
        if _table_has_column(c, 'chat_history', 'session_id') and not _table_has_column(c, 'chat_history', 'user_id'):
            c.execute('ALTER TABLE chat_history RENAME TO chat_history_old')
            c.execute('''
                CREATE TABLE chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_id INTEGER NOT NULL DEFAULT 1,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES chat_sessions(session_id),
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            ''')
            c.execute('INSERT INTO chat_history (session_id, user_id, role, content, timestamp) SELECT session_id, 1, role, content, timestamp FROM chat_history_old')
            c.execute('DROP TABLE chat_history_old')

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
    c.execute("SELECT id, username, role FROM users WHERE username = ? AND password_hash = ?",
              (username, hash_password(password)))
    user = c.fetchone()
    conn.close()
    return dict(user) if user else None


def seed_admin_if_needed(admin_password):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
    if c.fetchone()[0] == 0:
        create_user("admin", admin_password, "admin")
    conn.close()


# --- Documents ---
def add_document(filename, owner_id, status='active', visibility='public'):
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO documents (filename, status, visibility, owner_id) VALUES (?, ?, ?, ?)",
            (filename, status, visibility, owner_id)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def remove_document(filename, owner_id=None):
    conn = get_connection()
    c = conn.cursor()
    if owner_id is None:
        c.execute("DELETE FROM documents WHERE filename = ?", (filename,))
    else:
        c.execute("DELETE FROM documents WHERE filename = ? AND owner_id = ?", (filename, owner_id))
    conn.commit()
    conn.close()


def get_user_documents(user_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM documents WHERE owner_id = ? ORDER BY upload_date DESC", (user_id,))
    docs = c.fetchall()
    conn.close()
    return [dict(d) for d in docs]


def get_all_documents(is_admin=False, user_id=None):
    if is_admin:
        conn = get_connection()
        c = conn.cursor()
        c.execute("SELECT * FROM documents ORDER BY upload_date DESC")
        docs = c.fetchall()
        conn.close()
        return [dict(d) for d in docs]
    return get_user_documents(user_id)


def get_document_owner(filename, owner_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM documents WHERE filename = ? AND owner_id = ?", (filename, owner_id))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None


# --- Sessions ---
def create_chat_session(session_id, user_id):
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute("INSERT OR IGNORE INTO chat_sessions (session_id, user_id) VALUES (?, ?)", (session_id, user_id))
        conn.commit()
    finally:
        conn.close()


def add_chat_message(session_id, role, content, user_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "INSERT INTO chat_history (session_id, user_id, role, content) VALUES (?, ?, ?, ?)",
        (session_id, user_id, role, content)
    )
    conn.commit()
    conn.close()


def get_chat_history(session_id, user_id=None, is_admin=False):
    conn = get_connection()
    c = conn.cursor()
    if is_admin:
        c.execute("SELECT role, content FROM chat_history WHERE session_id = ? ORDER BY timestamp ASC", (session_id,))
    else:
        c.execute(
            "SELECT role, content FROM chat_history WHERE session_id = ? AND user_id = ? ORDER BY timestamp ASC",
            (session_id, user_id)
        )
    history = c.fetchall()
    conn.close()
    return [dict(row) for row in history]


def delete_chat_session(session_id, user_id=None, is_admin=False):
    conn = get_connection()
    c = conn.cursor()
    if is_admin:
        c.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
        c.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
    else:
        c.execute("DELETE FROM chat_history WHERE session_id = ? AND user_id = ?", (session_id, user_id))
        c.execute("DELETE FROM chat_sessions WHERE session_id = ? AND user_id = ?", (session_id, user_id))
    conn.commit()
    conn.close()


def get_all_sessions(user_id=None, is_admin=False):
    conn = get_connection()
    c = conn.cursor()
    if is_admin:
        c.execute('''
            SELECT cs.session_id,
                   cs.user_id,
                   u.username,
                   MIN(ch.timestamp) as start_time,
                   (SELECT content FROM chat_history ch2 WHERE ch2.session_id = cs.session_id AND role='user' ORDER BY timestamp ASC LIMIT 1) as title
            FROM chat_sessions cs
            LEFT JOIN users u ON u.id = cs.user_id
            LEFT JOIN chat_history ch ON ch.session_id = cs.session_id
            GROUP BY cs.session_id
            ORDER BY start_time DESC
        ''')
    else:
        c.execute('''
            SELECT cs.session_id,
                   cs.user_id,
                   u.username,
                   MIN(ch.timestamp) as start_time,
                   (SELECT content FROM chat_history ch2 WHERE ch2.session_id = cs.session_id AND role='user' ORDER BY timestamp ASC LIMIT 1) as title
            FROM chat_sessions cs
            LEFT JOIN users u ON u.id = cs.user_id
            LEFT JOIN chat_history ch ON ch.session_id = cs.session_id
            WHERE cs.user_id = ?
            GROUP BY cs.session_id
            ORDER BY start_time DESC
        ''', (user_id,))
    sessions = c.fetchall()
    conn.close()
    return [dict(row) for row in sessions]


def get_admin_messages():
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        SELECT ch.role, ch.content, ch.timestamp
        FROM chat_history ch
        JOIN chat_sessions cs ON cs.session_id = ch.session_id
        JOIN users u ON u.id = cs.user_id
        WHERE u.role = 'admin'
        ORDER BY ch.timestamp ASC
    ''')
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]
