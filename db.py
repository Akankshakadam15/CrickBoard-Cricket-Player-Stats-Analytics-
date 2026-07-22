"""
db.py — SQLite-backed users + favorites for CrickBoard.

Handles account creation, login verification, and per-user favorites that
persist across sessions and devices (unlike the plain st.session_state
favorites used when nobody is logged in).

Note on security: passwords are hashed with SHA-256 + a random per-user salt,
which is fine for a student project demo. For a real production app, use a
proper password-hashing library (e.g. bcrypt or argon2) instead — SHA-256 is
fast, which is actually a *weakness* for password hashing at scale.
"""

import hashlib
import os
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "crickboard.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            salt TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS favorites (
            user_id INTEGER NOT NULL,
            player_id TEXT NOT NULL,
            PRIMARY KEY (user_id, player_id),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )
    conn.commit()
    conn.close()


def _hash_password(password, salt=None):
    if salt is None:
        salt = os.urandom(16).hex()
    digest = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
    return salt, digest


def register_user(username, password):
    """Returns (success: bool, message: str)."""
    username = username.strip()
    if not username or not password:
        return False, "Username and password can't be empty."
    if len(password) < 4:
        return False, "Password must be at least 4 characters."

    conn = get_connection()
    existing = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
    if existing:
        conn.close()
        return False, "That username is already taken."

    salt, password_hash = _hash_password(password)
    conn.execute(
        "INSERT INTO users (username, salt, password_hash) VALUES (?, ?, ?)",
        (username, salt, password_hash),
    )
    conn.commit()
    conn.close()
    return True, "Account created — you can log in now."


def authenticate(username, password):
    """Returns the user_id on success, or None on failure."""
    conn = get_connection()
    row = conn.execute(
        "SELECT id, salt, password_hash FROM users WHERE username = ?", (username,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    _, check_hash = _hash_password(password, salt=row["salt"])
    if check_hash == row["password_hash"]:
        return row["id"]
    return None


def get_favorites(user_id):
    conn = get_connection()
    rows = conn.execute(
        "SELECT player_id FROM favorites WHERE user_id = ?", (user_id,)
    ).fetchall()
    conn.close()
    return [r["player_id"] for r in rows]


def toggle_favorite(user_id, player_id):
    conn = get_connection()
    existing = conn.execute(
        "SELECT 1 FROM favorites WHERE user_id = ? AND player_id = ?", (user_id, player_id)
    ).fetchone()
    if existing:
        conn.execute(
            "DELETE FROM favorites WHERE user_id = ? AND player_id = ?", (user_id, player_id)
        )
    else:
        conn.execute(
            "INSERT INTO favorites (user_id, player_id) VALUES (?, ?)", (user_id, player_id)
        )
    conn.commit()
    conn.close()