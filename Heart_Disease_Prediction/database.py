"""
database.py
--------------------------------------------------------------
Central SQLite database module for the Heart Disease Prediction
web application.

Responsibilities:
    1. Create/maintain the `users` table (authentication module)
    2. Create/maintain the `predictions` table (existing ML feature)
    3. Provide small helper functions used by app.py for:
       - user registration / login
       - profile management
       - prediction history (list / delete / count)

NOTE: The existing `predictions` table and its columns used by the
original ML/SHAP/PDF pipeline are NOT modified. We only ADD a new
`user_id` column (via migration) so a prediction can be linked to
the logged-in user. Nothing that already existed is removed.
--------------------------------------------------------------
"""

import sqlite3

DB_NAME = "heart_disease.db"


def get_connection():
    """Return a new SQLite connection with Row factory for dict-like access."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


# ------------------------------------------------------------------
# TABLE CREATION
# ------------------------------------------------------------------
def create_database():
    """Create all required tables if they do not already exist."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # ---------------- USERS TABLE (Authentication Module) ----------------
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fullname TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        mobile TEXT NOT NULL,
        password TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # ---------------- PREDICTIONS TABLE (existing ML feature) ----------------
    # Kept 100% identical to the original schema. `user_id` is included
    # here for fresh installs; for an already-existing DB file the
    # migration step below adds the column safely.
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        age INTEGER,
        sex TEXT,
        chest_pain TEXT,
        resting_bp INTEGER,
        cholesterol INTEGER,
        fasting_bs INTEGER,
        rest_ecg TEXT,
        max_hr INTEGER,
        exercise_angina TEXT,
        oldpeak REAL,
        st_slope TEXT,
        ca INTEGER,
        thal TEXT,
        prediction TEXT,
        risk_level TEXT,
        confidence REAL,
        prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

    # Run migration in case an older DB file (without user_id) already exists
    _migrate_add_user_id_column()


def _migrate_add_user_id_column():
    """
    Safely add the `user_id` column to an already-existing `predictions`
    table (from before the authentication module existed) without
    touching or deleting any existing prediction rows.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(predictions)")
    columns = [row[1] for row in cursor.fetchall()]

    if "user_id" not in columns:
        cursor.execute("ALTER TABLE predictions ADD COLUMN user_id INTEGER")
        conn.commit()

    conn.close()


# ------------------------------------------------------------------
# USER (AUTHENTICATION) HELPERS
# ------------------------------------------------------------------
def add_user(fullname, email, mobile, hashed_password):
    """Insert a new user. Returns the new user's id."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO users (fullname, email, mobile, password)
        VALUES (?, ?, ?, ?)
    """, (fullname, email, mobile, hashed_password))
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id


def get_user_by_email(email):
    """Return a user row (or None) matching the given email."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()
    return user


def get_user_by_id(user_id):
    """Return a user row (or None) matching the given id."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()
    return user


def email_exists(email):
    """Return True if a user with this email already exists."""
    return get_user_by_email(email) is not None


def update_user_profile(user_id, fullname, mobile):
    """Update a user's editable profile fields."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE users SET fullname = ?, mobile = ? WHERE id = ?
    """, (fullname, mobile, user_id))
    conn.commit()
    conn.close()


def update_user_password(user_id, hashed_password):
    """Update a user's password hash."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE users SET password = ? WHERE id = ?
    """, (hashed_password, user_id))
    conn.commit()
    conn.close()


# ------------------------------------------------------------------
# PREDICTION HISTORY HELPERS (built on top of the existing table)
# ------------------------------------------------------------------
def get_predictions_by_user(user_id):
    """Return all predictions for a given user, most recent first."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM predictions
        WHERE user_id = ?
        ORDER BY prediction_date DESC
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_prediction_by_id(prediction_id, user_id):
    """Return a single prediction that belongs to the given user."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM predictions WHERE id = ? AND user_id = ?
    """, (prediction_id, user_id))
    row = cursor.fetchone()
    conn.close()
    return row


def delete_prediction(prediction_id, user_id):
    """Delete a prediction record (only if it belongs to the user)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        DELETE FROM predictions WHERE id = ? AND user_id = ?
    """, (prediction_id, user_id))
    conn.commit()
    conn.close()


def get_prediction_count(user_id):
    """Return the total number of predictions made by a user."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*) as total FROM predictions WHERE user_id = ?
    """, (user_id,))
    total = cursor.fetchone()["total"]
    conn.close()
    return total


def get_latest_prediction(user_id):
    """Return the most recent prediction made by a user (or None)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM predictions
        WHERE user_id = ?
        ORDER BY prediction_date DESC
        LIMIT 1
    """, (user_id,))
    row = cursor.fetchone()
    conn.close()
    return row


if __name__ == "__main__":
    create_database()
    print("Database initialized / migrated successfully.")
