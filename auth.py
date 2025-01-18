import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash  # type: ignore
import os

# Database file path
DATABASE = os.getenv("AUTH_DATABASE", "users.db")  # Use environment variable or default to "users.db"

def init_db() -> None:
    """
    Initialize the database by creating the users table if it doesn't already exist.
    """
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
            """
        )
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database initialization error: {e}")
    finally:
        conn.close()

def register_user(username: str, password: str) -> bool:
    """
    Register a new user with a hashed password.
    
    Args:
        username (str): The username of the user.
        password (str): The plaintext password to be hashed and stored.
    
    Returns:
        bool: True if the registration was successful, False if the username already exists.
    """
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        # Generate hash of the password
        password_hash = generate_password_hash(password)

        cursor.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, password_hash),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # Username already exists
        return False
    except sqlite3.Error as e:
        print(f"Error registering user: {e}")
        return False
    finally:
        conn.close()

def validate_user(username: str, password: str) -> bool:
    """
    Validate a user's credentials.
    
    Args:
        username (str): The username to validate.
        password (str): The plaintext password to check against the stored hash.
    
    Returns:
        bool: True if the user exists and the password is correct, otherwise False.
    """
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()

        if row is None:
            return False  # Username does not exist

        stored_hash = row[0]
        return check_password_hash(stored_hash, password)
    except sqlite3.Error as e:
        print(f"Error validating user: {e}")
        return False
    finally:
        conn.close()
