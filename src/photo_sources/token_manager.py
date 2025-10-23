"""
TokenManager - Secure OAuth Token Storage

Manages OAuth tokens with encryption at rest using Fernet symmetric encryption.
Encryption keys stored in OS keyring for maximum security.

Security Features:
- Tokens encrypted with Fernet (AES 128)
- Encryption key stored in OS keyring (not in files)
- SQLite database with encrypted token fields
- Automatic token refresh
- Secure revocation and cleanup
"""

import json
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
import keyring


class TokenManager:
    """
    Manages OAuth tokens with encryption.

    Token Lifecycle:
    1. User authenticates → receive access_token + refresh_token
    2. Store encrypted in SQLite
    3. Auto-refresh before expiry
    4. Revoke and cleanup on disconnect

    Storage Format:
    - Database: SQLite
    - Encryption: Fernet (symmetric)
    - Key Storage: OS keyring
    """

    KEYRING_SERVICE = "RememberTwelve"
    KEYRING_USERNAME = "google_photos_encryption"

    def __init__(self, storage_path: Path):
        """
        Initialize TokenManager.

        Args:
            storage_path: Path to SQLite database file
                         (e.g., ~/.remember_twelve/tokens.db)
        """
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Get or create encryption key
        self.encryption_key = self._get_or_create_key()
        self.cipher = Fernet(self.encryption_key)

        # Initialize database
        self._init_database()

    def _get_or_create_key(self) -> bytes:
        """
        Get encryption key from OS keyring or create new one.

        Returns:
            Fernet encryption key (32 bytes, base64 encoded)
        """
        # Try to get existing key from keyring
        key_str = keyring.get_password(
            self.KEYRING_SERVICE,
            self.KEYRING_USERNAME
        )

        if key_str:
            return key_str.encode()

        # Create new key
        new_key = Fernet.generate_key()
        keyring.set_password(
            self.KEYRING_SERVICE,
            self.KEYRING_USERNAME,
            new_key.decode()
        )
        return new_key

    def _init_database(self) -> None:
        """Initialize SQLite database with schema."""
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tokens (
                    user_email TEXT PRIMARY KEY,
                    encrypted_token TEXT NOT NULL,
                    token_type TEXT NOT NULL,
                    expires_at TEXT,
                    stored_at TEXT NOT NULL,
                    last_refreshed TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at
                ON tokens(expires_at)
            """)
            conn.commit()

    def store_token(
        self,
        user_email: str,
        token_data: Dict[str, Any]
    ) -> None:
        """
        Store encrypted token for user.

        Args:
            user_email: User's email (identifier)
            token_data: OAuth token data with keys:
                - access_token: str
                - refresh_token: str (optional)
                - expires_in: int (seconds)
                - token_type: str (usually 'Bearer')
                - scope: str

        Security:
            - Token serialized to JSON
            - Encrypted with Fernet
            - Stored in SQLite with user_email as key
        """
        # Calculate expiry time
        expires_in = token_data.get('expires_in', 3600)  # Default 1 hour
        expires_at = datetime.now() + timedelta(seconds=expires_in)

        # Serialize and encrypt token
        token_json = json.dumps(token_data)
        encrypted = self.cipher.encrypt(token_json.encode())

        # Store in database
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO tokens
                (user_email, encrypted_token, token_type, expires_at, stored_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                user_email,
                encrypted.decode(),
                token_data.get('token_type', 'Bearer'),
                expires_at.isoformat(),
                datetime.now().isoformat()
            ))
            conn.commit()

    def get_token(self, user_email: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve and decrypt token for user.

        Args:
            user_email: User's email

        Returns:
            Token data dict or None if not found

        Note: Does not check expiry - use is_token_valid() first
        """
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.execute("""
                SELECT encrypted_token FROM tokens
                WHERE user_email = ?
            """, (user_email,))
            row = cursor.fetchone()

            if not row:
                return None

            # Decrypt token
            encrypted = row[0].encode()
            decrypted = self.cipher.decrypt(encrypted)
            return json.loads(decrypted.decode())

    def is_token_valid(self, user_email: str) -> bool:
        """
        Check if token exists and is not expired.

        Args:
            user_email: User's email

        Returns:
            True if token valid, False otherwise
        """
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.execute("""
                SELECT expires_at FROM tokens
                WHERE user_email = ?
            """, (user_email,))
            row = cursor.fetchone()

            if not row or not row[0]:
                return False

            expires_at = datetime.fromisoformat(row[0])
            # Consider expired if less than 5 minutes remaining
            return expires_at > datetime.now() + timedelta(minutes=5)

    def revoke_token(self, user_email: str) -> None:
        """
        Revoke token and remove from storage.

        Args:
            user_email: User's email

        This should be called when user disconnects Google Photos.
        The caller is responsible for calling Google's revoke endpoint.
        """
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute("""
                DELETE FROM tokens WHERE user_email = ?
            """, (user_email,))
            conn.commit()

    def update_last_refreshed(self, user_email: str) -> None:
        """
        Update last_refreshed timestamp.

        Called after successful token refresh.
        """
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute("""
                UPDATE tokens
                SET last_refreshed = ?
                WHERE user_email = ?
            """, (datetime.now().isoformat(), user_email))
            conn.commit()

    def clear_all(self) -> None:
        """
        Clear all tokens from database.

        Use with caution - removes all stored credentials.
        """
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute("DELETE FROM tokens")
            conn.commit()

    def list_users(self) -> list[str]:
        """
        List all users with stored tokens.

        Returns:
            List of user emails
        """
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.execute("SELECT user_email FROM tokens")
            return [row[0] for row in cursor.fetchall()]

    def __del__(self):
        """Cleanup - ensure database is closed."""
        # SQLite connections auto-close, but explicit is better
        pass
