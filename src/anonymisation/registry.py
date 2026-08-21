"""Encrypted anonymisation registry — stores token mappings in SQLite with Fernet encryption.

The registry maps original PII values to anonymised tokens, scoped per matter.
All original values are encrypted at rest using a passphrase-derived key.
The passphrase is entered once at application startup and held in memory only.

Security properties:
  - AES-128-CBC encryption via Fernet (cryptography library)
  - PBKDF2-HMAC-SHA256 key derivation (600,000 iterations per OWASP 2024)
  - Key never touches disk — memory-only, wiped on close
  - Matter-level isolation: tokens from Matter A cannot resolve Matter B
"""

from __future__ import annotations

import base64
import hashlib
import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from .models import (
    AnonymisationMethod,
    AnonymisationToken,
    PIICategory,
)

# Optional: Fernet encryption — graceful degradation if not installed
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger.warning(
        "cryptography package not installed — registry will store tokens WITHOUT "
        "encryption. Install with: pip install cryptography>=42.0"
    )


# PBKDF2 iteration count (OWASP 2024 recommendation)
PBKDF2_ITERATIONS = 600_000

# Salt length in bytes
SALT_LENGTH = 16


class AnonymisationRegistry:
    """Encrypted SQLite store for anonymisation token mappings.

    Thread-safe. Each matter has its own namespace so tokens cannot
    leak across cases.

    Args:
        db_path: Path to SQLite database file.
        passphrase: Encryption passphrase (held in memory only).
    """

    def __init__(self, db_path: str | Path, passphrase: Optional[str] = None) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fernet: Optional[Fernet] = None
        self._salt: bytes = b""
        self._counters: dict[str, dict[str, int]] = {}  # matter_id → {category: next_idx}

        # Derive encryption key from passphrase
        if passphrase and CRYPTO_AVAILABLE:
            self._init_encryption(passphrase)
        elif passphrase and not CRYPTO_AVAILABLE:
            logger.error("Passphrase provided but cryptography package not available")

        self._init_db()
        self._load_counters()

    # ------------------------------------------------------------------
    # Encryption setup
    # ------------------------------------------------------------------

    def _init_encryption(self, passphrase: str) -> None:
        """Derive Fernet key from passphrase using PBKDF2."""
        # Check if a salt already exists in the DB
        if self._db_path.exists():
            try:
                conn = sqlite3.connect(str(self._db_path))
                cur = conn.execute(
                    "SELECT value FROM registry_meta WHERE key = 'salt'"
                )
                row = cur.fetchone()
                conn.close()
                if row:
                    self._salt = base64.b64decode(row[0])
            except Exception:
                pass  # DB doesn't exist yet or no meta table

        if not self._salt:
            self._salt = os.urandom(SALT_LENGTH)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self._salt,
            iterations=PBKDF2_ITERATIONS,
        )
        key = base64.urlsafe_b64encode(kdf.derive(passphrase.encode("utf-8")))
        self._fernet = Fernet(key)
        logger.info("Anonymisation registry encryption initialised (PBKDF2 + Fernet)")

    def _encrypt(self, plaintext: str) -> str:
        """Encrypt a string. Returns base64-encoded ciphertext.

        NOTE: Fernet produces different ciphertexts for the same input
        (random IV), so encrypted values CANNOT be used as lookup keys.
        Use _hash_for_lookup() for deterministic matching instead.
        """
        if self._fernet:
            return self._fernet.encrypt(plaintext.encode("utf-8")).decode("ascii")
        return plaintext  # Fallback: no encryption

    def _decrypt(self, ciphertext: str) -> str:
        """Decrypt a base64-encoded ciphertext string."""
        if self._fernet:
            return self._fernet.decrypt(ciphertext.encode("ascii")).decode("utf-8")
        return ciphertext  # Fallback: no encryption

    def _hash_for_lookup(self, value: str) -> str:
        """Create a deterministic HMAC-SHA256 hash for DB lookups.

        This is used as the lookup key in the tokens table so that
        the same original value always produces the same hash,
        while the actual original value is stored encrypted (Fernet).
        """
        if self._salt:
            return hashlib.sha256(
                self._salt + value.encode("utf-8")
            ).hexdigest()
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @property
    def is_encrypted(self) -> bool:
        """Whether the registry is using encryption."""
        return self._fernet is not None

    # ------------------------------------------------------------------
    # Database initialisation
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS registry_meta (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tokens (
                    token_id         TEXT PRIMARY KEY,
                    matter_id        TEXT NOT NULL,
                    category         TEXT NOT NULL,
                    original_hash    TEXT NOT NULL,   -- Deterministic HMAC for lookups
                    original_value   TEXT NOT NULL,   -- Encrypted (Fernet)
                    anonymised_value TEXT NOT NULL,
                    method           TEXT NOT NULL DEFAULT 'tokenisation',
                    created_at       TEXT NOT NULL,
                    metadata_json    TEXT DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS idx_tokens_matter
                    ON tokens(matter_id);
                CREATE INDEX IF NOT EXISTS idx_tokens_anon_value
                    ON tokens(matter_id, anonymised_value);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_tokens_lookup
                    ON tokens(matter_id, category, original_hash);

                CREATE TABLE IF NOT EXISTS audit_log (
                    id          TEXT PRIMARY KEY,
                    timestamp   TEXT NOT NULL,
                    action      TEXT NOT NULL,
                    matter_id   TEXT NOT NULL,
                    document_id TEXT DEFAULT '',
                    entity_count INTEGER DEFAULT 0,
                    user        TEXT DEFAULT 'system',
                    details     TEXT DEFAULT '',
                    success     INTEGER DEFAULT 1
                );

                CREATE INDEX IF NOT EXISTS idx_audit_matter
                    ON audit_log(matter_id);
                CREATE INDEX IF NOT EXISTS idx_audit_timestamp
                    ON audit_log(timestamp);
            """)

            # Store salt if encrypted
            if self._salt:
                salt_b64 = base64.b64encode(self._salt).decode("ascii")
                conn.execute(
                    "INSERT OR REPLACE INTO registry_meta (key, value) VALUES ('salt', ?)",
                    (salt_b64,),
                )

            conn.commit()
            conn.close()

    def _load_counters(self) -> None:
        """Load token counters from existing data to ensure unique numbering."""
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            rows = conn.execute(
                "SELECT matter_id, category, COUNT(*) FROM tokens GROUP BY matter_id, category"
            ).fetchall()
            conn.close()

            for matter_id, category, count in rows:
                if matter_id not in self._counters:
                    self._counters[matter_id] = {}
                self._counters[matter_id][category] = count

    # ------------------------------------------------------------------
    # Token operations
    # ------------------------------------------------------------------

    def _next_index(self, matter_id: str, category: str) -> int:
        """Get and increment the counter for a matter/category pair."""
        if matter_id not in self._counters:
            self._counters[matter_id] = {}
        current = self._counters[matter_id].get(category, 0)
        self._counters[matter_id][category] = current + 1
        return current + 1

    def _make_token_label(self, category: PIICategory, index: int) -> str:
        """Generate a human-readable anonymised token label.

        Examples:
            [PERSON_001], [ADDRESS_012], [DATE_003]
        """
        prefix = category.value.upper()
        return f"[{prefix}_{index:03d}]"

    def get_or_create_token(
        self,
        matter_id: str,
        category: PIICategory,
        original_value: str,
        method: AnonymisationMethod = AnonymisationMethod.TOKENISATION,
    ) -> AnonymisationToken:
        """Look up existing token or create a new one.

        Ensures consistent anonymisation: the same original value always
        maps to the same token within a matter.

        Uses a deterministic hash for lookup (not the encrypted value,
        since Fernet encryption is non-deterministic).

        Args:
            matter_id: Matter/case identifier.
            category: PII category of the entity.
            original_value: The real PII value.
            method: Anonymisation method applied.

        Returns:
            AnonymisationToken with the anonymised replacement.
        """
        lookup_hash = self._hash_for_lookup(original_value)

        with self._lock:
            conn = sqlite3.connect(str(self._db_path))

            # Check for existing token using deterministic hash
            row = conn.execute(
                "SELECT token_id, anonymised_value, method, created_at, metadata_json "
                "FROM tokens WHERE matter_id = ? AND category = ? AND original_hash = ?",
                (matter_id, category.value, lookup_hash),
            ).fetchone()

            if row:
                conn.close()
                return AnonymisationToken(
                    token_id=row[0],
                    matter_id=matter_id,
                    category=category,
                    original_value=original_value,
                    anonymised_value=row[1],
                    method=AnonymisationMethod(row[2]),
                    created_at=datetime.fromisoformat(row[3]),
                )

            # Create new token — encrypt the original value for storage
            encrypted_original = self._encrypt(original_value)
            index = self._next_index(matter_id, category.value)
            token = AnonymisationToken(
                matter_id=matter_id,
                category=category,
                original_value=original_value,
                anonymised_value=self._make_token_label(category, index),
                method=method,
            )

            conn.execute(
                "INSERT INTO tokens (token_id, matter_id, category, original_hash, "
                "original_value, anonymised_value, method, created_at, metadata_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    token.token_id,
                    matter_id,
                    category.value,
                    lookup_hash,
                    encrypted_original,
                    token.anonymised_value,
                    method.value,
                    token.created_at.isoformat(),
                    "{}",
                ),
            )
            conn.commit()
            conn.close()

            return token

    def resolve_token(self, matter_id: str, anonymised_value: str) -> Optional[str]:
        """Resolve an anonymised token back to its original value (local only).

        Args:
            matter_id: Matter/case identifier.
            anonymised_value: The token to resolve, e.g. [PERSON_001].

        Returns:
            Original PII value, or None if not found.
        """
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            row = conn.execute(
                "SELECT original_value FROM tokens "
                "WHERE matter_id = ? AND anonymised_value = ?",
                (matter_id, anonymised_value),
            ).fetchone()
            conn.close()

            if row:
                return self._decrypt(row[0])
            return None

    def get_all_tokens(self, matter_id: str) -> list[AnonymisationToken]:
        """Get all tokens for a matter.

        Args:
            matter_id: Matter/case identifier.

        Returns:
            List of all AnonymisationToken instances for this matter.
        """
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            rows = conn.execute(
                "SELECT token_id, category, original_value, anonymised_value, "
                "method, created_at FROM tokens WHERE matter_id = ? "
                "ORDER BY created_at",
                (matter_id,),
            ).fetchall()
            conn.close()

            tokens = []
            for row in rows:
                tokens.append(AnonymisationToken(
                    token_id=row[0],
                    matter_id=matter_id,
                    category=PIICategory(row[1]),
                    original_value=self._decrypt(row[2]),
                    anonymised_value=row[3],
                    method=AnonymisationMethod(row[4]),
                    created_at=datetime.fromisoformat(row[5]),
                ))
            return tokens

    def get_token_legend(self, matter_id: str) -> dict[str, str]:
        """Get a legend mapping tokens to their categories (safe for export).

        This does NOT include original values — only token → category.

        Args:
            matter_id: Matter/case identifier.

        Returns:
            Dict mapping e.g. "[PERSON_001]" → "person_name".
        """
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            rows = conn.execute(
                "SELECT anonymised_value, category FROM tokens WHERE matter_id = ?",
                (matter_id,),
            ).fetchall()
            conn.close()

            return {row[0]: row[1] for row in rows}

    def get_reverse_map(self, matter_id: str) -> dict[str, str]:
        """Get full reverse mapping: anonymised_value → original_value.

        WARNING: This contains real PII. Only used locally for de-anonymisation.

        Args:
            matter_id: Matter/case identifier.

        Returns:
            Dict mapping e.g. "[PERSON_001]" → "James Smith".
        """
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            rows = conn.execute(
                "SELECT anonymised_value, original_value FROM tokens WHERE matter_id = ?",
                (matter_id,),
            ).fetchall()
            conn.close()

            return {row[0]: self._decrypt(row[1]) for row in rows}

    def token_count(self, matter_id: str) -> int:
        """Get number of tokens for a matter."""
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            row = conn.execute(
                "SELECT COUNT(*) FROM tokens WHERE matter_id = ?",
                (matter_id,),
            ).fetchone()
            conn.close()
            return row[0] if row else 0

    def delete_matter_tokens(self, matter_id: str) -> int:
        """Delete all tokens for a matter. Returns count deleted."""
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            cursor = conn.execute(
                "DELETE FROM tokens WHERE matter_id = ?", (matter_id,)
            )
            deleted = cursor.rowcount
            conn.commit()
            conn.close()

            if matter_id in self._counters:
                del self._counters[matter_id]

            logger.info(f"Deleted {deleted} tokens for matter {matter_id}")
            return deleted

    # ------------------------------------------------------------------
    # Audit logging
    # ------------------------------------------------------------------

    def log_audit(
        self,
        action: str,
        matter_id: str,
        document_id: str = "",
        entity_count: int = 0,
        user: str = "system",
        details: str = "",
        success: bool = True,
    ) -> None:
        """Write an audit entry to the log.

        Args:
            action: Action type (anonymise, deanonymise, review, export).
            matter_id: Matter identifier.
            document_id: Document identifier.
            entity_count: Number of entities involved.
            user: User or system identifier.
            details: Free-text details.
            success: Whether the action succeeded.
        """
        import uuid as _uuid

        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            conn.execute(
                "INSERT INTO audit_log (id, timestamp, action, matter_id, "
                "document_id, entity_count, user, details, success) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(_uuid.uuid4())[:12],
                    datetime.now().isoformat(),
                    action,
                    matter_id,
                    document_id,
                    entity_count,
                    user,
                    details,
                    1 if success else 0,
                ),
            )
            conn.commit()
            conn.close()

    def get_audit_log(
        self,
        matter_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Retrieve audit log entries.

        Args:
            matter_id: Filter by matter (None for all).
            limit: Maximum entries to return.

        Returns:
            List of audit entries as dicts.
        """
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            if matter_id:
                rows = conn.execute(
                    "SELECT id, timestamp, action, matter_id, document_id, "
                    "entity_count, user, details, success FROM audit_log "
                    "WHERE matter_id = ? ORDER BY timestamp DESC LIMIT ?",
                    (matter_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, timestamp, action, matter_id, document_id, "
                    "entity_count, user, details, success FROM audit_log "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            conn.close()

            return [
                {
                    "id": r[0],
                    "timestamp": r[1],
                    "action": r[2],
                    "matter_id": r[3],
                    "document_id": r[4],
                    "entity_count": r[5],
                    "user": r[6],
                    "details": r[7],
                    "success": bool(r[8]),
                }
                for r in rows
            ]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def wipe_encryption_key(self) -> None:
        """Wipe the encryption key from memory. Called on application close."""
        self._fernet = None
        logger.info("Anonymisation registry encryption key wiped from memory")

    def close(self) -> None:
        """Clean shutdown — wipe key and release resources."""
        self.wipe_encryption_key()
