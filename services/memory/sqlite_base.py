from __future__ import annotations

from services.memory.database import Database
from services.memory.encryption import SensitiveMemoryEncryption


class SqliteRepositoryBase:
    """Shared typed state for the repositories behind ``SqliteMemory``."""

    database: Database
    sensitive_encryption: SensitiveMemoryEncryption | None
    sensitive_encryption_enabled: bool
