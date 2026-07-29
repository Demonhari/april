from __future__ import annotations

import base64
import json
import secrets
from dataclasses import dataclass
from typing import Protocol

from april_common.credentials import (
    CredentialKey,
    CredentialStore,
    CredentialStoreError,
    select_credential_store,
)
from april_common.settings import AprilSettings
from services.memory.database import Database

ENVELOPE_PREFIX = "april:enc:v1:"
KEYRING_FORMAT_VERSION = 1
UNAVAILABLE_CONTENT = "[encrypted memory unavailable]"


class MemoryEncryptionError(RuntimeError):
    pass


class AuthenticatedEncryption(Protocol):
    def encrypt(self, key: bytes, nonce: bytes, plaintext: bytes, aad: bytes) -> bytes: ...

    def decrypt(self, key: bytes, nonce: bytes, ciphertext: bytes, aad: bytes) -> bytes: ...


class AESGCMEncryption:
    def encrypt(self, key: bytes, nonce: bytes, plaintext: bytes, aad: bytes) -> bytes:
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError as exc:
            raise MemoryEncryptionError(
                "Sensitive-memory encryption requires APRIL's security extra."
            ) from exc
        return AESGCM(key).encrypt(nonce, plaintext, aad)

    def decrypt(self, key: bytes, nonce: bytes, ciphertext: bytes, aad: bytes) -> bytes:
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError as exc:
            raise MemoryEncryptionError(
                "Sensitive-memory encryption requires APRIL's security extra."
            ) from exc
        try:
            return AESGCM(key).decrypt(nonce, ciphertext, aad)
        except Exception as exc:
            raise MemoryEncryptionError("Encrypted memory authentication failed.") from exc


@dataclass(frozen=True, slots=True)
class MemoryKeyring:
    active_key_id: str
    keys: dict[str, bytes]

    def encode(self) -> str:
        return json.dumps(
            {
                "format_version": KEYRING_FORMAT_VERSION,
                "active_key_id": self.active_key_id,
                "keys": {
                    key_id: base64.urlsafe_b64encode(value).decode("ascii")
                    for key_id, value in sorted(self.keys.items())
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def decode(cls, value: str) -> MemoryKeyring:
        try:
            payload = json.loads(value)
            if payload.get("format_version") != KEYRING_FORMAT_VERSION:
                raise ValueError
            active = str(payload["active_key_id"])
            encoded = payload["keys"]
            keys = {
                str(key_id): base64.urlsafe_b64decode(str(key_value).encode("ascii"))
                for key_id, key_value in encoded.items()
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise MemoryEncryptionError("Memory encryption keyring is malformed.") from exc
        if active not in keys or any(len(key) != 32 for key in keys.values()):
            raise MemoryEncryptionError("Memory encryption keyring is invalid.")
        return cls(active_key_id=active, keys=keys)


class SensitiveMemoryEncryption:
    def __init__(
        self,
        keyring: MemoryKeyring,
        *,
        encryption: AuthenticatedEncryption | None = None,
    ) -> None:
        self.keyring = keyring
        self.encryption = encryption or AESGCMEncryption()

    def encrypt(self, memory_id: str, plaintext: str) -> str:
        key_id = self.keyring.active_key_id
        nonce = secrets.token_bytes(12)
        ciphertext = self.encryption.encrypt(
            self.keyring.keys[key_id],
            nonce,
            plaintext.encode("utf-8"),
            memory_id.encode("utf-8"),
        )
        envelope = {
            "kid": key_id,
            "nonce": base64.urlsafe_b64encode(nonce).decode("ascii"),
            "ciphertext": base64.urlsafe_b64encode(ciphertext).decode("ascii"),
        }
        encoded = base64.urlsafe_b64encode(
            json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).decode("ascii")
        return ENVELOPE_PREFIX + encoded

    def decrypt(self, memory_id: str, envelope: str) -> str:
        if not envelope.startswith(ENVELOPE_PREFIX):
            return envelope
        try:
            payload = json.loads(
                base64.urlsafe_b64decode(envelope.removeprefix(ENVELOPE_PREFIX)).decode("utf-8")
            )
            key_id = str(payload["kid"])
            nonce = base64.urlsafe_b64decode(str(payload["nonce"]))
            ciphertext = base64.urlsafe_b64decode(str(payload["ciphertext"]))
            key = self.keyring.keys[key_id]
        except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MemoryEncryptionError("Encrypted memory envelope is malformed.") from exc
        plaintext = self.encryption.decrypt(
            key,
            nonce,
            ciphertext,
            memory_id.encode("utf-8"),
        )
        try:
            return plaintext.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MemoryEncryptionError("Encrypted memory plaintext is invalid.") from exc


def is_encrypted_memory(value: str) -> bool:
    return value.startswith(ENVELOPE_PREFIX)


def provision_memory_key(store: CredentialStore) -> str:
    if store.exists(CredentialKey.MEMORY_ENCRYPTION_KEY):
        keyring = _load_keyring(store)
        return keyring.active_key_id
    key_id = secrets.token_hex(8)
    keyring = MemoryKeyring(active_key_id=key_id, keys={key_id: secrets.token_bytes(32)})
    store.set(CredentialKey.MEMORY_ENCRYPTION_KEY, keyring.encode())
    if _load_keyring(store).active_key_id != key_id:
        raise CredentialStoreError("Memory encryption key read-back verification failed.")
    return key_id


def sensitive_encryption_for_settings(
    settings: AprilSettings,
    *,
    store: CredentialStore | None = None,
    encryption: AuthenticatedEncryption | None = None,
) -> SensitiveMemoryEncryption | None:
    if not settings.memory.sensitive_encryption_enabled:
        return None
    active_store = store or _store_for_settings(settings)
    return SensitiveMemoryEncryption(_load_keyring(active_store), encryption=encryption)


async def rotate_memory_key(
    settings: AprilSettings,
    database: Database,
    *,
    store: CredentialStore | None = None,
    encryption: AuthenticatedEncryption | None = None,
) -> dict[str, int | str]:
    """Crash-safe key rotation using a staged keyring and one DB transaction."""
    active_store = store or _store_for_settings(settings)
    old_ring = _load_keyring(active_store)
    new_id = secrets.token_hex(8)
    staged = MemoryKeyring(
        active_key_id=old_ring.active_key_id,
        keys={**old_ring.keys, new_id: secrets.token_bytes(32)},
    )
    active_store.set(CredentialKey.MEMORY_ENCRYPTION_KEY, staged.encode())
    old_cipher = SensitiveMemoryEncryption(staged, encryption=encryption)
    new_ring = MemoryKeyring(active_key_id=new_id, keys=staged.keys)
    new_cipher = SensitiveMemoryEncryption(new_ring, encryption=encryption)
    rotated = 0
    try:
        async with database.transaction() as connection:
            cursor = await connection.execute(
                "SELECT id, content FROM memories WHERE content LIKE ? ORDER BY id",
                (f"{ENVELOPE_PREFIX}%",),
            )
            rows = await cursor.fetchall()
            for row in rows:
                memory_id = str(row["id"])
                plaintext = old_cipher.decrypt(memory_id, str(row["content"]))
                replacement = new_cipher.encrypt(memory_id, plaintext)
                await connection.execute(
                    "UPDATE memories SET content = ? WHERE id = ?",
                    (replacement, memory_id),
                )
                rotated += 1
        active_store.set(CredentialKey.MEMORY_ENCRYPTION_KEY, new_ring.encode())
        confirmed = _load_keyring(active_store)
        if confirmed.active_key_id != new_id:
            raise MemoryEncryptionError("Memory key rotation read-back verification failed.")
    except BaseException:
        # If the DB transaction failed it rolled back, so the previous active key
        # remains valid. Keeping the staged key also makes the DB-committed crash
        # window recoverable if final keyring activation was interrupted.
        active_store.set(CredentialKey.MEMORY_ENCRYPTION_KEY, staged.encode())
        raise
    return {"rotated_records": rotated, "active_key_id": new_id}


def _load_keyring(store: CredentialStore) -> MemoryKeyring:
    value = store.get(CredentialKey.MEMORY_ENCRYPTION_KEY)
    if value is None:
        raise MemoryEncryptionError("Memory encryption key is unavailable.")
    return MemoryKeyring.decode(value)


def _store_for_settings(settings: AprilSettings) -> CredentialStore:
    file_path = settings.security.credential_file_path
    resolved_file = settings.resolve_path(file_path) if file_path is not None else None
    return select_credential_store(
        backend=settings.security.credential_store,
        environment=settings.environment,
        repository_root=settings.home,
        file_path=resolved_file,
    )
