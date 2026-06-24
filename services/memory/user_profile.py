from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from april_common.time import utc_now_iso
from services.memory.database import Database

# This is a single-user, local-first assistant; the profile is a singleton row.
LOCAL_USER_ID = "local-user"


class UserProfile(BaseModel):
    id: str = LOCAL_USER_ID
    display_name: str
    # Preferred form of address (e.g. "Sam", "Dr. Lee"). Never inferred.
    preferred_address: str | None = None
    timezone: str | None = None
    preferences: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class UserProfileStore:
    """Local-only CRUD for the user profile.

    Only the explicitly-provided fields are stored; nothing is inferred. The
    profile is never added to model requests by this store — callers must opt in
    through the memory policy if they want to surface any of it to a model.
    """

    def __init__(self, database: Database) -> None:
        self.database = database

    async def get(self, user_id: str = LOCAL_USER_ID) -> UserProfile | None:
        row = await self.database.fetchone(
            "SELECT id, name, address, timezone, preferences_json, created_at, updated_at "
            "FROM users WHERE id = ?",
            (user_id,),
        )
        if row is None:
            return None
        return self._row_to_profile(row)

    async def set(
        self,
        *,
        display_name: str,
        preferred_address: str | None = None,
        timezone: str | None = None,
        preferences: dict[str, Any] | None = None,
        user_id: str = LOCAL_USER_ID,
    ) -> UserProfile:
        now = utc_now_iso()
        preferences_json = json.dumps(preferences or {}, sort_keys=True)
        existing = await self.get(user_id)
        if existing is None:
            await self.database.execute(
                "INSERT INTO users(id, name, address, timezone, preferences_json, "
                "created_at, updated_at) VALUES(?, ?, ?, ?, ?, ?, ?)",
                (user_id, display_name, preferred_address, timezone, preferences_json, now, now),
            )
        else:
            await self.database.execute(
                "UPDATE users SET name = ?, address = ?, timezone = ?, preferences_json = ?, "
                "updated_at = ? WHERE id = ?",
                (display_name, preferred_address, timezone, preferences_json, now, user_id),
            )
        profile = await self.get(user_id)
        assert profile is not None
        return profile

    async def delete(self, user_id: str = LOCAL_USER_ID) -> bool:
        cursor = await self.database.execute("DELETE FROM users WHERE id = ?", (user_id,))
        return bool(cursor.rowcount)

    def _row_to_profile(self, row: Any) -> UserProfile:
        preferences: dict[str, Any] = {}
        raw = row["preferences_json"]
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    preferences = parsed
            except json.JSONDecodeError:
                preferences = {}
        return UserProfile(
            id=row["id"],
            display_name=row["name"],
            preferred_address=row["address"],
            timezone=row["timezone"],
            preferences=preferences,
            created_at=row["created_at"],
            updated_at=row["updated_at"] or row["created_at"],
        )
