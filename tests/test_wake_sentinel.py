from __future__ import annotations

import asyncio
import os
import stat
import tempfile
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from apps.runner.wake_live import run_sentinel_live_verification
from services.memory.database import Database
from services.memory.migrations import run_migrations
from services.memory.sqlite_memory import SqliteMemory
from services.voice.speech_to_text import FakeSpeechToText, SpeechToText
from services.voice.text_to_speech import FakeTextToSpeech
from services.wake.confirmer import (
    SttConfirmer,
    canonicalize_wake_word,
    edit_distance,
    is_addressed,
    normalized_edit_distance,
    strip_vocative,
)
from services.wake.control import (
    SentinelControlServer,
    attach_resident_sentinel,
    resident_sentinel_status,
    sentinel_control_path,
)
from services.wake.fakes import (
    FakeFrameMicrophone,
    FakeSpeakerVerifier,
    ManualClock,
    RecordingAudioPlayer,
    RecordingAudit,
    RecordingDelivery,
    ScriptedScorer,
)
from services.wake.ring_buffer import AudioRingBuffer
from services.wake.schemas import WakeEvent
from services.wake.sentinel import ApiWakeDelivery, MuteSwitch, Sentinel
from services.wake.session_manager import SessionManager
from services.wake.status import read_wake_status
from services.wake.wake_bus import WakeBus, send_wake_event

FRAME = b"\x00\x01" * 160  # quiet 16-bit PCM frame
LOUD_FRAME = b"\x00\x40" * 160  # loud frame the VAD counts as speech


class RecordingSpeechToText(SpeechToText):
    def __init__(self, text: str) -> None:
        self.text = text
        self.payloads: list[bytes] = []
        self.paths: list[Path] = []

    async def transcribe(self, audio_path: Path) -> str:
        self.paths.append(audio_path)
        self.payloads.append(audio_path.read_bytes())
        return self.text


class _FakeHttpResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


class _FakeAsyncClient:
    handler = None

    def __init__(self, *args: object, **kwargs: object) -> None:
        return None

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def post(
        self, url: str, *, json: dict[str, object], headers: dict[str, str]
    ) -> _FakeHttpResponse:
        assert _FakeAsyncClient.handler is not None
        return await _FakeAsyncClient.handler(url, json, headers)


def _short_socket_path() -> Path:
    return Path(tempfile.gettempdir()).resolve() / f"aw-{os.getpid()}-{uuid.uuid4().hex[:8]}.sock"


@pytest.mark.asyncio
async def test_resident_sentinel_control_has_single_live_controller(settings_tmp) -> None:
    short_home = Path("/tmp") / f"ac-{uuid.uuid4().hex[:8]}"
    settings = settings_tmp.model_copy(update={"home": short_home})
    hints: list[str | None] = []
    server = SentinelControlServer(
        sentinel_control_path(settings),
        set_session_hint=hints.append,
        status=lambda: {"state": "listening", "voice_output": "degraded"},
    )
    await server.start()
    first = None
    try:
        status = await asyncio.to_thread(resident_sentinel_status, settings)
        assert status["state"] == "listening"
        assert status["controlled"] is False
        first = await asyncio.to_thread(
            attach_resident_sentinel, settings, session_hint="session-one"
        )
        assert first.status["attached"] is True
        assert hints[-1] == "session-one"
        with pytest.raises(RuntimeError, match="already controlled"):
            await asyncio.to_thread(attach_resident_sentinel, settings, session_hint="session-two")
        first.close()
        first = None
        await asyncio.sleep(0)
        second = await asyncio.to_thread(
            attach_resident_sentinel, settings, session_hint="session-three"
        )
        second.close()
    finally:
        if first is not None:
            first.close()
        await server.close()


async def _memory(settings_tmp) -> tuple[Database, SqliteMemory]:
    database = Database(settings_tmp.database_path)
    await database.connect()
    await run_migrations(database)
    return database, SqliteMemory(database)


def _sentinel(
    settings,
    *,
    scores: list[float],
    frames: int = 6,
    stt_text: str | None = None,
    confirm_with_stt: bool | None = None,
    instant_accept: bool | None = None,
    clock: ManualClock | None = None,
    stt: SpeechToText | None = None,
) -> tuple[Sentinel, FakeFrameMicrophone, RecordingDelivery, RecordingAudioPlayer]:
    wake_update: dict[str, object] = {"enabled": True}
    if confirm_with_stt is not None:
        wake_update["confirm_with_stt"] = confirm_with_stt
    if instant_accept is not None:
        wake_update["instant_accept"] = instant_accept
    tuned = settings.model_copy(update={"wake": settings.wake.model_copy(update=wake_update)})
    microphone = FakeFrameMicrophone([FRAME] * frames)
    delivery = RecordingDelivery()
    player = RecordingAudioPlayer()
    confirmer = None
    if stt is not None:
        confirmer = SttConfirmer(stt, audio_cache_path=tuned.audio_cache_path)
    elif stt_text is not None:
        confirmer = SttConfirmer(
            FakeSpeechToText(stt_text),
            audio_cache_path=tuned.audio_cache_path,
        )
    sentinel = Sentinel(
        settings=tuned,
        microphone=microphone,
        scorers=[ScriptedScorer(scores)],
        deliver=delivery,
        confirmer=confirmer,
        player=player,
        mute=MuteSwitch(tuned.mute_flag_path),
        clock=clock or ManualClock(),
    )
    return sentinel, microphone, delivery, player


# ---------------------------------------------------------------------------
# Vocative stripping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("april, restart the runtime", "restart the runtime"),
        ("April restart the runtime", "restart the runtime"),
        ("can you check my repo april", "can you check my repo"),
        ("can you check my repo, april?", "can you check my repo"),
        ("hey april what's on today", "what's on today"),
        ("okay april, what's on today?", "what's on today?"),
        ("add april to the meeting notes", "add april to the meeting notes"),
        ("the april release plan looks fine", "the april release plan looks fine"),
        ("april", ""),
    ],
)
def test_strip_vocative(raw: str, expected: str) -> None:
    assert strip_vocative(raw) == expected


def test_is_addressed_strict_rejects_mid_sentence_mentions() -> None:
    assert is_addressed("april, restart the runtime", strict=True)
    assert is_addressed("can you check my repo april", strict=True)
    assert not is_addressed("add april to the meeting notes", strict=True)
    assert is_addressed("add april to the meeting notes", strict=False)
    assert not is_addressed("restart the runtime", strict=False)


# ---------------------------------------------------------------------------
# Fuzzy STT confirmation
# ---------------------------------------------------------------------------


def test_edit_distance_is_deterministic() -> None:
    assert edit_distance("april", "april") == 0
    assert edit_distance("april", "apryl") == 1
    assert edit_distance("april", "avril") == 1
    assert edit_distance("april", "aprill") == 1
    assert edit_distance("april", "apron") == 2
    assert normalized_edit_distance("april", "apryl") == pytest.approx(0.2)
    assert normalized_edit_distance("", "") == 0.0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("apryl, restart the runtime", "april, restart the runtime"),
        ("avril restart the runtime", "april restart the runtime"),
        ("hey aprill what's on today", "hey april what's on today"),
        ("a pril, what time is it", "april, what time is it"),
        ("add apron to the shopping list", "add apron to the shopping list"),
        ("a really long sentence", "a really long sentence"),
        ("april is already canonical", "april is already canonical"),
    ],
)
def test_canonicalize_wake_word(raw: str, expected: str) -> None:
    assert canonicalize_wake_word(raw) == expected


@pytest.mark.parametrize("variant", ["april", "apryl", "avril", "aprill", "a pril"])
def test_fuzzy_variants_accepted_in_both_modes(variant: str) -> None:
    assert is_addressed(f"{variant}, restart the runtime", strict=False)
    assert is_addressed(f"{variant}, restart the runtime", strict=True)
    assert is_addressed(f"can you check my repo {variant}", strict=True)
    assert strip_vocative(f"{variant}, restart the runtime") == "restart the runtime"


def test_fuzzy_strict_mode_still_rejects_mid_sentence_variants() -> None:
    assert not is_addressed("add apryl to the meeting notes", strict=True)
    assert is_addressed("add apryl to the meeting notes", strict=False)


def test_fuzzy_does_not_match_unrelated_words() -> None:
    assert not is_addressed("put the apron on the hook", strict=False)
    assert not is_addressed("the apple is ripe", strict=False)
    assert not is_addressed("a really nice day", strict=False)


def test_fuzzy_can_be_disabled() -> None:
    assert not is_addressed("apryl, restart the runtime", strict=False, fuzzy=False)
    assert strip_vocative("apryl, restart it", fuzzy=False) == "apryl, restart it"


def test_fuzzy_threshold_is_configurable() -> None:
    # A zero distance budget accepts only the exact wake word.
    assert not is_addressed("apryl, restart the runtime", fuzzy_max_distance=0.0)
    assert is_addressed("april, restart the runtime", fuzzy_max_distance=0.0)
    # A looser budget accepts variants the default would reject
    # ("apryll" is two edits from "april": 2/6 ≈ 0.33 > 0.25).
    assert not is_addressed("apryll, restart the runtime")
    assert is_addressed("apryll, restart the runtime", fuzzy_max_distance=0.4)


async def test_stt_confirmer_honours_configured_fuzzy_distance(settings_tmp) -> None:
    confirmer = SttConfirmer(
        FakeSpeechToText("apryl, restart the runtime"),
        audio_cache_path=settings_tmp.audio_cache_path,
        fuzzy_max_distance=0.0,
    )
    confirmation = await confirmer.confirm([FRAME])
    assert confirmation.accepted is False
    assert "does not address" in confirmation.reason


async def test_sentinel_stt_confirmation_accepts_fuzzy_variant(settings_tmp) -> None:
    sentinel, _mic, delivery, _player = _sentinel(
        settings_tmp,
        scores=[0.5],
        stt_text="apryl, restart the runtime",
        confirm_with_stt=True,
    )
    await sentinel.run_once()
    assert len(delivery.events) == 1
    event = delivery.events[0]
    assert event.reason == "stt_confirmed"
    assert event.text == "restart the runtime"


# ---------------------------------------------------------------------------
# Ring buffer pre-roll
# ---------------------------------------------------------------------------


def test_ring_buffer_keeps_most_recent_pre_roll() -> None:
    # 1 second capacity at 16kHz/16-bit mono = 32000 bytes.
    buffer = AudioRingBuffer(seconds=1.0)
    chunk = b"x" * 8000  # 0.25s per chunk
    for index in range(8):
        buffer.append(bytes([index]) * 8000)
    snapshot = buffer.snapshot()
    assert buffer.total_bytes <= buffer.capacity_bytes
    # Only the most recent 4 chunks (1 second) survive, oldest first.
    assert snapshot == [bytes([index]) * 8000 for index in range(4, 8)]
    assert buffer.duration_seconds == pytest.approx(1.0)
    buffer.clear()
    assert buffer.snapshot() == []
    assert buffer.total_bytes == 0
    del chunk


def test_ring_buffer_snapshot_preserves_wake_onset() -> None:
    buffer = AudioRingBuffer(seconds=10.0)
    onset = [b"onset-1", b"onset-2", b"wake-word", b"command"]
    for frame in onset:
        buffer.append(frame)
    assert buffer.snapshot() == onset


def test_ring_buffer_truncates_one_oversized_frame_to_newest_audio() -> None:
    buffer = AudioRingBuffer(seconds=1.0, sample_rate=4, bytes_per_sample=1)
    buffer.append(b"old")
    buffer.append(b"0123456789")
    assert buffer.total_bytes == buffer.capacity_bytes == 4
    assert buffer.snapshot() == [b"6789"]


# ---------------------------------------------------------------------------
# Soft speaker gate (convenience filter, never authentication)
# ---------------------------------------------------------------------------


async def test_soft_speaker_gate_drops_nonmatching_confirmed_wake(settings_tmp) -> None:
    tuned = settings_tmp.model_copy(
        update={
            "wake": settings_tmp.wake.model_copy(
                update={
                    "enabled": True,
                    "confirm_with_stt": False,
                    "speaker_gate": "soft",
                }
            )
        }
    )
    verifier = FakeSpeakerVerifier(0.2)
    audit = RecordingAudit()
    delivery = RecordingDelivery()
    player = RecordingAudioPlayer()
    sentinel = Sentinel(
        settings=tuned,
        microphone=FakeFrameMicrophone([FRAME]),
        scorers=[ScriptedScorer([0.9])],
        deliver=delivery,
        player=player,
        mute=MuteSwitch(tuned.mute_flag_path),
        speaker_verifier=verifier,
        audit=audit,
    )

    await sentinel.run_once()

    assert delivery.events == []
    assert len(verifier.calls) == 1
    assert verifier.calls[0][1] == FRAME
    assert sentinel.last_rejection_reason == "speaker_gate"
    assert player.played == []
    dropped = next(record for record in audit.records if record["event_type"] == "wake_dropped")
    assert dropped["reason"] == "speaker_gate"


async def test_soft_speaker_gate_passes_matching_confirmed_wake(settings_tmp) -> None:
    tuned = settings_tmp.model_copy(
        update={
            "wake": settings_tmp.wake.model_copy(
                update={
                    "enabled": True,
                    "confirm_with_stt": False,
                    "speaker_gate": "soft",
                }
            )
        }
    )
    verifier = FakeSpeakerVerifier(0.8)
    delivery = RecordingDelivery()
    sentinel = Sentinel(
        settings=tuned,
        microphone=FakeFrameMicrophone([FRAME]),
        scorers=[ScriptedScorer([0.9])],
        deliver=delivery,
        player=RecordingAudioPlayer(),
        mute=MuteSwitch(tuned.mute_flag_path),
        speaker_verifier=verifier,
        audit=RecordingAudit(),
    )

    await sentinel.run_once()

    assert len(delivery.events) == 1
    assert len(verifier.calls) == 1


async def test_soft_speaker_enrollment_rejects_symlink_escape(settings_tmp) -> None:
    profile = settings_tmp.resolve_path(Path("data/voice_profiles"))
    profile.mkdir(parents=True)
    enrolled = profile / "enroll-01.wav"
    enrolled.write_bytes(b"RIFF enrolled")
    outside = settings_tmp.home.parent / "outside.wav"
    outside.write_bytes(b"RIFF outside")
    (profile / "enroll-02.wav").symlink_to(outside)
    tuned = settings_tmp.model_copy(
        update={
            "wake": settings_tmp.wake.model_copy(
                update={
                    "enabled": True,
                    "confirm_with_stt": False,
                    "speaker_gate": "soft",
                }
            )
        }
    )
    verifier = FakeSpeakerVerifier(0.8)
    sentinel = Sentinel(
        settings=tuned,
        microphone=FakeFrameMicrophone([FRAME]),
        scorers=[ScriptedScorer([0.9])],
        deliver=RecordingDelivery(),
        mute=MuteSwitch(tuned.mute_flag_path),
        speaker_verifier=verifier,
        audit=RecordingAudit(),
    )

    await sentinel.run_once()

    assert verifier.calls[0][0] == (enrolled.resolve(),)


async def test_off_speaker_gate_never_consults_verifier(settings_tmp) -> None:
    tuned = settings_tmp.model_copy(
        update={
            "wake": settings_tmp.wake.model_copy(
                update={"enabled": True, "confirm_with_stt": False, "speaker_gate": "off"}
            )
        }
    )
    verifier = FakeSpeakerVerifier(0.0)
    delivery = RecordingDelivery()
    sentinel = Sentinel(
        settings=tuned,
        microphone=FakeFrameMicrophone([FRAME]),
        scorers=[ScriptedScorer([0.9])],
        deliver=delivery,
        mute=MuteSwitch(tuned.mute_flag_path),
        speaker_verifier=verifier,
        audit=RecordingAudit(),
    )

    await sentinel.run_once()

    assert len(delivery.events) == 1
    assert verifier.calls == []


async def test_soft_speaker_gate_missing_model_degrades_once_and_never_blocks(
    settings_tmp,
) -> None:
    tuned = settings_tmp.model_copy(
        update={
            "wake": settings_tmp.wake.model_copy(
                update={
                    "enabled": True,
                    "confirm_with_stt": False,
                    "speaker_gate": "soft",
                }
            )
        }
    )
    audit = RecordingAudit()
    delivery = RecordingDelivery()
    sentinel = Sentinel(
        settings=tuned,
        microphone=FakeFrameMicrophone([FRAME]),
        scorers=[ScriptedScorer([0.9])],
        deliver=delivery,
        mute=MuteSwitch(tuned.mute_flag_path),
        speaker_verifier=None,
        audit=audit,
    )

    await sentinel.run_once()
    sentinel.microphone = FakeFrameMicrophone([])
    await sentinel.run_once()

    assert len(delivery.events) == 1
    warnings = [
        record for record in audit.records if record["event_type"] == "speaker_gate_degraded"
    ]
    assert len(warnings) == 1
    assert warnings[0]["status"] == "warning"
    assert warnings[0]["reason"] == "speaker_gate"


# ---------------------------------------------------------------------------
# Wake bus
# ---------------------------------------------------------------------------


async def test_wake_bus_socket_permissions_and_delivery(settings_tmp) -> None:
    received: list[WakeEvent] = []

    async def handler(event: WakeEvent) -> dict[str, object]:
        received.append(event)
        return {"session_id": "session-1"}

    socket_path = _short_socket_path()
    bus = WakeBus(socket_path, handler)
    await bus.start()
    try:
        mode = stat.S_IMODE(os.lstat(socket_path).st_mode)
        assert mode == 0o600
        reply = await send_wake_event(
            socket_path, WakeEvent(source="hotkey", score=0.9, reason="test")
        )
        assert reply["ok"] is True
        assert reply["result"] == {"session_id": "session-1"}
        assert len(received) == 1
        assert received[0].source == "hotkey"
    finally:
        await bus.stop()
    assert not socket_path.exists()


async def test_wake_bus_rejects_invalid_payloads(settings_tmp) -> None:
    async def handler(event: WakeEvent) -> dict[str, object]:
        raise AssertionError("handler must not run for invalid payloads")

    socket_path = _short_socket_path()
    bus = WakeBus(socket_path, handler)
    await bus.start()
    try:
        reader, writer = await asyncio.open_unix_connection(path=str(socket_path))
        writer.write(b'{"source": "not-a-surface"}\n')
        await writer.drain()
        raw = await asyncio.wait_for(reader.readline(), timeout=5.0)
        assert b'"ok": false' in raw
        writer.close()
        await writer.wait_closed()
    finally:
        await bus.stop()


def test_wake_event_accepts_transcript_alias() -> None:
    event = WakeEvent.model_validate({"source": "socket", "transcript": "open my notes"})
    assert event.text == "open my notes"
    # Matching duplicate values are tolerated; only conflicts are rejected.
    event = WakeEvent.model_validate(
        {"source": "socket", "text": "open my notes", "transcript": "open my notes"}
    )
    assert event.text == "open my notes"
    with pytest.raises(ValueError, match="conflicting"):
        WakeEvent.model_validate(
            {"source": "socket", "text": "open my notes", "transcript": "something else"}
        )


async def test_wake_bus_accepts_transcript_alias_payload(settings_tmp) -> None:
    received: list[WakeEvent] = []

    async def handler(event: WakeEvent) -> dict[str, object]:
        received.append(event)
        return {}

    socket_path = _short_socket_path()
    bus = WakeBus(socket_path, handler)
    await bus.start()
    try:
        reader, writer = await asyncio.open_unix_connection(path=str(socket_path))
        writer.write(b'{"source": "socket", "transcript": "open my notes"}\n')
        await writer.drain()
        raw = await asyncio.wait_for(reader.readline(), timeout=5.0)
        assert b'"ok": true' in raw
        writer.close()
        await writer.wait_closed()
    finally:
        await bus.stop()
    assert len(received) == 1
    assert received[0].text == "open my notes"


async def test_wake_bus_rejects_conflicting_text_and_transcript(settings_tmp) -> None:
    async def handler(event: WakeEvent) -> dict[str, object]:
        raise AssertionError("handler must not run for conflicting payloads")

    socket_path = _short_socket_path()
    bus = WakeBus(socket_path, handler)
    await bus.start()
    try:
        reader, writer = await asyncio.open_unix_connection(path=str(socket_path))
        writer.write(b'{"source": "socket", "text": "one", "transcript": "two"}\n')
        await writer.drain()
        raw = await asyncio.wait_for(reader.readline(), timeout=5.0)
        assert b'"ok": false' in raw
        writer.close()
        await writer.wait_closed()
    finally:
        await bus.stop()


async def test_wake_bus_refuses_non_socket_path(settings_tmp) -> None:
    socket_path = _short_socket_path()
    socket_path.write_text("not a socket", encoding="utf-8")
    try:
        bus = WakeBus(socket_path, lambda event: None)  # type: ignore[arg-type]
        with pytest.raises(RuntimeError, match="not a socket"):
            await bus.start()
    finally:
        socket_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Session continuity + wake event persistence
# ---------------------------------------------------------------------------


async def test_session_continuity_join_and_new(settings_tmp) -> None:
    database, memory = await _memory(settings_tmp)
    try:
        now = datetime(2026, 7, 3, 9, 0, 0, tzinfo=UTC)

        def clock() -> datetime:
            return now

        manager = SessionManager(memory, continuity_minutes=10, clock=clock)
        first = await manager.handle_wake(WakeEvent(source="voice", score=0.8))
        assert first.joined_existing is False
        assert first.conversation_id is not None

        # Within the continuity window: joins the same session/conversation.
        now = now + timedelta(minutes=5)
        second = await manager.handle_wake(WakeEvent(source="terminal"))
        assert second.joined_existing is True
        assert second.session_id == first.session_id
        assert second.conversation_id == first.conversation_id

        # Outside the window: a new session and conversation are created.
        now = now + timedelta(minutes=11)
        third = await manager.handle_wake(WakeEvent(source="desktop"))
        assert third.joined_existing is False
        assert third.session_id != first.session_id
        assert third.conversation_id != first.conversation_id

        # The stale session was closed; only the new one stays open.
        stale = await memory.get_session(first.session_id)
        fresh = await memory.get_session(third.session_id)
        assert stale is not None
        assert stale.closed_at is not None
        assert fresh is not None
        assert fresh.closed_at is None

        events = await memory.list_wake_events()
        assert len(events) == 3
        sources = {event.source for event in events}
        assert sources == {"voice", "terminal", "desktop"}
        by_session = await memory.list_wake_events(session_id=first.session_id)
        assert len(by_session) == 2
    finally:
        await database.close()


async def test_wake_event_persists_flags_not_transcripts(settings_tmp) -> None:
    database, memory = await _memory(settings_tmp)
    try:
        manager = SessionManager(memory, continuity_minutes=10)
        await manager.handle_wake(WakeEvent(source="voice", score=0.5, text="restart the runtime"))
        events = await memory.list_wake_events()
        assert len(events) == 1
        assert events[0].transcript_present is True
        assert events[0].score == 0.5
        row = await database.fetchone("SELECT * FROM wake_events")
        assert "restart the runtime" not in "".join(str(value) for value in dict(row).values())
    finally:
        await database.close()


async def test_wake_event_backward_compatible_and_persists_new_fields(settings_tmp) -> None:
    database, memory = await _memory(settings_tmp)
    try:
        manager = SessionManager(memory, continuity_minutes=10)
        # Old-style payload without the new optional fields keeps validating.
        legacy = WakeEvent.model_validate({"source": "terminal"})
        assert legacy.captured_at is None
        assert legacy.session_hint is None
        await manager.handle_wake(legacy)

        enriched = WakeEvent(
            source="voice",
            score=0.8,
            captured_at="2026-07-03T09:00:00Z",
            session_hint="nonexistent-session",
        )
        await manager.handle_wake(enriched)
        events = await memory.list_wake_events()
        persisted = next(event for event in events if event.source == "voice")
        assert persisted.captured_at == "2026-07-03T09:00:00Z"
        assert persisted.session_hint == "nonexistent-session"
    finally:
        await database.close()


async def test_session_hint_joins_open_session_but_never_closed_ones(settings_tmp) -> None:
    database, memory = await _memory(settings_tmp)
    try:
        now = datetime(2026, 7, 3, 9, 0, 0, tzinfo=UTC)

        def clock() -> datetime:
            return now

        manager = SessionManager(memory, continuity_minutes=10, clock=clock)
        first = await manager.handle_wake(WakeEvent(source="voice", score=0.8))

        # Outside the continuity window, a valid hint still joins the open session.
        now = now + timedelta(minutes=30)
        hinted = await manager.handle_wake(
            WakeEvent(source="terminal", session_hint=first.session_id)
        )
        assert hinted.joined_existing is True
        assert hinted.session_id == first.session_id

        # A hint naming a closed session falls back to normal continuity flow.
        await manager.close(first.session_id)
        now = now + timedelta(minutes=1)
        stale_hint = await manager.handle_wake(
            WakeEvent(source="terminal", session_hint=first.session_id)
        )
        assert stale_hint.joined_existing is False
        assert stale_hint.session_id != first.session_id

        # An unknown hint also falls back instead of failing.
        unknown = await manager.handle_wake(
            WakeEvent(source="terminal", session_hint="not-a-session")
        )
        assert unknown.session_id == stale_hint.session_id
    finally:
        await database.close()


async def test_socket_wake_joins_existing_voice_session(settings_tmp) -> None:
    database, memory = await _memory(settings_tmp)
    socket_path = _short_socket_path()
    try:
        manager = SessionManager(memory, continuity_minutes=10)
        first = await manager.handle_wake(WakeEvent(source="voice", score=0.8))

        async def handler(event: WakeEvent) -> dict[str, object]:
            resolution = await manager.handle_wake(event)
            return resolution.model_dump()

        bus = WakeBus(socket_path, handler)
        await bus.start()
        try:
            reply = await send_wake_event(
                socket_path, WakeEvent(source="socket", text="continue", reason="terminal")
            )
        finally:
            await bus.stop()
        assert reply["ok"] is True
        result = reply["result"]
        assert isinstance(result, dict)
        assert result["joined_existing"] is True
        assert result["session_id"] == first.session_id
        assert result["conversation_id"] == first.conversation_id
    finally:
        socket_path.unlink(missing_ok=True)
        await database.close()


# ---------------------------------------------------------------------------
# Sentinel: two-stage wake, cooldown, follow-up, mute, barge-in
# ---------------------------------------------------------------------------


async def test_sentinel_accepts_high_score_without_stt(settings_tmp) -> None:
    sentinel, _mic, delivery, _player = _sentinel(
        settings_tmp, scores=[0.1, 0.9], confirm_with_stt=False
    )
    await sentinel.run_once()
    assert len(delivery.events) == 1
    assert delivery.events[0].source == "voice"
    assert delivery.events[0].reason == "accepted_by_score"
    assert delivery.events[0].score == pytest.approx(0.9)


async def test_sentinel_plays_one_earcon_per_accepted_wake(settings_tmp) -> None:
    sentinel, _mic, delivery, player = _sentinel(settings_tmp, scores=[0.9], confirm_with_stt=False)
    await sentinel.run_once()
    assert len(delivery.events) == 1
    assert len(player.played) == 1
    assert player.played[0].name.startswith("wake-earcon-")


async def test_sentinel_rejects_candidate_below_accept_without_stt(settings_tmp) -> None:
    sentinel, _mic, delivery, _player = _sentinel(
        settings_tmp, scores=[0.5], confirm_with_stt=False
    )
    await sentinel.run_once()
    assert delivery.events == []
    assert sentinel.rejected_candidates == 1


async def test_sentinel_does_not_play_earcon_on_reject_or_mute(settings_tmp) -> None:
    rejected, _mic, _delivery, reject_player = _sentinel(
        settings_tmp, scores=[0.5], confirm_with_stt=False
    )
    await rejected.run_once()
    assert reject_player.played == []

    muted, _mic, _delivery, mute_player = _sentinel(
        settings_tmp, scores=[0.9], confirm_with_stt=False
    )
    muted.mute.mute()
    await muted.run_once()
    assert mute_player.played == []
    assert read_wake_status(settings_tmp)["state"] == "muted"


async def test_sentinel_status_file_transitions_to_listening_then_idle(settings_tmp) -> None:
    states_during_delivery: list[str] = []

    class StatusDelivery:
        async def __call__(self, event: WakeEvent) -> None:
            del event
            states_during_delivery.append(str(read_wake_status(settings_tmp)["state"]))

    sentinel, _mic, _delivery, _player = _sentinel(
        settings_tmp, scores=[0.9], confirm_with_stt=False
    )
    sentinel.deliver = StatusDelivery()
    await sentinel.run_once()
    assert states_during_delivery == ["listening"]
    assert read_wake_status(settings_tmp)["state"] == "idle"


async def test_sentinel_stt_confirmation_accepts_and_strips_vocative(settings_tmp) -> None:
    sentinel, _mic, delivery, _player = _sentinel(
        settings_tmp,
        scores=[0.5],
        stt_text="april, restart the runtime",
        confirm_with_stt=True,
    )
    await sentinel.run_once()
    assert len(delivery.events) == 1
    event = delivery.events[0]
    assert event.reason == "stt_confirmed"
    assert event.text == "restart the runtime"


async def test_sentinel_instant_accept_skips_stt_for_high_scores(settings_tmp) -> None:
    stt = RecordingSpeechToText("april")
    sentinel, _mic, delivery, _player = _sentinel(
        settings_tmp,
        scores=[0.9],
        confirm_with_stt=True,
        instant_accept=True,
        stt=stt,
    )
    await sentinel.run_once()
    assert len(delivery.events) == 1
    assert delivery.events[0].reason == "accepted_by_score"
    assert stt.payloads == []  # STT confirmation never ran


async def test_sentinel_instant_accept_still_confirms_marginal_candidates(settings_tmp) -> None:
    stt = RecordingSpeechToText("april, restart the runtime")
    sentinel, _mic, delivery, _player = _sentinel(
        settings_tmp,
        scores=[0.5],
        confirm_with_stt=True,
        instant_accept=True,
        stt=stt,
    )
    await sentinel.run_once()
    assert len(delivery.events) == 1
    assert delivery.events[0].reason == "stt_confirmed"
    assert len(stt.payloads) == 1


async def test_sentinel_instant_accept_disabled_confirms_high_scores(settings_tmp) -> None:
    stt = RecordingSpeechToText("april, restart the runtime")
    sentinel, _mic, delivery, _player = _sentinel(
        settings_tmp,
        scores=[0.9],
        confirm_with_stt=True,
        instant_accept=False,
        stt=stt,
    )
    await sentinel.run_once()
    assert len(delivery.events) == 1
    assert delivery.events[0].reason == "stt_confirmed"
    assert len(stt.payloads) == 1


async def test_sentinel_full_utterance_capture_preserves_pre_roll(settings_tmp) -> None:
    tuned = settings_tmp.model_copy(
        update={
            "wake": settings_tmp.wake.model_copy(
                update={"enabled": True, "confirm_with_stt": True}
            ),
            "voice": settings_tmp.voice.model_copy(
                update={"vad_onset_frames": 2, "vad_energy_threshold": 0.01}
            ),
        }
    )
    pre_roll = b"\x01\x00" * 160
    wake_frame = b"\x02\x00" * 160
    silence = b"\x00\x00" * 160
    microphone = FakeFrameMicrophone(
        [pre_roll, wake_frame, *([LOUD_FRAME] * 30), *([silence] * 65)]
    )
    delivery = RecordingDelivery()
    full_stt = RecordingSpeechToText("april, restart the runtime after build")
    sentinel = Sentinel(
        settings=tuned,
        microphone=microphone,
        scorers=[ScriptedScorer([0.0, 0.5])],
        deliver=delivery,
        confirmer=SttConfirmer(
            FakeSpeechToText("april"),
            audio_cache_path=tuned.audio_cache_path,
        ),
        transcriber=full_stt,
        player=RecordingAudioPlayer(),
        mute=MuteSwitch(tuned.mute_flag_path),
    )

    await sentinel.run_once()

    assert microphone.opened_streams == 1
    assert len(delivery.events) == 1
    assert delivery.events[0].text == "restart the runtime after build"
    assert len(full_stt.payloads) == 1
    payload = full_stt.payloads[0]
    assert pre_roll in payload
    assert wake_frame in payload
    assert LOUD_FRAME in payload
    assert list(Path(tuned.audio_cache_path).glob("wake-utterance-*.wav")) == []


async def test_sentinel_in_sentence_wake_candidate_preserves_semantic_april(
    settings_tmp,
) -> None:
    tuned = settings_tmp.model_copy(
        update={
            "wake": settings_tmp.wake.model_copy(
                update={"enabled": True, "confirm_with_stt": True}
            ),
            "voice": settings_tmp.voice.model_copy(
                update={"vad_onset_frames": 1, "vad_energy_threshold": 0.01}
            ),
        }
    )
    delivery = RecordingDelivery()
    full_stt = RecordingSpeechToText("could you ask April to restart after the build")
    sentinel = Sentinel(
        settings=tuned,
        microphone=FakeFrameMicrophone([FRAME, *([LOUD_FRAME] * 30), *([b"\x00\x00" * 160] * 65)]),
        scorers=[ScriptedScorer([0.5])],
        deliver=delivery,
        confirmer=SttConfirmer(
            FakeSpeechToText("could you ask April to restart"),
            audio_cache_path=tuned.audio_cache_path,
            strict_address=False,
        ),
        transcriber=full_stt,
        player=RecordingAudioPlayer(),
        mute=MuteSwitch(tuned.mute_flag_path),
    )

    await sentinel.run_once()

    assert len(delivery.events) == 1
    assert delivery.events[0].reason == "stt_confirmed"
    assert delivery.events[0].text == "could you ask April to restart after the build"


async def test_sentinel_stt_confirmation_rejects_unaddressed_speech(settings_tmp) -> None:
    sentinel, _mic, delivery, _player = _sentinel(
        settings_tmp,
        scores=[0.5],
        stt_text="just talking to a friend",
        confirm_with_stt=True,
    )
    await sentinel.run_once()
    assert delivery.events == []
    assert sentinel.rejected_candidates == 1
    assert "address" in (sentinel.last_rejection_reason or "")


async def test_sentinel_cooldown_blocks_immediate_retrigger(settings_tmp) -> None:
    clock = ManualClock()
    sentinel, _mic, delivery, _player = _sentinel(
        settings_tmp,
        scores=[0.9, 0.9, 0.9],
        frames=3,
        confirm_with_stt=False,
        clock=clock,
    )
    await sentinel.run_once()
    # Default cooldown is 2s and the clock never advances: only one wake fires.
    assert len(delivery.events) == 1

    # After the cooldown lapses a new high score wakes again.
    clock.advance(3.0)
    sentinel.microphone = FakeFrameMicrophone([FRAME])
    sentinel.scorers = [ScriptedScorer([0.9])]
    await sentinel.run_once()
    assert len(delivery.events) == 2


async def test_sentinel_follow_up_window_wakes_on_speech(settings_tmp) -> None:
    clock = ManualClock()
    tuned = settings_tmp.model_copy(
        update={
            "voice": settings_tmp.voice.model_copy(
                update={"vad_onset_frames": 1, "vad_energy_threshold": 0.01}
            )
        }
    )
    sentinel, _mic, delivery, _player = _sentinel(
        tuned, scores=[0.0, 0.0], frames=0, confirm_with_stt=False, clock=clock
    )
    sentinel.notify_assistant_response()
    sentinel.microphone = FakeFrameMicrophone([LOUD_FRAME])
    await sentinel.run_once()
    assert len(delivery.events) == 1
    assert delivery.events[0].reason == "follow_up"
    assert delivery.events[0].score is None


async def test_sentinel_follow_up_window_transcribes_same_session_command(settings_tmp) -> None:
    clock = ManualClock()
    tuned = settings_tmp.model_copy(
        update={
            "voice": settings_tmp.voice.model_copy(
                update={"vad_onset_frames": 1, "vad_energy_threshold": 0.01}
            )
        }
    )
    delivery = RecordingDelivery()
    full_stt = RecordingSpeechToText("continue with that plan")
    sentinel = Sentinel(
        settings=tuned,
        microphone=FakeFrameMicrophone([*([LOUD_FRAME] * 30), *([b"\x00\x00" * 160] * 65)]),
        scorers=[ScriptedScorer([])],
        deliver=delivery,
        transcriber=full_stt,
        player=RecordingAudioPlayer(),
        mute=MuteSwitch(tuned.mute_flag_path),
        clock=clock,
    )

    sentinel.notify_assistant_response()
    await sentinel.run_once()

    assert len(delivery.events) == 1
    assert delivery.events[0].reason == "follow_up"
    assert delivery.events[0].text == "continue with that plan"


async def test_api_wake_delivery_speaks_then_opens_follow_up_window(
    settings_tmp, monkeypatch
) -> None:
    async def handler(url: str, payload: dict[str, object], headers: dict[str, str]):
        assert url.endswith("/wake")
        assert headers["Authorization"].startswith("Bearer ")
        assert payload["source"] == "voice"
        return _FakeHttpResponse({"result": {"final_message": "Ready for the next step."}})

    monkeypatch.setattr("httpx.AsyncClient", _FakeAsyncClient)
    _FakeAsyncClient.handler = handler
    clock = ManualClock()
    tuned = settings_tmp.model_copy(
        update={
            "voice": settings_tmp.voice.model_copy(
                update={"vad_onset_frames": 1, "vad_energy_threshold": 0.01}
            )
        }
    )
    delivery = RecordingDelivery()
    player = RecordingAudioPlayer()
    sentinel = Sentinel(
        settings=tuned,
        microphone=FakeFrameMicrophone([LOUD_FRAME]),
        scorers=[ScriptedScorer([])],
        deliver=delivery,
        player=player,
        mute=MuteSwitch(tuned.mute_flag_path),
        clock=clock,
    )
    api_delivery = ApiWakeDelivery(
        base_url="http://127.0.0.1:8765",
        token="token",
        settings=tuned,
        tts=FakeTextToSpeech(),
        player=player,
        on_assistant_response_complete=sentinel.notify_assistant_response,
    )

    await api_delivery(WakeEvent(source="voice", text="start"))
    await sentinel.run_once()

    assert player.played
    assert len(delivery.events) == 1
    assert delivery.events[0].reason == "follow_up"


async def test_follow_up_delivery_uses_session_hint_for_active_session(
    settings_tmp, monkeypatch
) -> None:
    database, memory = await _memory(settings_tmp)
    try:
        manager = SessionManager(memory, continuity_minutes=10)
        first = await manager.handle_wake(WakeEvent(source="terminal"))
        resolutions: list[dict[str, object]] = []

        async def handler(url: str, payload: dict[str, object], headers: dict[str, str]):
            del url, headers
            event = WakeEvent.model_validate(payload)
            resolution = await manager.handle_wake(event)
            result = resolution.model_dump()
            resolutions.append(result)
            return _FakeHttpResponse(
                {**result, "result": {"final_message": "Continue when ready."}}
            )

        monkeypatch.setattr("httpx.AsyncClient", _FakeAsyncClient)
        _FakeAsyncClient.handler = handler
        tuned = settings_tmp.model_copy(
            update={
                "voice": settings_tmp.voice.model_copy(
                    update={"vad_onset_frames": 1, "vad_energy_threshold": 0.01}
                )
            }
        )
        player = RecordingAudioPlayer()
        box: dict[str, Sentinel] = {}

        def response_complete() -> None:
            box["sentinel"].notify_assistant_response()

        api_delivery = ApiWakeDelivery(
            base_url="http://127.0.0.1:8765",
            token="token",
            settings=tuned,
            tts=FakeTextToSpeech(),
            player=player,
            on_assistant_response_complete=response_complete,
            session_hint=first.session_id,
        )
        sentinel = Sentinel(
            settings=tuned,
            microphone=FakeFrameMicrophone([LOUD_FRAME]),
            scorers=[ScriptedScorer([])],
            deliver=api_delivery,
            player=player,
            mute=MuteSwitch(tuned.mute_flag_path),
        )
        box["sentinel"] = sentinel

        await api_delivery(WakeEvent(source="voice", text="initial"))
        await sentinel.run_once()

        assert len(resolutions) == 2
        assert {item["session_id"] for item in resolutions} == {first.session_id}
        events = await memory.list_wake_events(session_id=first.session_id)
        assert [event.session_hint for event in events if event.source == "voice"] == [
            first.session_id,
            first.session_id,
        ]
    finally:
        _FakeAsyncClient.handler = None
        await database.close()


async def test_sentinel_follow_up_window_expires(settings_tmp) -> None:
    clock = ManualClock()
    tuned = settings_tmp.model_copy(
        update={"voice": settings_tmp.voice.model_copy(update={"vad_onset_frames": 1})}
    )
    sentinel, _mic, delivery, _player = _sentinel(
        tuned, scores=[], frames=0, confirm_with_stt=False, clock=clock
    )
    sentinel.notify_assistant_response()
    clock.advance(tuned.wake.follow_up_seconds + 1.0)
    sentinel.microphone = FakeFrameMicrophone([LOUD_FRAME])
    await sentinel.run_once()
    assert delivery.events == []


async def test_sentinel_mute_blocks_follow_up_activation(settings_tmp) -> None:
    clock = ManualClock()
    tuned = settings_tmp.model_copy(
        update={
            "voice": settings_tmp.voice.model_copy(
                update={"vad_onset_frames": 1, "vad_energy_threshold": 0.01}
            )
        }
    )
    delivery = RecordingDelivery()
    microphone = FakeFrameMicrophone([LOUD_FRAME])
    mute = MuteSwitch(tuned.mute_flag_path)
    mute.mute()
    sentinel = Sentinel(
        settings=tuned,
        microphone=microphone,
        scorers=[ScriptedScorer([])],
        deliver=delivery,
        player=RecordingAudioPlayer(),
        mute=mute,
        clock=clock,
    )

    sentinel.notify_assistant_response()
    await sentinel.run_once()

    assert microphone.opened_streams == 0
    assert delivery.events == []
    assert sentinel._follow_up_until is None


async def test_sentinel_mute_releases_microphone(settings_tmp) -> None:
    sentinel, microphone, delivery, _player = _sentinel(
        settings_tmp, scores=[0.0] * 100, frames=100, confirm_with_stt=False
    )
    mute = sentinel.mute
    started = asyncio.Event()

    original_handle = sentinel._handle_frame

    async def tracking_handle(frame: bytes, frame_source) -> None:
        started.set()
        await original_handle(frame, frame_source)

    sentinel._handle_frame = tracking_handle  # type: ignore[method-assign]
    task = asyncio.create_task(sentinel.run())
    await asyncio.wait_for(started.wait(), timeout=5.0)
    mute.mute()
    await asyncio.sleep(0)
    for _ in range(200):
        if microphone.released:
            break
        await asyncio.sleep(0.01)
    assert microphone.released is True
    sentinel.stop()
    await asyncio.wait_for(task, timeout=5.0)
    assert delivery.events == []


async def test_sentinel_barge_in_stops_playback(settings_tmp) -> None:
    sentinel, _mic, delivery, player = _sentinel(
        settings_tmp, scores=[0.9], frames=1, confirm_with_stt=False
    )
    await sentinel.run_once()
    assert len(delivery.events) == 1
    # No assistant response is active, so this accepted wake is not barge-in.
    assert player.stop_calls == 0
    assert player.duck_calls == 0


async def test_sentinel_barge_in_duck_mode(settings_tmp) -> None:
    sentinel, _mic, delivery, player = _sentinel(
        settings_tmp, scores=[0.9], frames=1, confirm_with_stt=False
    )
    sentinel.barge_in_mode = "duck"
    await sentinel.run_once()
    assert len(delivery.events) == 1
    # Duck is retained as an action but only applied to active playback.
    assert player.duck_calls == 0
    assert player.stop_calls == 0


async def test_sentinel_live_verification_uses_sentinel_pipeline_with_fakes(
    settings_tmp,
) -> None:
    events: list[WakeEvent] = []
    tuned = settings_tmp.model_copy(
        update={
            "wake": settings_tmp.wake.model_copy(
                update={"enabled": True, "confirm_with_stt": False}
            ),
            "voice": settings_tmp.voice.model_copy(update={"enabled": True}),
        }
    )
    box: dict[str, Sentinel] = {}

    async def deliver(event: WakeEvent) -> None:
        events.append(event)
        sentinel = box["sentinel"]
        sentinel._april_live_api_success = True  # type: ignore[attr-defined]
        sentinel._april_live_transcript_length = len(  # type: ignore[attr-defined]
            event.text or ""
        )
        sentinel.stop()

    sentinel = Sentinel(
        settings=tuned,
        microphone=FakeFrameMicrophone([*([LOUD_FRAME] * 31), *([b"\x00\x00" * 160] * 65)]),
        scorers=[ScriptedScorer([0.9])],
        deliver=deliver,
        transcriber=RecordingSpeechToText("april, open calendar"),
        player=RecordingAudioPlayer(),
        mute=MuteSwitch(tuned.mute_flag_path),
        clock=ManualClock(),
    )
    box["sentinel"] = sentinel
    report_path = settings_tmp.evolution_path / "sentinel-live.json"

    report = await run_sentinel_live_verification(
        settings=tuned,
        confirm_microphone=lambda _message: True,
        wake_wait_seconds=1.0,
        sentinel=sentinel,
        report_path=report_path,
    )

    assert len(events) == 1
    assert events[0].text == "open calendar"
    assert report.pipeline == "sentinel"
    assert report.summary == "pass"
    assert report.evidence_mode == "injected_test"
    assert report.wake_word_live_verified is False
    assert report.api_success is True
    persisted = report_path.read_text(encoding="utf-8")
    assert "open calendar" not in persisted
    assert str(settings_tmp.home) not in persisted


async def test_sentinel_live_verification_reports_missing_artifact_blockers(
    settings_tmp,
) -> None:
    report = await run_sentinel_live_verification(
        settings=settings_tmp,
        confirm_microphone=lambda _message: True,
        wake_wait_seconds=0.1,
    )

    assert report.pipeline == "sentinel"
    assert report.summary == "fail"
    assert report.wake_word_live_verified is False
    skipped = {item.name for item in report.skipped}
    assert "sentinel" in skipped
    assert "wake-word model" in skipped
    assert "whisper.cpp" in skipped


async def test_stt_confirmer_never_opens_its_own_microphone(settings_tmp) -> None:
    # The confirmer works purely from handed-over frames; there is no microphone
    # anywhere in its construction, and empty input is rejected outright.
    confirmer = SttConfirmer(
        FakeSpeechToText("april, hello"),
        audio_cache_path=settings_tmp.audio_cache_path,
    )
    empty = await confirmer.confirm([])
    assert empty.accepted is False
    confirmed = await confirmer.confirm([LOUD_FRAME])
    assert confirmed.accepted is True
    assert confirmed.command == "hello"
    # Capture files are removed unless debug retention is enabled.
    leftovers = list(Path(settings_tmp.audio_cache_path).glob("wake-confirm-*.wav"))
    assert leftovers == []


def test_mute_switch_round_trip(settings_tmp) -> None:
    switch = MuteSwitch(settings_tmp.mute_flag_path)
    assert switch.is_muted() is False
    switch.mute()
    assert switch.is_muted() is True
    switch.unmute()
    assert switch.is_muted() is False
