from __future__ import annotations

import asyncio
import os
import stat
import tempfile
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from services.memory.database import Database
from services.memory.migrations import run_migrations
from services.memory.sqlite_memory import SqliteMemory
from services.voice.speech_to_text import FakeSpeechToText, SpeechToText
from services.wake.confirmer import (
    SttConfirmer,
    canonicalize_wake_word,
    edit_distance,
    is_addressed,
    normalized_edit_distance,
    strip_vocative,
)
from services.wake.fakes import (
    FakeFrameMicrophone,
    ManualClock,
    RecordingAudioPlayer,
    RecordingDelivery,
    ScriptedScorer,
)
from services.wake.ring_buffer import AudioRingBuffer
from services.wake.schemas import WakeEvent
from services.wake.sentinel import MuteSwitch, Sentinel
from services.wake.session_manager import SessionManager
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


def _short_socket_path() -> Path:
    return Path(tempfile.gettempdir()).resolve() / f"aw-{os.getpid()}-{uuid.uuid4().hex[:8]}.sock"


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
    clock: ManualClock | None = None,
) -> tuple[Sentinel, FakeFrameMicrophone, RecordingDelivery, RecordingAudioPlayer]:
    wake_update: dict[str, object] = {"enabled": True}
    if confirm_with_stt is not None:
        wake_update["confirm_with_stt"] = confirm_with_stt
    tuned = settings.model_copy(
        update={"wake": settings.wake.model_copy(update=wake_update)}
    )
    microphone = FakeFrameMicrophone([FRAME] * frames)
    delivery = RecordingDelivery()
    player = RecordingAudioPlayer()
    confirmer = None
    if stt_text is not None:
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
        await manager.handle_wake(
            WakeEvent(source="voice", score=0.5, text="restart the runtime")
        )
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


async def test_sentinel_rejects_candidate_below_accept_without_stt(settings_tmp) -> None:
    sentinel, _mic, delivery, _player = _sentinel(
        settings_tmp, scores=[0.5], confirm_with_stt=False
    )
    await sentinel.run_once()
    assert delivery.events == []
    assert sentinel.rejected_candidates == 1


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


async def test_sentinel_full_utterance_capture_preserves_pre_roll(settings_tmp) -> None:
    tuned = settings_tmp.model_copy(
        update={
            "wake": settings_tmp.wake.model_copy(
                update={"enabled": True, "confirm_with_stt": True}
            ),
            "voice": settings_tmp.voice.model_copy(
                update={"vad_required_frames": 2, "vad_energy_threshold": 0.01}
            ),
        }
    )
    pre_roll = b"\x01\x00" * 160
    wake_frame = b"\x02\x00" * 160
    silence = b"\x00\x00" * 160
    microphone = FakeFrameMicrophone(
        [pre_roll, wake_frame, LOUD_FRAME, LOUD_FRAME, silence, silence]
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
                update={"vad_required_frames": 1, "vad_energy_threshold": 0.01}
            ),
        }
    )
    delivery = RecordingDelivery()
    full_stt = RecordingSpeechToText("could you ask April to restart after the build")
    sentinel = Sentinel(
        settings=tuned,
        microphone=FakeFrameMicrophone([FRAME, LOUD_FRAME, b"\x00\x00" * 160]),
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
                update={"vad_required_frames": 1, "vad_energy_threshold": 0.01}
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
                update={"vad_required_frames": 1, "vad_energy_threshold": 0.01}
            )
        }
    )
    delivery = RecordingDelivery()
    full_stt = RecordingSpeechToText("continue with that plan")
    sentinel = Sentinel(
        settings=tuned,
        microphone=FakeFrameMicrophone([LOUD_FRAME, b"\x00\x00" * 160]),
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


async def test_sentinel_follow_up_window_expires(settings_tmp) -> None:
    clock = ManualClock()
    tuned = settings_tmp.model_copy(
        update={
            "voice": settings_tmp.voice.model_copy(update={"vad_required_frames": 1})
        }
    )
    sentinel, _mic, delivery, _player = _sentinel(
        tuned, scores=[], frames=0, confirm_with_stt=False, clock=clock
    )
    sentinel.notify_assistant_response()
    clock.advance(tuned.wake.follow_up_seconds + 1.0)
    sentinel.microphone = FakeFrameMicrophone([LOUD_FRAME])
    await sentinel.run_once()
    assert delivery.events == []


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
    assert player.stop_calls == 1
    assert player.duck_calls == 0


async def test_sentinel_barge_in_duck_mode(settings_tmp) -> None:
    sentinel, _mic, delivery, player = _sentinel(
        settings_tmp, scores=[0.9], frames=1, confirm_with_stt=False
    )
    sentinel.barge_in_mode = "duck"
    await sentinel.run_once()
    assert len(delivery.events) == 1
    assert player.duck_calls == 1
    assert player.stop_calls == 0


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
