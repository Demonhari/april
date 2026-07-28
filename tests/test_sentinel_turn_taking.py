from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from services.voice.endpointing import EndpointMetrics
from services.voice.microphone import Microphone
from services.wake.fakes import (
    ManualClock,
    RecordingAudioPlayer,
    RecordingAudit,
    ScriptedScorer,
)
from services.wake.response_coordinator import ResponseState
from services.wake.schemas import WakeEvent
from services.wake.sentinel import MuteSwitch, Sentinel

FRAME = b"\x00\x01" * 160
LOUD_FRAME = b"\x00\x40" * 160


async def _empty_frames() -> AsyncIterator[bytes]:
    if False:
        yield b""


class _ObservedMicrophone(Microphone):
    def __init__(self, count: int) -> None:
        self.count = count
        self.consumed = 0
        self.all_consumed = asyncio.Event()

    async def frames(self) -> AsyncIterator[bytes]:
        for _ in range(self.count):
            self.consumed += 1
            if self.consumed == self.count:
                self.all_consumed.set()
            await asyncio.sleep(0)
            yield FRAME


class _PendingGenerationDelivery:
    def __init__(self, *, speaking: bool = False) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.speaking = speaking
        self.calls = 0

    async def __call__(self, event: WakeEvent) -> None:
        del event

    async def deliver_generation(
        self,
        event: WakeEvent,
        *,
        generation: int,
        is_current,
        set_state,
    ) -> None:
        del event
        self.calls += 1
        self.started.set()
        if self.speaking:
            set_state("speaking", generation)
        await self.release.wait()
        assert is_current(generation)


class _FirstPlaybackBlocks:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, event: WakeEvent) -> None:
        del event

    async def deliver_generation(
        self,
        event: WakeEvent,
        *,
        generation: int,
        is_current,
        set_state,
    ) -> None:
        del event
        self.calls += 1
        if self.calls == 1:
            set_state("speaking", generation)
            await asyncio.Event().wait()
        assert is_current(generation)


async def test_microphone_consumes_frames_while_api_delivery_is_pending(settings_tmp) -> None:
    mic = _ObservedMicrophone(8)
    delivery = _PendingGenerationDelivery()
    tuned = settings_tmp.model_copy(
        update={
            "voice": settings_tmp.voice.model_copy(
                update={"wake_word_cooldown_seconds": 30.0}
            ),
            "wake": settings_tmp.wake.model_copy(
                update={"enabled": True, "confirm_with_stt": False}
            ),
        }
    )
    sentinel = Sentinel(
        settings=tuned,
        microphone=mic,
        scorers=[ScriptedScorer([0.9])],
        deliver=delivery,
        player=RecordingAudioPlayer(),
        mute=MuteSwitch(tuned.mute_flag_path),
    )
    task = asyncio.create_task(sentinel.run_once())
    await asyncio.wait_for(delivery.started.wait(), timeout=1)
    await asyncio.wait_for(mic.all_consumed.wait(), timeout=1)
    assert task.done() is False
    assert mic.consumed == 8
    delivery.release.set()
    await task


async def test_microphone_consumes_frames_while_playback_state_is_active(settings_tmp) -> None:
    mic = _ObservedMicrophone(8)
    delivery = _PendingGenerationDelivery(speaking=True)
    tuned = settings_tmp.model_copy(
        update={
            "voice": settings_tmp.voice.model_copy(
                update={"wake_word_cooldown_seconds": 30.0}
            ),
            "wake": settings_tmp.wake.model_copy(
                update={"enabled": True, "confirm_with_stt": False}
            ),
        }
    )
    sentinel = Sentinel(
        settings=tuned,
        microphone=mic,
        scorers=[ScriptedScorer([0.9])],
        deliver=delivery,
        player=RecordingAudioPlayer(),
        mute=MuteSwitch(tuned.mute_flag_path),
    )
    task = asyncio.create_task(sentinel.run_once())
    await asyncio.wait_for(delivery.started.wait(), timeout=1)
    await asyncio.wait_for(mic.all_consumed.wait(), timeout=1)
    assert sentinel.response_coordinator.playback_active is True
    delivery.release.set()
    await task


async def test_second_wake_from_live_microphone_interrupts_active_playback(settings_tmp) -> None:
    mic = _ObservedMicrophone(3)
    delivery = _FirstPlaybackBlocks()
    player = RecordingAudioPlayer()
    tuned = settings_tmp.model_copy(
        update={
            "voice": settings_tmp.voice.model_copy(
                update={
                    "wake_word_cooldown_seconds": 0.0,
                    "barge_in_trigger": "wake_word",
                    "barge_in_action": "stop",
                }
            ),
            "wake": settings_tmp.wake.model_copy(
                update={"enabled": True, "confirm_with_stt": False}
            ),
        }
    )
    sentinel = Sentinel(
        settings=tuned,
        microphone=mic,
        scorers=[ScriptedScorer([0.9, 0.0, 0.9])],
        deliver=delivery,
        player=player,
        mute=MuteSwitch(tuned.mute_flag_path),
    )
    await sentinel.run_once()
    assert delivery.calls == 2
    assert player.stop_calls == 1
    assert "accepted_by_score" in sentinel.response_coordinator.interrupt_reasons


async def test_barge_in_off_ignores_microphone_while_response_is_active(settings_tmp) -> None:
    delivery = _PendingGenerationDelivery(speaking=True)
    player = RecordingAudioPlayer()
    tuned = settings_tmp.model_copy(
        update={
            "voice": settings_tmp.voice.model_copy(update={"barge_in_trigger": "off"}),
            "wake": settings_tmp.wake.model_copy(
                update={"enabled": True, "confirm_with_stt": False}
            ),
        }
    )
    sentinel = Sentinel(
        settings=tuned,
        microphone=_ObservedMicrophone(0),
        scorers=[ScriptedScorer([0.9] * 10)],
        deliver=delivery,
        player=player,
        mute=MuteSwitch(tuned.mute_flag_path),
    )
    await sentinel.response_coordinator.submit(WakeEvent(source="voice"))
    await asyncio.wait_for(delivery.started.wait(), timeout=1)
    for _ in range(10):
        await sentinel._handle_frame(LOUD_FRAME, _empty_frames())
    assert player.stop_calls == 0
    assert sentinel.accepted_wakes == 0
    delivery.release.set()
    await sentinel.response_coordinator.drain()


async def test_speech_barge_in_is_conservative_and_honours_playback_grace(
    settings_tmp,
) -> None:
    clock = ManualClock()
    delivery = _FirstPlaybackBlocks()
    player = RecordingAudioPlayer()
    tuned = settings_tmp.model_copy(
        update={
            "voice": settings_tmp.voice.model_copy(
                update={
                    "barge_in_trigger": "speech",
                    "barge_in_speech_onset_frames": 6,
                    "barge_in_playback_grace_ms": 400,
                }
            ),
            "wake": settings_tmp.wake.model_copy(
                update={"enabled": True, "confirm_with_stt": False}
            ),
        }
    )
    sentinel = Sentinel(
        settings=tuned,
        microphone=_ObservedMicrophone(0),
        scorers=[ScriptedScorer([0.0] * 20)],
        deliver=delivery,
        player=player,
        mute=MuteSwitch(tuned.mute_flag_path),
        clock=clock,
    )
    await sentinel.response_coordinator.submit(WakeEvent(source="voice"))
    await asyncio.sleep(0)
    for _ in range(6):
        await sentinel._handle_frame(LOUD_FRAME, _empty_frames())
    assert player.stop_calls == 0
    clock.advance(0.5)
    for _ in range(5):
        await sentinel._handle_frame(LOUD_FRAME, _empty_frames())
    assert player.stop_calls == 0
    await sentinel._handle_frame(LOUD_FRAME, _empty_frames())
    await sentinel.response_coordinator.drain()
    assert player.stop_calls == 1
    assert "speech_barge_in" in sentinel.response_coordinator.interrupt_reasons


def test_public_response_states_include_turn_taking_states() -> None:
    states: set[ResponseState] = {
        "idle",
        "listening",
        "capturing",
        "thinking",
        "speaking",
        "interrupted",
        "degraded",
    }
    assert len(states) == 7


def test_endpoint_audit_contains_safe_metrics_only(settings_tmp) -> None:
    audit = RecordingAudit()
    tuned = settings_tmp.model_copy(
        update={
            "wake": settings_tmp.wake.model_copy(
                update={"enabled": True, "confirm_with_stt": False}
            )
        }
    )
    sentinel = Sentinel(
        settings=tuned,
        microphone=_ObservedMicrophone(0),
        scorers=[],
        deliver=_PendingGenerationDelivery(),
        audit=audit,
        mute=MuteSwitch(tuned.mute_flag_path),
    )
    sentinel._audit_endpoint(
        EndpointMetrics(
            stop_reason="end_of_speech",
            frame_count=95,
            speech_frame_count=30,
            captured_duration_ms=950,
            speech_duration_ms=300,
            trailing_silence_ms=650,
            endpoint_latency_ms=650,
            calibrated_noise_floor=0.002,
            effective_energy_threshold=0.01,
            speech_started=True,
            minimum_duration_met=True,
        ),
        source_type="accepted_by_score",
    )
    serialized = str(audit.records)
    assert "transcript" not in serialized
    assert "response" not in serialized
    assert "path" not in serialized
    assert "audio" not in serialized
