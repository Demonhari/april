from __future__ import annotations

import json
from types import SimpleNamespace

from apps.runner.voice_conversation_live import run_voice_conversation_live_verification
from services.voice.endpointing import EndpointMetrics


def _metrics() -> EndpointMetrics:
    return EndpointMetrics(
        stop_reason="end_of_speech",
        frame_count=95,
        speech_frame_count=30,
        captured_duration_ms=950,
        speech_duration_ms=300,
        trailing_silence_ms=650,
        endpoint_latency_ms=650,
        calibrated_noise_floor=0.003,
        effective_energy_threshold=0.01,
        speech_started=True,
        minimum_duration_met=True,
    )


class _InjectedSentinel:
    def __init__(self, *, turns: int = 2, barge: bool = True) -> None:
        self.completed_endpoint_metrics = [_metrics() for _ in range(turns)]
        self.accepted_transcript_lengths = [12 for _ in range(turns)]
        self._follow_up_until = 1.0
        self.response_coordinator = SimpleNamespace(
            interrupt_reasons=["accepted_by_score"] if barge else [],
            last_interrupt_latency_ms=24,
            last_barge_in_latency_ms=24,
            shutdown=self._shutdown,
        )

    async def run(self) -> None:
        return None

    def stop(self) -> None:
        return None

    async def _shutdown(self) -> None:
        return None


class _Delivery:
    def __init__(self, *, turns: int = 2) -> None:
        complete = {"api_success", "tts_success", "playback_started", "playback_completed"}
        self.generation_stages = {index + 1: set(complete) for index in range(turns)}
        self.conversation_ids = ["conversation-1" for _ in range(turns)]
        self.session_ids = ["session-1" for _ in range(turns)]


async def test_injected_two_turn_evidence_is_redacted_and_never_hardware_verified(
    settings_tmp, tmp_path
) -> None:
    output = tmp_path / "conversation-live.json"
    report = await run_voice_conversation_live_verification(
        settings=settings_tmp,
        confirm_microphone=lambda _message: True,
        sentinel=_InjectedSentinel(),
        delivery=_Delivery(),
        report_path=output,
    )
    assert report.turn_count == 2
    assert report.two_turns_completed is True
    assert report.same_conversation is True
    assert report.barge_in_detected is True
    assert report.evidence_mode == "injected_test"
    assert report.voice_conversation_live_verified is False
    persisted = output.read_text(encoding="utf-8")
    assert "transcript" not in persisted.lower() or "transcript_length" in persisted
    assert str(settings_tmp.home) not in persisted


async def test_one_turn_or_failed_barge_in_cannot_pass(settings_tmp) -> None:
    one_turn = await run_voice_conversation_live_verification(
        settings=settings_tmp,
        confirm_microphone=lambda _message: True,
        sentinel=_InjectedSentinel(turns=1),
        delivery=_Delivery(turns=1),
    )
    no_barge = await run_voice_conversation_live_verification(
        settings=settings_tmp,
        confirm_microphone=lambda _message: True,
        sentinel=_InjectedSentinel(barge=False),
        delivery=_Delivery(),
    )
    assert one_turn.two_turns_completed is False
    assert one_turn.voice_conversation_live_verified is False
    assert no_barge.barge_in_detected is False
    assert no_barge.voice_conversation_live_verified is False


async def test_microphone_denial_does_not_run_injected_sentinel(settings_tmp) -> None:
    sentinel = _InjectedSentinel()

    async def forbidden() -> None:
        raise AssertionError("microphone pipeline must not run")

    sentinel.run = forbidden  # type: ignore[method-assign]
    report = await run_voice_conversation_live_verification(
        settings=settings_tmp,
        confirm_microphone=lambda _message: False,
        sentinel=sentinel,
    )
    assert report.turn_count == 0
    assert report.warning_codes == ["microphone_not_authorized"]


def test_report_contains_no_free_form_content_fields(settings_tmp) -> None:
    payload = json.loads(
        '{"report_type":"voice_conversation_live","evidence_mode":"injected_test"}'
    )
    assert "transcript" not in payload
    assert "response" not in payload
