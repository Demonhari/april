from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

from april_common.settings import AprilSettings
from services.voice.audio_player import AudioPlayer
from services.voice.microphone import Microphone, aclose_frame_source, write_pcm_wav
from services.voice.speech_to_text import SpeechToText
from services.voice.text_to_speech import TextToSpeech
from services.voice.vad import VoiceActivityDetector
from services.wake.confirmer import SttConfirmer, strip_vocative
from services.wake.control import SentinelControlServer, sentinel_control_path
from services.wake.ring_buffer import AudioRingBuffer
from services.wake.schemas import WakeEvent
from services.wake.speaker import SPEAKER_MATCH_THRESHOLD, SpeakerVerifier
from services.wake.status import WakeListeningState, write_wake_status

logger = logging.getLogger(__name__)

WakeDelivery = Callable[[WakeEvent], Awaitable[None]]


class AuditSink(Protocol):
    def write(self, payload: dict[str, Any]) -> None: ...


class WakeScorer(Protocol):
    """One wake model producing a raw confidence in [0, 1] per frame."""

    def score(self, frame: bytes) -> float: ...


class MuteSwitch:
    """Hard-mute flag backed by a local file so any process can flip it.

    While muted the Sentinel closes its microphone stream entirely (the OS input
    indicator goes dark); nothing is buffered or scored.
    """

    def __init__(self, flag_path: Path) -> None:
        self.flag_path = flag_path

    def is_muted(self) -> bool:
        return self.flag_path.exists()

    def mute(self) -> None:
        self.flag_path.parent.mkdir(parents=True, exist_ok=True)
        self.flag_path.write_text("muted\n", encoding="utf-8")

    def unmute(self) -> None:
        self.flag_path.unlink(missing_ok=True)


class Sentinel:
    """The single microphone owner: two-stage wake detection with pre-roll.

    Stage one scores every frame with one or more wake models. A score at or
    above ``accept_threshold`` (with STT confirmation disabled) wakes directly;
    otherwise scores at or above ``candidate_threshold`` are confirmed by local
    STT over the ring-buffer capture. STT never opens its own microphone stream.
    Accepted wakes are delivered as :class:`WakeEvent` (source ``voice``) and
    interrupt any assistant speech via ``AudioPlayer.stop()``/``duck()``. A soft
    speaker verifier may suppress delivery as a convenience filter only. It is
    never authentication and never affects APRIL permissions.
    """

    def __init__(
        self,
        *,
        settings: AprilSettings,
        microphone: Microphone,
        scorers: Sequence[WakeScorer],
        deliver: WakeDelivery,
        confirmer: SttConfirmer | None = None,
        transcriber: SpeechToText | None = None,
        player: AudioPlayer | None = None,
        vad: VoiceActivityDetector | None = None,
        mute: MuteSwitch | None = None,
        speaker_verifier: SpeakerVerifier | None = None,
        speaker_enrollment: Sequence[Path] | None = None,
        audit: AuditSink | None = None,
        wake_word: str = "april",
        sample_rate: int = 16_000,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        barge_in_mode: Literal["stop", "duck"] = "stop",
        mute_poll_seconds: float = 0.5,
    ) -> None:
        self.settings = settings
        self.microphone = microphone
        self.scorers = list(scorers)
        self.deliver = deliver
        self.confirmer = confirmer
        self.transcriber = transcriber
        self.player = player
        self.wake_word = wake_word
        self.sample_rate = sample_rate
        self.vad = vad or VoiceActivityDetector(
            energy_threshold=settings.voice.vad_energy_threshold,
            required_frames=settings.voice.vad_required_frames,
        )
        self.mute = mute or MuteSwitch(settings.mute_flag_path)
        self.speaker_verifier = speaker_verifier
        self.speaker_enrollment = tuple(
            speaker_enrollment
            if speaker_enrollment is not None
            else _speaker_enrollment_paths(settings)
        )
        self.audit = audit
        self.clock = clock
        self._sleep = sleep or asyncio.sleep
        self.barge_in_mode = barge_in_mode
        self.mute_poll_seconds = mute_poll_seconds
        self.ring_buffer = AudioRingBuffer(seconds=settings.wake.ring_buffer_seconds)
        self._stopped = False
        self._cooldown_until: float | None = None
        self._follow_up_until: float | None = None
        self.accepted_wakes = 0
        self.rejected_candidates = 0
        self.last_rejection_reason: str | None = None
        self._speaker_gate_started = False
        self._speaker_gate_degraded = False

    def stop(self) -> None:
        self._stopped = True

    def notify_assistant_response(self) -> None:
        """Open the follow-up window: speech soon after a reply wakes directly."""
        if self.mute.is_muted():
            self._follow_up_until = None
            return
        follow_up = self.settings.wake.follow_up_seconds
        if follow_up > 0:
            self._follow_up_until = self.clock() + follow_up
            self.vad.reset()

    async def run(self) -> None:
        """Own the microphone until stopped. Mute fully releases the stream."""
        self._start_speaker_gate()
        while not self._stopped:
            if self.mute.is_muted():
                self._set_status("muted")
                self._follow_up_until = None
                await self._sleep(self.mute_poll_seconds)
                continue
            await self.run_once()

    async def run_once(self) -> None:
        """Consume one microphone stream until mute/stop/stream end."""
        self._start_speaker_gate()
        if self.mute.is_muted() or self._stopped:
            self._set_status("muted" if self.mute.is_muted() else "idle")
            self._follow_up_until = None
            return
        self._set_status("idle")
        frame_source = self.microphone.frames()
        try:
            async for frame in frame_source:
                if self._stopped or self.mute.is_muted():
                    break
                await self._handle_frame(frame, frame_source)
        finally:
            # Every exit path (mute, stop, exhaustion, error) releases the mic.
            await aclose_frame_source(frame_source)
            self._set_status("muted" if self.mute.is_muted() else "idle")

    async def _handle_frame(self, frame: bytes, frame_source: AsyncIterator[bytes]) -> None:
        self.ring_buffer.append(frame)
        now = self.clock()
        if self._follow_up_window_open(now) and self.vad.is_speech(frame):
            self._follow_up_until = None
            await self._accept(
                score=None,
                reason="follow_up",
                text=None,
                frame_source=frame_source,
                speech_seen=True,
            )
            return
        if self._in_cooldown(now):
            return
        score = 0.0
        scorer_succeeded = False
        for scorer in self.scorers:
            try:
                score = max(score, float(scorer.score(frame)))
                scorer_succeeded = True
            except Exception as exc:
                self._audit_adapter_failure("wake_scorer", exc)
        if not scorer_succeeded:
            self._reset_detection_state()
            return
        wake = self.settings.wake
        if score < wake.candidate_threshold:
            return
        if not wake.confirm_with_stt:
            if score >= wake.accept_threshold:
                await self._accept(
                    score=score,
                    reason="accepted_by_score",
                    text=None,
                    frame_source=frame_source,
                )
            else:
                self._reject(score, "below accept threshold without STT confirmation")
            return
        if wake.instant_accept and score >= wake.accept_threshold:
            # High-confidence scores wake immediately; STT confirmation is
            # reserved for candidates between the two thresholds.
            await self._accept(
                score=score,
                reason="accepted_by_score",
                text=None,
                frame_source=frame_source,
            )
            return
        if self.confirmer is None:
            # Confirmation is required but unavailable: only a high-confidence
            # score may wake, so a marginal candidate can never slip through.
            if score >= wake.accept_threshold:
                await self._accept(
                    score=score,
                    reason="accepted_by_score",
                    text=None,
                    frame_source=frame_source,
                )
            else:
                self._reject(score, "no STT confirmer available")
            return
        try:
            confirmation = await self.confirmer.confirm(self.ring_buffer.snapshot())
        except Exception as exc:
            self._audit_adapter_failure("wake_confirmation_stt", exc)
            self._reset_detection_state()
            self._set_status("muted" if self.mute.is_muted() else "idle")
            return
        if confirmation.accepted:
            await self._accept(
                score=score,
                reason="stt_confirmed",
                text=confirmation.command or None,
                frame_source=frame_source,
            )
        else:
            self._reject(score, confirmation.reason)

    def _follow_up_window_open(self, now: float) -> bool:
        if self._follow_up_until is None:
            return False
        if now >= self._follow_up_until:
            self._follow_up_until = None
            return False
        return True

    def _in_cooldown(self, now: float) -> bool:
        cooldown = self.settings.voice.wake_word_cooldown_seconds
        if self._cooldown_until is None or cooldown <= 0:
            return False
        return now < self._cooldown_until

    async def _accept(
        self,
        *,
        score: float | None,
        reason: str,
        text: str | None,
        frame_source: AsyncIterator[bytes],
        speech_seen: bool = False,
    ) -> None:
        pre_roll = self.ring_buffer.snapshot()
        self._set_status("listening")
        if not self._speaker_allowed(pre_roll):
            self._cooldown_until = self.clock() + self.settings.voice.wake_word_cooldown_seconds
            self._reset_detection_state()
            self._reject(score or 0.0, "speaker_gate")
            self._audit(
                {
                    "event_type": "wake_dropped",
                    "actor": "sentinel",
                    "reason": "speaker_gate",
                }
            )
            self._set_status("muted" if self.mute.is_muted() else "idle")
            return
        if self.player is not None:
            # Barge-in: the user speaking over APRIL always interrupts playback
            # before APRIL emits the short acknowledgement earcon.
            if self.barge_in_mode == "duck":
                await self.player.duck()
            else:
                await self.player.stop()
        await self._play_earcon()
        if self.transcriber is not None:
            text = await self._transcribe_full_utterance(
                pre_roll,
                frame_source,
                fallback_text=text,
                speech_seen=speech_seen,
            )
        self._cooldown_until = self.clock() + self.settings.voice.wake_word_cooldown_seconds
        self._reset_detection_state()
        event = WakeEvent(source="voice", score=score, text=text, reason=reason)
        self.accepted_wakes += 1
        try:
            await self.deliver(event)
        except Exception as exc:  # delivery failure must not kill the mic loop
            logger.warning("Wake delivery failed: %s", exc)
        finally:
            self._set_status("muted" if self.mute.is_muted() else "idle")

    def _start_speaker_gate(self) -> None:
        if self._speaker_gate_started:
            return
        self._speaker_gate_started = True
        if self.settings.wake.speaker_gate == "soft" and self.speaker_verifier is None:
            self._degrade_speaker_gate("local_verifier_unavailable")

    def _degrade_speaker_gate(self, detail: str) -> None:
        if self._speaker_gate_degraded:
            return
        self._speaker_gate_degraded = True
        self._audit(
            {
                "event_type": "speaker_gate_degraded",
                "actor": "sentinel",
                "status": "warning",
                "reason": "speaker_gate",
                "detail": detail,
            }
        )

    def _speaker_allowed(self, frames: Sequence[bytes]) -> bool:
        if self.settings.wake.speaker_gate != "soft" or self._speaker_gate_degraded:
            return True
        verifier = self.speaker_verifier
        if verifier is None:
            return True
        try:
            match_score = float(verifier.score(self.speaker_enrollment, b"".join(frames)))
            if not math.isfinite(match_score) or not 0.0 <= match_score <= 1.0:
                raise ValueError("speaker verifier score must be between 0 and 1")
        except Exception as exc:
            logger.warning("Speaker verifier unavailable; soft gate disabled: %s", exc)
            self._degrade_speaker_gate("local_verifier_failed")
            return True
        return match_score >= SPEAKER_MATCH_THRESHOLD

    def _reset_detection_state(self) -> None:
        self.ring_buffer.clear()
        self.vad.reset()
        for scorer in self.scorers:
            reset = getattr(scorer, "reset", None)
            if callable(reset):
                reset()

    def _audit(self, payload: dict[str, Any]) -> None:
        if self.audit is None:
            return
        try:
            self.audit.write(payload)
        except Exception as exc:
            # Speaker verification is explicitly a convenience feature. Audit
            # trouble must not turn it into a wake availability/security gate.
            logger.warning("Speaker-gate audit write failed: %s", exc)

    def _audit_adapter_failure(self, adapter: str, exc: Exception) -> None:
        logger.warning("Sentinel %s failed: %s", adapter, type(exc).__name__)
        self._audit(
            {
                "event_type": "sentinel_adapter_failed",
                "actor": "sentinel",
                "adapter": adapter,
                "error_type": type(exc).__name__,
            }
        )

    async def _play_earcon(self) -> None:
        if self.player is None or not self.settings.wake.earcon_enabled:
            return
        output_path = self.settings.audio_cache_path / f"wake-earcon-{uuid.uuid4()}.wav"
        try:
            frames = [_earcon_pcm(sample_rate=self.sample_rate)]
            write_pcm_wav(output_path, frames, sample_rate=self.sample_rate)
            await self.player.play(output_path)
        except Exception as exc:
            logger.debug("Wake earcon playback skipped: %s", exc)
        finally:
            if not self.settings.voice.retain_debug_audio:
                output_path.unlink(missing_ok=True)

    async def _transcribe_full_utterance(
        self,
        pre_roll: Sequence[bytes],
        frame_source: AsyncIterator[bytes],
        *,
        fallback_text: str | None,
        speech_seen: bool,
    ) -> str | None:
        transcriber = self.transcriber
        if transcriber is None:
            return fallback_text
        frames = list(pre_roll)
        frames.extend(await self._capture_post_wake_frames(frame_source, speech_seen=speech_seen))
        if not frames:
            return fallback_text
        capture_path = self.settings.audio_cache_path / f"wake-utterance-{uuid.uuid4()}.wav"
        try:
            write_pcm_wav(capture_path, frames, sample_rate=self.sample_rate)
            transcript = await transcriber.transcribe(capture_path)
        except Exception as exc:
            self._audit_adapter_failure("utterance_transcription_stt", exc)
            return fallback_text
        finally:
            if not self.settings.voice.retain_debug_audio:
                capture_path.unlink(missing_ok=True)
        cleaned = strip_vocative(transcript, wake_word=self.wake_word)
        return cleaned or fallback_text

    async def _capture_post_wake_frames(
        self, frame_source: AsyncIterator[bytes], *, speech_seen: bool
    ) -> list[bytes]:
        vad = VoiceActivityDetector(
            energy_threshold=self.settings.voice.vad_energy_threshold,
            required_frames=self.settings.voice.vad_required_frames,
        )
        frames: list[bytes] = []
        silence_frames = 0
        deadline = self.clock() + self.settings.voice.utterance_max_seconds
        async for frame in frame_source:
            if self._stopped or self.mute.is_muted():
                break
            frames.append(frame)
            if vad.is_speech(frame):
                speech_seen = True
                silence_frames = 0
            elif speech_seen:
                silence_frames += 1
                if silence_frames >= self.settings.voice.vad_required_frames:
                    break
            if self.clock() >= deadline:
                break
        return frames

    def _reject(self, score: float, reason: str) -> None:
        self.rejected_candidates += 1
        self.last_rejection_reason = reason
        logger.debug("Wake candidate rejected (score=%.3f): %s", score, reason)

    def _set_status(self, state: WakeListeningState) -> None:
        try:
            write_wake_status(self.settings, state)
        except Exception as exc:
            logger.debug("Wake status write skipped: %s", exc)


def _earcon_pcm(*, sample_rate: int) -> bytes:
    import numpy as np

    duration_seconds = 0.15
    frequency_hz = 660.0
    samples = int(sample_rate * duration_seconds)
    t = np.arange(samples, dtype=np.float32) / float(sample_rate)
    envelope = np.linspace(1.0, 0.2, samples, dtype=np.float32)
    wave = np.sin(2.0 * np.pi * frequency_hz * t) * envelope * 0.18
    return np.asarray(wave * 32767, dtype=np.int16).tobytes()


def _speaker_enrollment_paths(settings: AprilSettings) -> tuple[Path, ...]:
    """Return regular enrollment WAVs fenced inside APRIL's profile directory."""
    profile_dir = settings.resolve_path(Path("data/voice_profiles"))
    try:
        root = profile_dir.resolve(strict=False)
    except OSError:
        return ()
    enrollment: list[Path] = []
    for candidate in sorted(profile_dir.glob("*.wav")):
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if resolved.is_relative_to(root) and resolved.is_file():
            enrollment.append(resolved)
    return tuple(enrollment)


class ApiWakeDelivery:
    """Deliver accepted wakes to the loopback Core API POST /wake.

    When TTS/player are supplied, any assistant reply returned by ``/wake`` is
    spoken locally and ``on_assistant_response_complete`` is called only after
    playback finishes. That callback is the production follow-up wake handoff;
    no speech completion is simulated.
    """

    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        timeout: float = 30.0,
        settings: AprilSettings | None = None,
        tts: TextToSpeech | None = None,
        player: AudioPlayer | None = None,
        on_assistant_response_complete: Callable[[], None] | None = None,
        session_hint: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.settings = settings
        self.tts = tts
        self.player = player
        self.on_assistant_response_complete = on_assistant_response_complete
        self.session_hint = session_hint

    async def __call__(self, event: WakeEvent) -> None:
        import httpx

        if self.session_hint and event.session_hint is None:
            event = event.model_copy(update={"session_hint": self.session_hint})
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/wake",
                json=event.model_dump(),
                headers={"Authorization": f"Bearer {self.token}"},
            )
            response.raise_for_status()
        await self._speak_response(response.json())

    async def _speak_response(self, payload: object) -> None:
        if self.settings is None or self.tts is None or self.player is None:
            return
        if not isinstance(payload, dict):
            return
        result = payload.get("result")
        if not isinstance(result, dict):
            return
        final_message = result.get("final_message")
        if not isinstance(final_message, str) or not final_message.strip():
            return
        self.settings.audio_cache_path.mkdir(parents=True, exist_ok=True)
        output_path = self.settings.audio_cache_path / f"sentinel-reply-{uuid.uuid4()}.wav"
        try:
            spoken_path = await self.tts.synthesize(final_message, output_path)
            await self.player.play(spoken_path)
        except Exception as exc:
            logger.warning("Assistant voice response playback failed: %s", exc)
            return
        finally:
            if not self.settings.voice.retain_debug_audio:
                output_path.unlink(missing_ok=True)
        if self.on_assistant_response_complete is not None:
            self.on_assistant_response_complete()


def build_scorers(settings: AprilSettings) -> list[WakeScorer]:
    """Build one openWakeWord scorer per configured wake model path."""
    from services.voice.wake_word import OpenWakeWordDetector

    scorers: list[WakeScorer] = []
    for path in settings.voice.effective_wake_word_model_paths:
        resolved = settings.resolve_path(path)
        scorers.append(
            OpenWakeWordDetector(
                resolved,
                threshold=settings.wake.candidate_threshold,
                cooldown_seconds=0.0,
            )
        )
    return scorers


async def run_sentinel(settings: AprilSettings, *, session_hint: str | None = None) -> None:
    """Production entry point: real microphone, wake models, STT confirmation.

    Requires ``voice.enabled`` and ``wake.enabled`` plus at least one wake model.
    Raises rather than pretending to listen when prerequisites are missing.
    """
    from april_common.audit import AuditLogger
    from april_common.errors import RuntimeUnavailableError
    from services.voice.audio_player import SoundDeviceAudioPlayer
    from services.voice.microphone import SoundDeviceMicrophone
    from services.voice.speech_to_text import WhisperCppSpeechToText
    from services.voice.text_to_speech import PiperTextToSpeech

    if not settings.voice.enabled or not settings.wake.enabled:
        raise RuntimeUnavailableError("Sentinel requires voice.enabled and wake.enabled.")
    scorers = build_scorers(settings)
    if not scorers:
        raise RuntimeUnavailableError("Sentinel requires at least one wake-word model path.")
    transcription_binary = settings.voice.effective_transcription_whisper_binary_path
    transcription_model = settings.voice.effective_transcription_whisper_model_path
    if transcription_binary is None or transcription_model is None:
        raise RuntimeUnavailableError(
            "Sentinel full utterance capture requires whisper.cpp binary and model paths."
        )
    stt = WhisperCppSpeechToText(
        settings.resolve_path(transcription_binary),
        settings.resolve_path(transcription_model),
    )
    piper_binary = (
        settings.resolve_path(settings.voice.piper_binary_path)
        if settings.voice.piper_binary_path is not None
        else None
    )
    piper_model = (
        settings.resolve_path(settings.voice.piper_model_path)
        if settings.voice.piper_model_path is not None
        else None
    )
    tts = (
        PiperTextToSpeech(piper_binary, piper_model)
        if piper_binary is not None
        and piper_model is not None
        and piper_binary.is_file()
        and piper_model.is_file()
        else None
    )
    confirmer: SttConfirmer | None = None
    if settings.wake.confirm_with_stt:
        confirmation_binary = settings.voice.effective_confirmation_whisper_binary_path
        confirmation_model = settings.voice.effective_confirmation_whisper_model_path
        if confirmation_binary is None or confirmation_model is None:
            raise RuntimeUnavailableError(
                "Wake confirmation requires whisper.cpp binary and model paths."
            )
        confirmation_stt = WhisperCppSpeechToText(
            settings.resolve_path(confirmation_binary),
            settings.resolve_path(confirmation_model),
        )
        confirmer = SttConfirmer(
            confirmation_stt,
            audio_cache_path=settings.audio_cache_path,
            strict_address=settings.wake.strict_address,
            retain_debug_audio=settings.voice.retain_debug_audio,
            fuzzy_max_distance=settings.wake.fuzzy_max_distance,
        )
    # A player is active only when a real TTS path exists. Missing Piper keeps
    # wake/STT/API delivery usable in truthful text-only mode and disables
    # earcons/barge-in rather than reporting fake playback success.
    player = (
        SoundDeviceAudioPlayer(device=settings.voice.output_device) if tts is not None else None
    )
    sentinel_ref: Sentinel | None = None

    def assistant_response_complete() -> None:
        if sentinel_ref is not None:
            sentinel_ref.notify_assistant_response()

    delivery = ApiWakeDelivery(
        base_url=f"http://{settings.api.host}:{settings.api.port}",
        token=settings.api.token,
        settings=settings,
        tts=tts,
        player=player,
        on_assistant_response_complete=assistant_response_complete,
        session_hint=session_hint,
    )
    sentinel = Sentinel(
        settings=settings,
        microphone=SoundDeviceMicrophone(device=settings.voice.input_device),
        scorers=scorers,
        deliver=delivery,
        confirmer=confirmer,
        transcriber=stt,
        player=player,
        # No production verifier model ships with APRIL. In soft mode Sentinel
        # records one degraded warning and behaves as off until an operator
        # supplies a local adapter implementing SpeakerVerifier.
        speaker_verifier=None,
        audit=AuditLogger(settings.audit_path),
    )
    if tts is None:
        sentinel._audit(
            {
                "event_type": "sentinel_voice_output_degraded",
                "actor": "sentinel",
                "reason": "piper_binary_or_model_missing",
            }
        )
    sentinel_ref = sentinel
    control = SentinelControlServer(
        sentinel_control_path(settings),
        set_session_hint=lambda value: setattr(delivery, "session_hint", value),
        status=lambda: {
            "state": "muted" if sentinel.mute.is_muted() else "listening",
            "voice_output": "available" if tts is not None else "degraded",
        },
    )
    try:
        await control.start()
        with contextlib.suppress(KeyboardInterrupt):
            await sentinel.run()
    finally:
        await control.close()


def main() -> None:
    from april_common.settings import get_settings

    asyncio.run(run_sentinel(get_settings()))


if __name__ == "__main__":
    main()
