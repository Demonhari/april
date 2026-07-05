from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from pathlib import Path
from typing import Literal, Protocol

from april_common.settings import AprilSettings
from services.voice.audio_player import AudioPlayer
from services.voice.microphone import Microphone, aclose_frame_source, write_pcm_wav
from services.voice.speech_to_text import SpeechToText
from services.voice.text_to_speech import TextToSpeech
from services.voice.vad import VoiceActivityDetector
from services.wake.confirmer import SttConfirmer, strip_vocative
from services.wake.ring_buffer import AudioRingBuffer
from services.wake.schemas import WakeEvent

logger = logging.getLogger(__name__)

WakeDelivery = Callable[[WakeEvent], Awaitable[None]]


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
    interrupt any assistant speech via ``AudioPlayer.stop()``/``duck()``.
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
        while not self._stopped:
            if self.mute.is_muted():
                self._follow_up_until = None
                await self._sleep(self.mute_poll_seconds)
                continue
            await self.run_once()

    async def run_once(self) -> None:
        """Consume one microphone stream until mute/stop/stream end."""
        if self.mute.is_muted() or self._stopped:
            self._follow_up_until = None
            return
        frame_source = self.microphone.frames()
        try:
            async for frame in frame_source:
                if self._stopped or self.mute.is_muted():
                    break
                await self._handle_frame(frame, frame_source)
        finally:
            # Every exit path (mute, stop, exhaustion, error) releases the mic.
            await aclose_frame_source(frame_source)

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
        for scorer in self.scorers:
            score = max(score, float(scorer.score(frame)))
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
        confirmation = await self.confirmer.confirm(self.ring_buffer.snapshot())
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
        if self.transcriber is not None:
            text = await self._transcribe_full_utterance(
                pre_roll,
                frame_source,
                fallback_text=text,
                speech_seen=speech_seen,
            )
        self._cooldown_until = self.clock() + self.settings.voice.wake_word_cooldown_seconds
        self.ring_buffer.clear()
        self.vad.reset()
        for scorer in self.scorers:
            reset = getattr(scorer, "reset", None)
            if callable(reset):
                reset()
        if self.player is not None:
            # Barge-in: the user speaking over APRIL always interrupts playback.
            if self.barge_in_mode == "duck":
                await self.player.duck()
            else:
                await self.player.stop()
        event = WakeEvent(source="voice", score=score, text=text, reason=reason)
        self.accepted_wakes += 1
        try:
            await self.deliver(event)
        except Exception as exc:  # delivery failure must not kill the mic loop
            logger.warning("Wake delivery failed: %s", exc)

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
        write_pcm_wav(capture_path, frames, sample_rate=self.sample_rate)
        try:
            transcript = await transcriber.transcribe(capture_path)
        except Exception as exc:
            logger.warning("Full wake utterance transcription failed: %s", exc)
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
    stt: WhisperCppSpeechToText | None = None
    if settings.voice.whisper_binary_path is None or settings.voice.whisper_model_path is None:
        raise RuntimeUnavailableError(
            "Sentinel full utterance capture requires whisper.cpp binary and model paths."
        )
    stt = WhisperCppSpeechToText(
        settings.voice.whisper_binary_path,
        settings.voice.whisper_model_path,
    )
    if settings.voice.piper_binary_path is None or settings.voice.piper_model_path is None:
        raise RuntimeUnavailableError(
            "Sentinel hands-free replies require Piper binary and model paths."
        )
    tts = PiperTextToSpeech(
        settings.voice.piper_binary_path,
        settings.voice.piper_model_path,
    )
    confirmer: SttConfirmer | None = None
    if settings.wake.confirm_with_stt:
        confirmer = SttConfirmer(
            stt,
            audio_cache_path=settings.audio_cache_path,
            strict_address=settings.wake.strict_address,
            retain_debug_audio=settings.voice.retain_debug_audio,
            fuzzy_max_distance=settings.wake.fuzzy_max_distance,
        )
    player = SoundDeviceAudioPlayer(device=settings.voice.output_device)
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
    )
    sentinel_ref = sentinel
    with contextlib.suppress(KeyboardInterrupt):
        await sentinel.run()


def main() -> None:
    from april_common.settings import get_settings

    asyncio.run(run_sentinel(get_settings()))


if __name__ == "__main__":
    main()
