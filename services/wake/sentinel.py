from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Literal, Protocol

from april_common.settings import AprilSettings
from services.voice.audio_player import AudioPlayer
from services.voice.microphone import Microphone, aclose_frame_source
from services.voice.vad import VoiceActivityDetector
from services.wake.confirmer import SttConfirmer
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
        player: AudioPlayer | None = None,
        vad: VoiceActivityDetector | None = None,
        mute: MuteSwitch | None = None,
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
        self.player = player
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
        follow_up = self.settings.wake.follow_up_seconds
        if follow_up > 0:
            self._follow_up_until = self.clock() + follow_up
            self.vad.reset()

    async def run(self) -> None:
        """Own the microphone until stopped. Mute fully releases the stream."""
        while not self._stopped:
            if self.mute.is_muted():
                await self._sleep(self.mute_poll_seconds)
                continue
            await self.run_once()

    async def run_once(self) -> None:
        """Consume one microphone stream until mute/stop/stream end."""
        if self.mute.is_muted() or self._stopped:
            return
        frame_source = self.microphone.frames()
        try:
            async for frame in frame_source:
                if self._stopped or self.mute.is_muted():
                    break
                await self._handle_frame(frame)
        finally:
            # Every exit path (mute, stop, exhaustion, error) releases the mic.
            await aclose_frame_source(frame_source)

    async def _handle_frame(self, frame: bytes) -> None:
        self.ring_buffer.append(frame)
        now = self.clock()
        if self._follow_up_window_open(now):
            if self.vad.is_speech(frame):
                self._follow_up_until = None
                await self._accept(score=None, reason="follow_up", text=None)
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
                await self._accept(score=score, reason="accepted_by_score", text=None)
            else:
                self._reject(score, "below accept threshold without STT confirmation")
            return
        if self.confirmer is None:
            # Confirmation is required but unavailable: only a high-confidence
            # score may wake, so a marginal candidate can never slip through.
            if score >= wake.accept_threshold:
                await self._accept(score=score, reason="accepted_by_score", text=None)
            else:
                self._reject(score, "no STT confirmer available")
            return
        confirmation = await self.confirmer.confirm(self.ring_buffer.snapshot())
        if confirmation.accepted:
            await self._accept(
                score=score,
                reason="stt_confirmed",
                text=confirmation.command or None,
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

    async def _accept(self, *, score: float | None, reason: str, text: str | None) -> None:
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

    def _reject(self, score: float, reason: str) -> None:
        self.rejected_candidates += 1
        self.last_rejection_reason = reason
        logger.debug("Wake candidate rejected (score=%.3f): %s", score, reason)


class ApiWakeDelivery:
    """Deliver accepted wakes to the loopback Core API POST /wake."""

    def __init__(self, *, base_url: str, token: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    async def __call__(self, event: WakeEvent) -> None:
        import httpx

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/wake",
                json=event.model_dump(),
                headers={"Authorization": f"Bearer {self.token}"},
            )
            response.raise_for_status()


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


async def run_sentinel(settings: AprilSettings) -> None:
    """Production entry point: real microphone, wake models, STT confirmation.

    Requires ``voice.enabled`` and ``wake.enabled`` plus at least one wake model.
    Raises rather than pretending to listen when prerequisites are missing.
    """
    from april_common.errors import RuntimeUnavailableError
    from services.voice.audio_player import SoundDeviceAudioPlayer
    from services.voice.microphone import SoundDeviceMicrophone
    from services.voice.speech_to_text import WhisperCppSpeechToText

    if not settings.voice.enabled or not settings.wake.enabled:
        raise RuntimeUnavailableError("Sentinel requires voice.enabled and wake.enabled.")
    scorers = build_scorers(settings)
    if not scorers:
        raise RuntimeUnavailableError("Sentinel requires at least one wake-word model path.")
    confirmer: SttConfirmer | None = None
    if settings.wake.confirm_with_stt:
        confirmer = SttConfirmer(
            WhisperCppSpeechToText(
                settings.voice.whisper_binary_path,
                settings.voice.whisper_model_path,
            ),
            audio_cache_path=settings.audio_cache_path,
            strict_address=settings.wake.strict_address,
            retain_debug_audio=settings.voice.retain_debug_audio,
        )
    sentinel = Sentinel(
        settings=settings,
        microphone=SoundDeviceMicrophone(device=settings.voice.input_device),
        scorers=scorers,
        deliver=ApiWakeDelivery(
            base_url=f"http://{settings.api.host}:{settings.api.port}",
            token=settings.api.token,
        ),
        confirmer=confirmer,
        player=SoundDeviceAudioPlayer(device=settings.voice.output_device),
    )
    with contextlib.suppress(KeyboardInterrupt):
        await sentinel.run()
