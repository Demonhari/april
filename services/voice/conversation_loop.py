from __future__ import annotations

import asyncio
import contextlib
import inspect
import sys
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

from apps.cli.client import AprilApiClient
from april_common.settings import get_settings
from services.voice.audio_player import AudioPlayer, SoundDeviceAudioPlayer
from services.voice.endpointing import (
    EndpointMetrics,
    capture_streamed_utterance,
    endpoint_detector_from_settings,
)
from services.voice.microphone import (
    Microphone,
    SoundDeviceMicrophone,
    aclose_frame_source,
    write_pcm_wav,
)
from services.voice.push_to_talk import PushToTalkSession
from services.voice.speech_to_text import SpeechToText, WhisperCppSpeechToText
from services.voice.text_to_speech import PiperTextToSpeech, TextToSpeech
from services.voice.vad import VoiceActivityDetector
from services.voice.wake_word import OpenWakeWordDetector

# A capture strategy turns "the user wants to talk" into a finished WAV file.
# Injecting it lets the loop use either fixed-duration recording (deterministic,
# for scripts/--seconds) or an interactive stop-controlled session, without the
# loop duplicating any recording logic.
CaptureStrategy = Callable[[Path], Awaitable[Path]]
LineReader = Callable[[], str | Awaitable[str]]


async def _prepend_frame(
    first_frame: bytes, frame_source: AsyncIterator[bytes]
) -> AsyncIterator[bytes]:
    yield first_frame
    async for frame in frame_source:
        yield frame


class VoiceTimeout(RuntimeError):
    """Raised when no wake word arrives within the wake-word waiting timeout."""


class VoiceUtteranceRejected(RuntimeError):
    """A safe automatic-capture rejection that must not reach STT or Core."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Voice utterance was not accepted ({reason}); please try again.")
        self.reason = reason


async def read_stdin_line() -> str:
    """Read one terminal line without leaving a cancellable worker thread.

    APRIL supports interactive voice control on macOS terminals. ``add_reader``
    makes the wait owned by the event loop, so cancellation removes the file
    descriptor watcher instead of abandoning an ``input()`` call in the default
    executor.
    """

    loop = asyncio.get_running_loop()
    try:
        file_descriptor = sys.stdin.fileno()
    except (AttributeError, OSError, ValueError) as exc:
        raise RuntimeError("Interactive terminal input is unavailable.") from exc
    result: asyncio.Future[str] = loop.create_future()

    def ready() -> None:
        with contextlib.suppress(Exception):
            loop.remove_reader(file_descriptor)
        try:
            line = sys.stdin.readline()
        except (OSError, ValueError):
            if not result.done():
                result.set_exception(RuntimeError("Interactive terminal input failed."))
            return
        if not line:
            if not result.done():
                result.set_exception(EOFError)
            return
        if not result.done():
            result.set_result(line)

    try:
        loop.add_reader(file_descriptor, ready)
    except (NotImplementedError, OSError, RuntimeError) as exc:
        raise RuntimeError("Interactive terminal input is unavailable.") from exc
    try:
        return await result
    finally:
        with contextlib.suppress(Exception):
            loop.remove_reader(file_descriptor)


async def _read_line(reader: LineReader) -> str:
    value = reader()
    if inspect.isawaitable(value):
        return await value
    return value


def interactive_capture_strategy(
    microphone: Microphone,
    *,
    max_seconds: float,
    read_line: LineReader = read_stdin_line,
    announce: Callable[[str], None],
) -> CaptureStrategy:
    """Build an Enter-to-start / Enter-to-stop push-to-talk capture strategy.

    Capture ends on the first of: the explicit stop, ``max_seconds``, the frame
    source ending, cancellation, or error. The microphone is always released by
    ``PushToTalkSession`` on every exit path.
    """

    async def capture(output_path: Path) -> Path:
        announce("Press Enter to start recording...")
        await _read_line(read_line)
        session = PushToTalkSession(microphone, max_seconds=max_seconds)
        announce("Recording... press Enter to stop.")

        async def _watch_stop() -> None:
            with contextlib.suppress(EOFError, OSError, RuntimeError):
                await _read_line(read_line)
            session.request_stop()

        stop_task = asyncio.create_task(_watch_stop())
        try:
            return await session.capture(output_path)
        finally:
            stop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await stop_task

    return capture


def normalize_transcript(text: str, *, wake_word: str | None = None) -> str:
    normalized = " ".join(text.split())
    if wake_word and normalized.lower().startswith(wake_word.lower()):
        normalized = normalized[len(wake_word) :].lstrip(" ,.:;")
    return normalized


class NoSpeechDetected(ValueError):
    """The speech adapter returned no usable user utterance."""


def require_usable_transcript(text: str, *, wake_word: str | None = None) -> str:
    """Normalize STT output and reject the known no-speech placeholder exactly."""

    normalized = normalize_transcript(text, wake_word=wake_word)
    if not normalized:
        raise NoSpeechDetected("Voice transcript was empty; no usable speech was detected.")
    if normalized.casefold() == "[blank_audio]":
        raise NoSpeechDetected("No usable speech was detected.")
    return normalized


class PushToTalkLoop:
    def __init__(
        self,
        *,
        api_client: AprilApiClient,
        microphone: Microphone | None = None,
        stt: SpeechToText | None = None,
        tts: TextToSpeech | None = None,
        player: AudioPlayer | None = None,
        conversation_id: str | None = None,
        record_seconds: float | None = None,
        capture: CaptureStrategy | None = None,
        transcript_observer: Callable[[str], None] | None = None,
    ) -> None:
        settings = get_settings()
        self.settings = settings
        max_seconds = record_seconds or settings.voice.max_record_seconds
        self.api_client = api_client
        self.microphone = microphone or SoundDeviceMicrophone(
            device=settings.voice.input_device,
            max_seconds=max_seconds,
        )
        # Default capture is fixed-duration recording (resolved lazily in
        # run_once); the CLI can inject an interactive stop-controlled strategy.
        self._capture: CaptureStrategy | None = capture
        self.stt = stt or WhisperCppSpeechToText(
            settings.voice.whisper_binary_path,
            settings.voice.whisper_model_path,
        )
        self.tts = tts or PiperTextToSpeech(
            settings.voice.piper_binary_path,
            settings.voice.piper_model_path,
        )
        self.player = player or SoundDeviceAudioPlayer(device=settings.voice.output_device)
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.record_seconds = max_seconds
        self.transcript_observer = transcript_observer
        self.vad = VoiceActivityDetector(
            energy_threshold=settings.voice.vad_energy_threshold,
            required_frames=settings.voice.vad_onset_frames,
        )

    async def run_once(self) -> str:
        self.settings.audio_cache_path.mkdir(parents=True, exist_ok=True)
        audio_path = self.settings.audio_cache_path / f"{uuid.uuid4()}.wav"
        tts_path = self.settings.audio_cache_path / f"{uuid.uuid4()}-reply.wav"
        capture = self._capture or self.microphone.record_push_to_talk
        try:
            spoken_path = await capture(audio_path)
            text = require_usable_transcript(
                await self.stt.transcribe(spoken_path), wake_word="april"
            )
            if self.transcript_observer is not None:
                self.transcript_observer(text)
            response = await self.api_client.post(
                "/voice/input",
                {"message": text, "conversation_id": self.conversation_id},
            )
            answer = response["result"]["final_message"]
            output_path = await self.tts.synthesize(answer, tts_path)
            await self.player.play(output_path)
            return answer
        finally:
            # Temporary audio is removed on every exit path (success or error)
            # unless the user opted to retain it for debugging.
            if not self.settings.voice.retain_debug_audio:
                for path in (audio_path, tts_path):
                    if Path(path).exists():
                        Path(path).unlink()


class VoiceConversationLoop(PushToTalkLoop):
    async def run_forever(self) -> None:
        while True:
            await self.run_once()


class WakeWordConversationLoop(PushToTalkLoop):
    def __init__(
        self, *args: object, detector: OpenWakeWordDetector | None = None, **kwargs: object
    ):
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.detector = detector or OpenWakeWordDetector(
            self.settings.voice.wake_word_model_path,
            threshold=self.settings.voice.wake_word_threshold,
            cooldown_seconds=self.settings.voice.wake_word_cooldown_seconds,
        )
        self.last_endpoint_metrics: EndpointMetrics | None = None

    async def run_once(self) -> str:
        if not self.detector.available():
            return await super().run_once()
        self.settings.audio_cache_path.mkdir(parents=True, exist_ok=True)
        audio_path = self.settings.audio_cache_path / f"{uuid.uuid4()}-utterance.wav"
        tts_path = self.settings.audio_cache_path / f"{uuid.uuid4()}-reply.wav"
        try:
            spoken_path = await self._capture_wake_utterance(audio_path)
            if self.last_endpoint_metrics is None or self.last_endpoint_metrics.stop_reason in {
                "no_speech",
                "too_short",
                "muted",
                "stopped",
            }:
                reason = (
                    self.last_endpoint_metrics.stop_reason
                    if self.last_endpoint_metrics is not None
                    else "no_speech"
                )
                raise VoiceUtteranceRejected(reason)
            text = require_usable_transcript(
                await self.stt.transcribe(spoken_path), wake_word="april"
            )
            if self.transcript_observer is not None:
                self.transcript_observer(text)
            response = await self.api_client.post(
                "/voice/input",
                {"message": text, "conversation_id": self.conversation_id},
            )
            answer = response["result"]["final_message"]
            output_path = await self.tts.synthesize(answer, tts_path)
            await self.player.play(output_path)
            return answer
        finally:
            if not self.settings.voice.retain_debug_audio:
                for path in (audio_path, tts_path):
                    Path(path).unlink(missing_ok=True)

    async def run_forever(self) -> None:
        try:
            while True:
                await self.run_once()
        except KeyboardInterrupt:
            return

    async def _capture_wake_utterance(
        self, output_path: Path, *, clock: Callable[[], float] = time.monotonic
    ) -> Path:
        voice = self.settings.voice
        self.last_endpoint_metrics = None
        # Reset detector and VAD at the conversation boundary so a prior
        # utterance cannot leak into this one.
        self.vad.reset()
        detector_reset = getattr(self.detector, "reset", None)
        if callable(detector_reset):
            detector_reset()

        pre_roll: deque[bytes] = deque(maxlen=max(0, voice.wake_pre_roll_frames))
        frames: list[bytes] = []
        wake_seen = False
        # The wake-word waiting timeout runs from the start; the utterance timeout
        # only begins once the wake word (or push-to-talk) has activated.
        wake_deadline = clock() + voice.wake_wait_seconds

        frame_source = self.microphone.frames()
        try:
            async for frame in frame_source:
                if not wake_seen:
                    pre_roll.append(frame)
                    if self.detector.detect(frame):
                        wake_seen = True
                        # Pre-roll recovers the audio captured while the wake word
                        # was being confirmed, so the onset is not discarded.
                        frames.extend(pre_roll)
                        pre_roll.clear()
                    elif clock() >= wake_deadline:
                        raise VoiceTimeout("No wake word detected before the wake timeout.")
                    continue
                captured = await capture_streamed_utterance(
                    _prepend_frame(frame, frame_source),
                    endpoint_detector=endpoint_detector_from_settings(voice),
                    pre_roll=frames,
                    clock=clock,
                    close_source=False,
                )
                frames = list(captured.frames)
                self.last_endpoint_metrics = captured.metrics
                break
        finally:
            # Release the microphone stream on every exit path (break, timeout,
            # cancellation, or shutdown).
            await aclose_frame_source(frame_source)
        if not frames:
            raise ValueError("Voice utterance was empty.")
        if self.last_endpoint_metrics is None:
            # This is reachable only when an injected source ends immediately
            # after wake activation.
            detector = endpoint_detector_from_settings(voice)
            self.last_endpoint_metrics = detector.finish()
        return write_pcm_wav(output_path, frames, sample_rate=16_000)
