from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from april_common.settings import AprilSettings

RequestOrigin = Literal["text", "voice", "wake", "agent", "unknown"]
VoicePrerequisites = Literal["not_checked", "unknown"]


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Application-owned, request-scoped provenance and voice facts.

    This deliberately describes transport and configuration only.  A request
    reaching the voice endpoint is still just text JSON; it is not evidence of
    microphone capture, transcription quality, speaker identity, or playback.
    """

    origin: RequestOrigin = "unknown"
    voice_supported: bool = True
    voice_configured: bool | None = None
    voice_enabled: bool | None = None
    voice_prerequisites: VoicePrerequisites = "unknown"
    voice_last_verified: str = "unknown"

    @classmethod
    def from_origin(cls, origin: RequestOrigin, settings: AprilSettings) -> RequestContext:
        voice = settings.voice
        configured = all(
            path is not None
            for path in (
                voice.effective_transcription_whisper_binary_path,
                voice.effective_transcription_whisper_model_path,
                voice.piper_binary_path,
                voice.piper_model_path,
            )
        )
        return cls(
            origin=origin,
            voice_configured=configured,
            voice_enabled=voice.enabled,
            voice_prerequisites="not_checked",
        )

    @classmethod
    def unknown(cls) -> RequestContext:
        return cls()

    @property
    def turn_description(self) -> str:
        return {
            "text": "text chat request",
            "voice": "text transcript received at APRIL's voice endpoint",
            "wake": "text transcript received from the wake/session path",
            "agent": "direct structured-agent request",
            "unknown": "request origin was not supplied",
        }[self.origin]

    @property
    def transport_warning(self) -> str:
        if self.origin == "voice":
            return (
                "The voice endpoint confirms text transport only; it does not prove "
                "physical capture, transcription correctness, speaker identity, or playback."
            )
        if self.origin == "wake":
            return (
                "The wake/session path confirms a text event only; it does not prove "
                "wake detection, physical capture, transcription correctness, or playback."
            )
        if self.origin == "text":
            return "This request arrived as text, so no audio was supplied for this turn."
        return "No audio evidence was supplied for this turn."


def _state(value: bool | None) -> str:
    if value is None:
        return "unknown"
    return "yes" if value else "no"


def render_request_context(context: RequestContext) -> list[str]:
    """Render bounded request facts for trusted application context."""

    return [
        "REQUEST PROVENANCE AND VOICE INTERFACE:",
        f"- Request origin: {context.origin} ({context.turn_description}).",
        "- APRIL supports an optional local microphone, speech-recognition, and "
        "speech-output interface.",
        "- The language model receives text, including a transcript; it does not "
        "receive the original sound.",
        f"- Voice interface configured: {_state(context.voice_configured)}; "
        f"enabled: {_state(context.voice_enabled)}; prerequisites: "
        f"{context.voice_prerequisites}; last live verification: "
        f"{context.voice_last_verified}.",
        f"- Current-turn evidence: {context.transport_warning}",
        "- Do not claim always-on listening, accurate transcription, acoustic "
        "perception, speaker recognition, or that playback was heard without "
        "specific evidence.",
    ]
