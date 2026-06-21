# Voice Design

Voice is optional and disabled by default.

Pipeline:

```mermaid
flowchart LR
  Mic[Microphone] --> Wake[Wake word]
  Wake --> VAD[VAD]
  VAD --> STT[whisper.cpp adapter]
  STT --> API[APRIL API]
  API --> TTS[Piper adapter]
  TTS --> Speaker[Audio player]
```

Rules:

- no downloads
- no microphone activation at API startup
- explicit CLI/service invocation only
- temporary audio under configured cache
- recordings removed by default
- fake STT/TTS are used in tests

Implemented adapters:

- `SoundDeviceMicrophone` records explicit push-to-talk captures as bounded
  16 kHz mono 16-bit PCM WAV and reports actionable macOS permission errors.
- `SoundDeviceAudioPlayer` validates WAV input, supports a configured output
  device, and plays through `sounddevice` only when explicitly invoked.
- `VoiceActivityDetector` provides deterministic energy-based VAD with
  configurable threshold and required speech frames.
- `OpenWakeWordDetector` lazily imports openWakeWord, requires an explicitly
  configured local ONNX model, validates 16-bit PCM frames, and never downloads
  models.

CLI commands:

```bash
april voice health
april voice doctor
april voice devices
april voice test-record --seconds 3
april voice test-stt /path/to/audio.wav
april voice test-tts "Hello Hari"
april voice ptt
april voice listen
```

`voice ptt` keeps a persistent conversation ID for the loop, transcribes with
the configured local whisper.cpp adapter, passes `conversation_id` through
`/voice/input`, synthesizes with Piper, plays the response, and removes
temporary audio unless debug retention is enabled. `voice listen` is optional
wake-word mode and falls back to explicit push-to-talk behavior when wake-word
support is unavailable.
