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
