# Wake-word model training runbook (openWakeWord)

This is a manual, local runbook for producing an "april" wake-word model that
the Sentinel can load. Nothing in this directory downloads models or data
automatically, and APRIL never trains in the background. Every step below is
something you run yourself, on your own machine, with data you recorded or
generated locally.

## What the Sentinel expects

- One or more openWakeWord models (`.onnx` or `.tflite`) whose paths are listed
  in `configs/april.yaml` under `voice.wake_word_model_path` (single) or
  `voice.wake_word_model_paths` (several, additive).
- Each model scores 16 kHz 16-bit mono PCM frames and outputs a confidence in
  `[0, 1]`. Scores are compared against `wake.candidate_threshold` and
  `wake.accept_threshold` (two-stage detection; marginal candidates are
  confirmed by local STT from the ring buffer).

## Prerequisites (install manually, never in the daemon)

1. A separate Python environment for training (training deps must not be added
   to APRIL's base or dev constraints):

   ```sh
   python3 -m venv ~/.venvs/oww-train
   ~/.venvs/oww-train/bin/pip install openwakeword
   ```

   For full custom training follow the upstream project's training notebook and
   requirements: https://github.com/dscripka/openWakeWord (see
   `notebooks/automatic_model_training.ipynb`). Pin versions yourself and
   review the upstream license before use.

2. Training data. Two workable options, in increasing effort/quality order:
   - **Synthetic positives**: generate "april" utterances with a local TTS
     (e.g. Piper voices you already have) at 16 kHz mono, plus negative clips
     of general speech/noise. The upstream notebook automates this pattern.
   - **Recorded positives**: record yourself (and anyone who will use APRIL)
     saying "april", "hey april", "okay april" in the rooms where the Mac
     lives. 100–300 clips materially beat pure synthetic data. You can reuse
     `april voice enroll` captures (`data/voice_profiles/…/enroll-*.wav`) as
     positive seed clips — they are already 16 kHz mono WAV.

## Training steps (summary of the upstream flow)

1. Organize data locally:

   ```
   training_data/
     positive/   # WAV, 16 kHz mono, ~1s each, containing "april"
     negative/   # WAV, 16 kHz mono, speech/noise NOT containing "april"
   ```

2. Run the upstream openWakeWord training flow (notebook or script) against
   that directory. It produces a model file such as `april.onnx`.

3. Evaluate before installing: play held-out positive and negative clips
   through the model and check the score distribution. You want the positive
   scores comfortably above `wake.accept_threshold` (default 0.70) and the
   negative scores below `wake.candidate_threshold` (default 0.35).

## Installing the model

1. Copy the model somewhere stable inside APRIL_HOME, e.g.
   `models/wake/april.onnx` (model binaries are never committed).
2. Point the config at it:

   ```yaml
   voice:
     wake_word_model_path: models/wake/april.onnx
   ```

3. Enable voice + wake (both are required before apriald supervises the
   Sentinel):

   ```yaml
   voice:
     enabled: true
   wake:
     enabled: true
   ```

4. Verify end-to-end with the existing live check: `run april wake live`
   (microphone required; this is the only step that listens).

## Tuning

- False wakes: raise `wake.candidate_threshold` / `wake.accept_threshold`, or
  keep `wake.confirm_with_stt: true` (default) so marginal candidates require a
  local STT transcript that actually addresses APRIL.
- Missed wakes: lower `wake.candidate_threshold` first; STT confirmation keeps
  precision. The confirmer also accepts close STT mishearings (apryl, avril,
  aprill, "a pril") via a bounded edit-distance match.
- Strict addressing: `wake.strict_address: true` requires a leading/trailing
  vocative ("april, …" / "… april") and rejects mid-sentence mentions.

## Known blocker: speaker verification (speaker gate)

`wake.speaker_gate` supports `"off"` and `"soft"`. APRIL ships the local verifier
adapter but not a speaker-embedding model. `april voice enroll` records local
enrollment samples under `data/voice_profiles/`, but enrollment alone does not
change wake behaviour. Follow `scripts/speaker_verifier/README.md` to supply and
validate a compatible local ONNX before enabling `soft`.
