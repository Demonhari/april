# Speaker-verifier model runbook (local ONNX)

This is a manual, local runbook for supplying the speaker-embedding model used
by Sentinel's soft speaker gate. APRIL never downloads a model, contacts a model
hub, or exports one in the background. Model binaries must stay uncommitted.

The speaker gate is a convenience filter for accidental wakes. It is not
authentication, does not establish identity, and never grants access or changes
the deterministic permission engine's decisions.

## What the verifier expects

- One local ONNX model configured as `wake.speaker_verifier_model_path`.
- One float32 input containing normalized 16 kHz mono waveform samples, either
  rank 1 (`samples`) or rank 2 (`batch, samples`).
- One output embedding that can be flattened to a finite, non-empty vector.
- Bounded audio: enrollment WAVs must be uncompressed 16 kHz mono int16 and at
  most 15 seconds; utterance scoring uses at most the most recent 15 seconds.

Sentinel embeds each WAV created by `april voice enroll`, averages those
enrollment embeddings, and cosine-compares the wake utterance. The cosine value
is deterministically mapped from `[-1, 1]` to `[0, 1]`; 0.5 is the default pass
threshold.

## Obtain or export a model manually

1. Choose a small speaker-embedding model whose license and training data are
   acceptable for your use. Common ECAPA-TDNN or x-vector checkpoints can be
   exported, but their native graphs often expect filter-bank features rather
   than waveform samples.
2. Manually download the chosen checkpoint and its license using a browser or a
   separate operator-controlled environment. Do not add download code to APRIL,
   and do not place credentials in this repository.
3. In a separate export environment, wrap any required 16 kHz waveform
   preprocessing (for example log-mel/filter-bank extraction) together with the
   embedding network, then export a single-input ONNX matching the contract
   above. Use a fixed model version and validate the exported output against the
   original model on held-out local WAVs.
4. Copy the resulting file to a stable ignored location such as
   `models/speaker/operator-embedding.onnx`. Do not commit it.

APRIL intentionally provides no auto-download or one-click export command: the
operator must review the source, license, preprocessing, and output quality of
the model used on the target Mac.

## Enroll and configure

1. Install the optional voice dependencies in APRIL's environment. ONNX Runtime
   is present only in this extra because the base/dev install does not need it:

   ```sh
   pip install -e '.[voice]'
   ```

2. Record several samples in the rooms and microphone positions you use:

   ```sh
   april voice enroll --samples 5 --seconds 3
   ```

3. Configure the local model and enable soft mode:

   ```yaml
   wake:
     speaker_gate: "soft"
     speaker_verifier_model_path: models/speaker/operator-embedding.onnx
   ```

4. Run `run april readiness`, restart Sentinel, and validate both matching and
   non-matching speakers on the target Mac. Enrollment never enables the gate by
   itself. Keep `speaker_gate: "off"` if the score distributions do not separate
   reliably.

## Audited degradation and rejection events

Soft mode fails open because it is not a security boundary. Sentinel writes at
most one `speaker_gate_degraded` startup/runtime event and then behaves as if the
gate were off:

- `local_verifier_unavailable`: soft mode has no configured model path.
- `model_missing`: the configured path does not name a regular file.
- `onnxruntime_unavailable`: the optional local runtime cannot be imported.
- `model_load_failed`: ONNX Runtime could not create a session for the model;
  check the graph, provider compatibility, and input/output contract.
- `local_verifier_failed`: enrollment parsing, embedding, or inference failed
  while Sentinel was running.

A valid score below the threshold does not degrade the adapter. Sentinel drops
that wake and audits `wake_dropped` with `reason: speaker_gate`. These audit
records contain no audio or embeddings.
