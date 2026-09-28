# On-Device Stress Detector

A fully offline, on-device pipeline that turns webcam + microphone signals into a
plain-language stress interpretation using a local small language model (Phi-3 Mini
via Ollama). No cloud APIs are called anywhere in this codebase.

```
Webcam + Microphone
   → Person A: feature extraction (MediaPipe / audio DSP)
   → data/features.json   (shared contract)
   → Person B: backend watcher → local Ollama (Phi-3 Mini) → response parser
   → Streamlit dashboard (stress score, reasoning, intervention, history)
```

---

## Folder structure

```
stress-detector/
├── person_a/                     # signal acquisition (this is Person A's half)
│   ├── main.py                   # CLI entry point: --mode live | demo
│   ├── config_a.py               # capture devices, thresholds, smoothing, scoring weights
│   ├── writer.py                 # atomic writer for the shared data/features.json
│   ├── video/
│   │   ├── face_mesh.py          # OpenCV capture + MediaPipe Face Mesh (guarded imports)
│   │   └── facial_features.py    # blink rate, eye openness, brow/jaw tension, head movement
│   ├── audio/
│   │   ├── mic_capture.py        # sounddevice mic capture (guarded imports)
│   │   └── audio_features.py     # pitch, pitch variance, energy, speech rate, silence
│   ├── fusion/
│   │   └── feature_fusion.py     # combines facial+audio into one reading, smoothing, scoring
│   └── demo/
│       ├── samples.py            # calm / medium / high canned stress profiles
│       └── demo_mode.py          # replays a profile into features.json without hardware
├── backend/                      # reasoning pipeline (Person B's half)
│   ├── watcher.py                # polls data/features.json for new readings
│   ├── ollama_client.py          # talks to the local Ollama HTTP API
│   ├── response_parser.py        # validates/repairs the model's JSON output
│   ├── data_store.py             # thread-safe shared state + history persistence
│   └── pipeline.py               # wires watcher -> ollama -> parser -> store together
├── ui/
│   └── dashboard.py              # Streamlit dashboard
├── prompts/
│   └── stress_prompt.py          # the strict-JSON, no-diagnosis prompt template
├── models/
│   └── schemas.py                # FeatureData / SLMResponse dataclasses (the shared contract)
├── utils/
│   ├── config.py                 # Person B tunables (model list, paths, intervals)
│   └── logger.py                 # shared logger
├── data/
│   ├── features.json             # the live shared contract file (created at runtime)
│   └── history.json              # persisted stress history (created at runtime)
├── simulate_person_a.py          # original standalone simulator (kept for backward compatibility)
├── requirements.txt              # Person B + core deps (streamlit, requests, pandas)
└── requirements-person-a.txt     # extra deps for LIVE capture (opencv, mediapipe, sounddevice)
```

---

## The shared contract: `data/features.json`

Person A overwrites this file continuously; Person B watches it. This is still the
single integration point between the two halves — nothing else needs to change if
you swap either side out.

```json
{
  "timestamp": "2026-08-07T12:30:15",
  "stress_score": 74,
  "stress_level": "High",
  "blink_rate": 23,
  "eye_openness": 0.63,
  "mouth_tension": 0.71,
  "brow_tension": 0.68,
  "head_movement": 0.24,
  "face_confidence": 0.94,
  "voice_pitch": 228,
  "pitch_variance": 31.5,
  "voice_energy": 0.77,
  "speech_rate": 138,
  "silence_detected": false,
  "confidence": 0.91
}
```

`brow_tension`, `face_confidence`, `pitch_variance`, and `silence_detected` are new
fields added to support Person A's live pipeline. **They're all optional** — the
original simulator, older recordings, and the schema/dashboard/prompt all still work
fine if these are missing or `null`. `mouth_tension` doubles as the jaw-tension proxy.

If you ever need a socket instead of a file, only `person_a/writer.py` (writer side)
and `backend/watcher.py` (reader side) need to change — everything else is unaffected.

---

## Setup

### 1. Install Ollama and pull a model

```bash
# Install Ollama: https://ollama.com/download
ollama serve            # starts the local Ollama server (usually auto-starts)
ollama pull phi3:mini   # preferred model
# Fallback if phi3 isn't available on your machine:
ollama pull gemma2:2b
```

The app auto-detects whichever of these is installed (see `PREFERRED_MODELS` in
`utils/config.py`) — no code changes needed if you use a fallback model.

### 2. Install Python dependencies

```bash
cd stress-detector
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

This is enough to run the dashboard and demo mode. Live webcam/mic capture needs one
more install:

```bash
pip install -r requirements-person-a.txt
```

Notes on that install:
- `mediapipe` wheels are large (~50–100MB); give it a few minutes.
- `sounddevice` needs the native PortAudio library on your system:
  - macOS: `brew install portaudio`
  - Ubuntu/Debian: `sudo apt-get install libportaudio2`
  - Windows: usually works out of the box via the bundled wheel.

### 3. Run the app

You need **two processes**: one feeding `data/features.json`, one running the
dashboard.

**Terminal 1 — feature source.** Pick one:

```bash
# Option A: live webcam + microphone
python -m person_a.main --mode live

# Option B: demo mode (no camera/mic needed — great for a reliable demo)
python -m person_a.main --mode demo --profile calm
python -m person_a.main --mode demo --profile medium
python -m person_a.main --mode demo --profile high

# Option C: the original random simulator (kept for backward compatibility)
python simulate_person_a.py
```

Live-mode flags:
```bash
python -m person_a.main --mode live --no-audio          # webcam only
python -m person_a.main --mode live --no-video           # mic only
python -m person_a.main --mode live --camera-index 1     # pick a different webcam
```

**Terminal 2 — dashboard:**

```bash
streamlit run ui/dashboard.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

---

## What the dashboard shows

- **Offline badge** — always visible, confirms no cloud calls are made.
- **Camera / Microphone status** — derived from how recently `features.json` was
  updated (active vs. no signal).
- **Stress score / level / confidence** — from the latest local-model interpretation.
- **Reasoning** — plain-language explanation grounded only in the supplied features.
- **Suggested intervention** — one short wellness action.
- **Generate New Intervention button** — re-runs the SLM on the most recent reading
  without waiting for a new one from Person A.
- **History graph** — stress score over time, persisted across restarts.
- **"Waiting for live data..."** — shown instead of crashing if no reading has
  arrived yet or the feed goes stale.

---

## How Person A works

- **Video** (`person_a/video/`): OpenCV grabs frames; MediaPipe Face Mesh (configured
  for a single primary face, so extra people in frame are ignored) supplies 468
  landmarks. `facial_features.py` computes blink rate (eye-aspect-ratio based),
  eye openness, brow tension and jaw/mouth tension (geometric proxies, not clinical
  measurements), head movement (nose-tip jitter), and a face-confidence score that
  decays smoothly if the face is briefly lost instead of snapping to zero.
- **Audio** (`person_a/audio/`): `sounddevice` streams the mic into a ring buffer.
  `audio_features.py` estimates pitch via autocorrelation, tracks rolling pitch
  variance, computes RMS-based energy, flags silence, and approximates a speech-rate
  from amplitude-envelope peaks — all lightweight, no heavy ML models.
- **Fusion** (`person_a/fusion/feature_fusion.py`): combines whatever modalities are
  currently available (gracefully handling either one going away), applies an
  exponential moving average so readings don't jitter frame-to-frame, and computes a
  heuristic (explicitly non-clinical) stress score, level, and confidence.
- **Demo mode** (`person_a/demo/`): three canned profiles (`calm`, `medium`, `high`)
  for a hardware-free, reliably repeatable demo.
- Camera/mic import failures, a missing device, multiple faces, or silence are all
  handled without crashing — see `person_a/main.py`'s guarded imports and try/except
  boundaries around every I/O call.

---

## Design choices worth knowing for the demo

- **Strict JSON via Ollama's `format: "json"` option** — combined with a parser that
  strips markdown fences and validates required keys, so a slightly-off model
  response never breaks the UI (`backend/response_parser.py`).
- **No diagnoses, ever** — the prompt (`prompts/stress_prompt.py`) explicitly
  forbids medical claims and requires the model to flag uncertainty when the
  incoming `confidence` is low.
- **Atomic writes** — Person A writes to a `.tmp` file and renames it into place, so
  Person B's watcher never reads a half-written file mid-write.
- **Background thread + Streamlit** — the polling/inference loop runs in a daemon
  thread (`backend/pipeline.py`) started once per process, so the UI stays
  responsive and reruns don't spawn duplicate loops.
- **Fully modular** — swapping the model, the transport (file → socket), the capture
  hardware, or the UI framework only touches one or two files each.

---

## Integration test checklist (all verified during development)

1. Person A (live or demo) continuously updates `data/features.json`. ✅
2. Person B's watcher detects each new reading via file-mtime polling. ✅
3. Ollama receives the updated features through the strict-JSON prompt. ✅
4. The dashboard reflects the new score/level/confidence. ✅
5. Reasoning text changes to match the new features (verified across
   calm → medium → high transitions). ✅
6. "Generate New Intervention" re-runs inference on the latest reading on demand. ✅
7. The history graph accumulates and persists across restarts. ✅
8. Truncated/corrupt JSON, a missing file, a lost camera, or a lost mic are all
   handled without crashing anywhere in the pipeline. ✅
