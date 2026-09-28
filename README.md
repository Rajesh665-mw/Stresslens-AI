# StressLens AI — On-Device Stress Detector

A fully offline, on-device pipeline that turns webcam + microphone signals into a
plain-language stress interpretation using a local small language model (Phi-3 Mini
via Ollama). No cloud APIs are called anywhere in this codebase.

```
Webcam + Microphone
   → Perception: feature extraction (MediaPipe / audio DSP)
   → data/features.json (shared contract)
   → Backend: feature watcher → local Ollama (Phi-3 Mini) → response parser
   → Streamlit dashboard (stress score, reasoning, intervention, history)
```

---

## Folder structure

```
Stresslens AI/
├── perception/                   # Multimodal perception & signal acquisition pipeline
│   ├── main.py                   # CLI entry point: --mode live | demo
│   ├── __main__.py               # Package executable wrapper (python -m perception)
│   ├── config.py                 # Capture devices, thresholds, smoothing, scoring weights
│   ├── calibration.py            # Baseline calibration engine (personal resting baseline)
│   ├── feature_writer.py         # Atomic writer for data/features.json and data/frame.jpg
│   ├── video/
│   │   ├── face_mesh.py          # OpenCV capture + MediaPipe Face Mesh (guarded imports)
│   │   └── facial_features.py    # Blink rate, eye openness, brow/jaw tension, head movement
│   ├── audio/
│   │   ├── mic_capture.py        # sounddevice mic capture (guarded imports)
│   │   └── audio_features.py     # Pitch, pitch variance, energy, speech rate, jitter, shimmer
│   ├── fusion/
│   │   └── feature_fusion.py     # Combines facial + audio into calibrated score and confidence
│   └── demo/
│       ├── samples.py            # Calm / medium / high canned stress profiles
│       └── demo_mode.py          # Replays a profile into features.json without hardware
├── backend/                      # Reasoning & inference pipeline
│   ├── feature_watcher.py        # Polls data/features.json for new readings
│   ├── ollama_client.py          # Talks to the local Ollama HTTP API
│   ├── response_parser.py        # Validates/repairs the model's JSON output
│   ├── data_store.py             # Thread-safe shared state + history persistence
│   └── inference_pipeline.py     # Wires feature watcher -> Ollama -> parser -> data store
├── ui/                           # User interface
│   ├── dashboard.py              # Streamlit command center dashboard
│   ├── assets/
│   │   └── theme.css             # Glassmorphic dashboard styles
│   └── components/
│       ├── audio_panel.html      # Microphone visualizer component
│       ├── camera_panel.html     # Camera preview component
│       └── pipeline_flow.html    # Animated architecture flow component
├── prompts/
│   └── stress_prompt.py          # Strict-JSON, no-diagnosis prompt template
├── models/
│   └── schemas.py                # FeatureData / SLMResponse dataclasses (shared contract)
├── utils/
│   ├── config.py                 # Central backend tunables (model list, paths, intervals)
│   └── logger.py                 # Shared logger
├── data/
│   ├── features.json             # Live shared contract file (created at runtime)
│   ├── history.json              # Persisted stress history (created at runtime)
│   └── frame.jpg                 # Live camera preview snapshot
├── simulate_features.py          # Standalone synthetic feature generator
├── test_stress_detector.py       # Automated 9-condition verification test suite
├── diagnose_audio.py             # Diagnostic script for testing microphone devices
├── requirements.txt              # Core + backend + UI dependencies (streamlit, requests, pandas)
└── requirements-perception.txt   # Perception dependencies for LIVE capture (opencv, mediapipe, sounddevice)
```

---

## The shared contract: `data/features.json`

The perception engine overwrites this file continuously; the backend watcher polls it.
This is the single integration point between the perception and reasoning layers —
nothing else needs to change if you swap either side out.

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

`brow_tension`, `face_confidence`, `pitch_variance`, and `silence_detected` are
fields supporting the live pipeline. **They are all optional** — the standalone
simulator, older recordings, and the schema/dashboard/prompt all work fine if these
are missing or `null`. `mouth_tension` doubles as the jaw-tension proxy.

If you ever need a socket instead of a file, only `perception/feature_writer.py` (writer side)
and `backend/feature_watcher.py` (reader side) need to change — everything else is unaffected.

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
# Core dependencies for dashboard and demo mode
pip install -r requirements.txt

# Perception dependencies for LIVE webcam/mic capture
pip install -r requirements-perception.txt
```

Notes on dependencies:
- `mediapipe` wheels are large (~50–100MB); give it a few minutes.
- `sounddevice` needs the native PortAudio library on your system:
  - Windows: works out of the box via the bundled wheel.
  - macOS: `brew install portaudio`
  - Ubuntu/Debian: `sudo apt-get install libportaudio2`

### 3. Run the app

You need **two processes**: one feeding `data/features.json`, one running the
dashboard.

**Terminal 1 — feature source.** Pick one:

```bash
# Option A: live webcam + microphone
python -m perception.main --mode live

# Option B: demo mode (no camera/mic needed — reliable, repeatable demo)
python -m perception.main --mode demo --profile calm
python -m perception.main --mode demo --profile medium
python -m perception.main --mode demo --profile high

# Option C: standalone synthetic feature generator
python simulate_features.py
```

Live-mode flags:
```bash
python -m perception.main --mode live --no-audio          # webcam only
python -m perception.main --mode live --no-video           # mic only
python -m perception.main --mode live --camera-index 1     # pick a different webcam
```

**Terminal 2 — dashboard:**

```bash
streamlit run ui/dashboard.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

---

## Verification & Diagnostics

Run the automated test suite:
```bash
python test_stress_detector.py
```

Test microphone capture and signal levels:
```bash
python diagnose_audio.py
```

---

## What the dashboard shows

- **Offline badge** — confirms no cloud calls are made.
- **Camera / Microphone status** — derived from how recently `features.json` was
  updated (active vs. no signal).
- **Stress score / level / confidence** — from the latest local-model interpretation.
- **Reasoning** — plain-language explanation grounded only in the supplied features.
- **Suggested intervention** — one short wellness action.
- **Generate New Intervention button** — re-runs the SLM on the most recent reading
  without waiting for a new reading from perception.
- **History graph** — stress score over time, persisted across restarts.
- **"Waiting for live data..."** — shown instead of crashing if no reading has
  arrived yet or the feed goes stale.

---

## How the perception pipeline works

- **Video** (`perception/video/`): OpenCV grabs frames; MediaPipe Face Mesh (configured
  for a single primary face, so extra people in frame are ignored) supplies 468
  landmarks. `facial_features.py` computes blink rate (eye-aspect-ratio based),
  eye openness, brow tension and jaw/mouth tension (geometric proxies, not clinical
  measurements), head movement (nose-tip jitter), and a face-confidence score that
  decays smoothly if the face is briefly lost instead of snapping to zero.
- **Audio** (`perception/audio/`): `sounddevice` streams the mic into a ring buffer.
  `audio_features.py` estimates pitch via autocorrelation, tracks rolling pitch
  variance, computes RMS-based energy, flags silence, and approximates speech-rate
  from amplitude-envelope peaks, along with jitter, shimmer, and spectral centroid.
- **Calibration** (`perception/calibration.py`): establishes personal baseline
  means and standard deviations for resting facial and vocal features during an
  initial observation period.
- **Fusion** (`perception/fusion/feature_fusion.py`): combines whatever modalities are
  currently available (gracefully handling either one going away), applies an
  exponential moving average so readings don't jitter frame-to-frame, and computes a
  calibrated stress score, level, and confidence.
- **Demo mode** (`perception/demo/`): three canned profiles (`calm`, `medium`, `high`)
  for a hardware-free, reliably repeatable demo.
- Camera/mic import failures, a missing device, multiple faces, or silence are all
  handled without crashing — see `perception/main.py`'s guarded imports and try/except
  boundaries around every I/O call.

---

## Design choices worth knowing for the demo

- **Strict JSON via Ollama's `format: "json"` option** — combined with a parser that
  strips markdown fences and validates required keys, so a slightly-off model
  response never breaks the UI (`backend/response_parser.py`).
- **No diagnoses, ever** — the prompt (`prompts/stress_prompt.py`) explicitly
  forbids medical claims and requires the model to flag uncertainty when the
  incoming `confidence` is low.
- **Atomic writes** — perception writes to a `.tmp` file and renames it into place, so
  the backend feature watcher never reads a half-written file mid-write (`perception/feature_writer.py`).
- **Background thread + Streamlit** — the polling/inference loop runs in a daemon
  thread (`backend/inference_pipeline.py`) started once per process, so the UI stays
  responsive and reruns don't spawn duplicate loops.
- **Fully modular** — swapping the model, the transport (file → socket), the capture
  hardware, or the UI framework only touches one or two files each.
