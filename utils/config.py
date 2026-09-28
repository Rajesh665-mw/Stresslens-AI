"""
Central configuration for the Stress Detector application.
Adjust these values to change model selection, polling behavior, and file paths.
"""
import os

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FEATURES_FILE = os.path.join(DATA_DIR, "features.json")   # perception pipeline writes here
FRAME_FILE = os.path.join(DATA_DIR, "frame.jpg")         # perception live video frame writes here
HISTORY_FILE = os.path.join(DATA_DIR, "history.json")     # persisted stress history

# ---- Ollama ----
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Preference order. The app will pick the first one that is actually
# installed/pulled in the local Ollama instance.
PREFERRED_MODELS = [
    "phi3:mini",
    "phi3",
    "gemma2:2b",
    "gemma:2b",
    "llama3.2:1b",
]

OLLAMA_REQUEST_TIMEOUT = int(os.environ.get("OLLAMA_REQUEST_TIMEOUT", "30"))  # seconds
OLLAMA_TEMPERATURE = float(os.environ.get("OLLAMA_TEMPERATURE", "0.3"))       # low temperature -> more consistent JSON

# ---- Polling / Pipeline ----
POLL_INTERVAL_SECONDS = float(os.environ.get("POLL_INTERVAL_SECONDS", "1"))    # check frequency for new perception features
STALE_DATA_SECONDS = float(os.environ.get("STALE_DATA_SECONDS", "45"))        # calculation interval threshold for staleness
HISTORY_MAX_POINTS = int(os.environ.get("HISTORY_MAX_POINTS", "300"))         # cap in-memory / on-disk history length

# ---- UI ----
APP_TITLE = os.environ.get("APP_TITLE", "StressLens AI — On-Device Stress Detector")
UI_REFRESH_SECONDS = int(os.environ.get("UI_REFRESH_SECONDS", "2"))
