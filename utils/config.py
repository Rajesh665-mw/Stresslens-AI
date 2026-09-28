"""
Central configuration for the Stress Detector application.
Adjust these values to change model selection, polling behavior, and file paths.
"""
import os

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FEATURES_FILE = os.path.join(DATA_DIR, "features.json")   # Person A writes here
FRAME_FILE = os.path.join(DATA_DIR, "frame.jpg")         # Person A live video frame writes here
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

OLLAMA_REQUEST_TIMEOUT = 30  # seconds
OLLAMA_TEMPERATURE = 0.3     # low temperature -> more consistent JSON

# ---- Polling / Pipeline ----
POLL_INTERVAL_SECONDS = 1          # how often to check for new features from Person A
STALE_DATA_SECONDS = 45            # accommodate 15-second calculation interval before declaring inactive
HISTORY_MAX_POINTS = 300           # cap in-memory / on-disk history length

# ---- UI ----
APP_TITLE = "On-Device Stress Detector"
UI_REFRESH_SECONDS = 2
