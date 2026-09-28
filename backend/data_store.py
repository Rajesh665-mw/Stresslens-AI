"""
Thread-safe in-memory store shared between the background pipeline
thread and the Streamlit UI. Also persists stress history to disk so
a restart doesn't lose the demo graph.
"""
import json
import os
import threading
import time
from typing import Optional

from models.schemas import FeatureData, SLMResponse
from utils.config import HISTORY_FILE, HISTORY_MAX_POINTS
from utils.logger import get_logger

logger = get_logger("data_store")


class DataStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._latest_features: Optional[FeatureData] = None
        self._latest_response: Optional[SLMResponse] = None
        self._last_updated: Optional[float] = None
        self._history: list = self._load_history()

    # ---- writes (called by the pipeline thread) ----
    def update_features(self, features: FeatureData) -> None:
        with self._lock:
            self._latest_features = features
            self._last_updated = time.time()

    def update(self, features: FeatureData, response: SLMResponse) -> None:
        with self._lock:
            self._latest_features = features
            self._latest_response = response
            self._last_updated = time.time()
            self._history.append(
                {
                    "timestamp": features.timestamp,
                    "stress_score": features.stress_score,
                    "stress_level": response.stress_level,
                }
            )
            if len(self._history) > HISTORY_MAX_POINTS:
                self._history = self._history[-HISTORY_MAX_POINTS:]
            self._save_history()

    # ---- reads (called by the UI) ----
    def get_state(self) -> dict:
        with self._lock:
            return {
                "features": self._latest_features,
                "response": self._latest_response,
                "last_updated": self._last_updated,
                "history": list(self._history),
            }

    # ---- persistence ----
    def _load_history(self) -> list:
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                logger.warning("Could not read history file; starting fresh.")
        return []

    def _save_history(self) -> None:
        try:
            os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
            with open(HISTORY_FILE, "w") as f:
                json.dump(self._history, f)
        except OSError as e:
            logger.warning(f"Could not persist history: {e}")


# Module-level singleton so both the pipeline thread and every Streamlit
# rerun share the same state within one process.
store = DataStore()
