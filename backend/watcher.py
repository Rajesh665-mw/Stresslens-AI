"""
Watches for new physiological feature readings written by Person A's
pipeline. Deliberately simple + defensive: file-based polling, so the
team can swap in a socket later without changing the contract
(read_latest() always returns Optional[FeatureData], never raises).
"""
import json
import os
import time
from typing import Optional

from models.schemas import FeatureData
from utils.config import FEATURES_FILE
from utils.logger import get_logger

logger = get_logger("watcher")


class FeatureWatcher:
    """Polls a JSON file for new readings from Person A's extraction pipeline."""

    def __init__(self, path: str = FEATURES_FILE):
        self.path = path
        self._last_mtime: Optional[float] = None
        self._last_features: Optional[FeatureData] = None

    def has_new_data(self) -> bool:
        """Cheap check using file modification time before doing a full read."""
        if not os.path.exists(self.path):
            return False
        mtime = os.path.getmtime(self.path)
        return self._last_mtime is None or mtime > self._last_mtime

    def read_latest(self) -> Optional[FeatureData]:
        """Reads and parses the latest features file.
        Returns None (and logs) on any failure instead of raising, so a
        half-written file from Person A never crashes the backend."""
        if not os.path.exists(self.path):
            return None
        try:
            mtime = os.path.getmtime(self.path)
            with open(self.path, "r") as f:
                raw = json.load(f)
            features = FeatureData.from_dict(raw)
            self._last_mtime = mtime
            self._last_features = features
            return features
        except (json.JSONDecodeError, ValueError, OSError) as e:
            logger.warning(f"Failed to read features file ({e}); keeping last known reading.")
            return None

    def last_known(self) -> Optional[FeatureData]:
        return self._last_features

    def seconds_since_last_write(self) -> Optional[float]:
        if self._last_mtime is None:
            return None
        return time.time() - self._last_mtime
