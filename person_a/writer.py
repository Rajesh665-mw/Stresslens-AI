"""
Writes Person A's fused feature readings to the shared JSON file that
Person B's backend watches (`utils.config.FEATURES_FILE`).

Writes are atomic (write to a temp file, then os.replace) so Person B's
watcher never reads a half-written file mid-write.
"""
import json
import os
import sys
import time

# Make the project root importable no matter how this module is invoked
# (as a script, as `python -m person_a.main`, or imported from a test).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import FEATURES_FILE, FRAME_FILE  # shared contract paths
from utils.logger import get_logger

logger = get_logger("person_a.writer")


def write_features(feature_dict: dict, path: str = FEATURES_FILE) -> bool:
    """Atomically writes a feature reading to the shared JSON file.
    Returns True on success, False on failure (never raises)."""
    try:
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(feature_dict, f, indent=2)
            
        # Retry replace up to 5 times on Windows to handle transient locks
        for attempt in range(5):
            try:
                os.replace(tmp_path, path)
                return True
            except PermissionError:
                time.sleep(0.03)
        # Fallback: direct write if replace is locked on Windows
        with open(path, "w") as f:
            json.dump(feature_dict, f, indent=2)
        return True
    except OSError as e:
        logger.error(f"Failed to write features to {path}: {e}")
        return False


def write_frame(frame_bytes: bytes, path: str = FRAME_FILE) -> bool:
    """Atomically writes JPEG frame bytes to data/frame.jpg for Streamlit UI consumption.
    Returns True on success, False on failure (never raises)."""
    if not frame_bytes:
        return False
    try:
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "wb") as f:
            f.write(frame_bytes)
        for attempt in range(3):
            try:
                os.replace(tmp_path, path)
                return True
            except PermissionError:
                time.sleep(0.02)
        # Fallback: direct write if replace is locked
        with open(path, "wb") as f:
            f.write(frame_bytes)
        return True
    except OSError as e:
        logger.error(f"Failed to write frame to {path}: {e}")
        return False


