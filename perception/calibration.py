"""
Baseline Calibration Engine for the perception signal acquisition pipeline.

Collects resting/normal physiological & vocal features during an initial
startup period (typically 10-15 seconds) to establish the user's personal
baseline. Subsequent feature computations measure relative deviations from
this baseline rather than relying on fixed arbitrary thresholds.
"""
import time
import numpy as np
from typing import Optional, Dict, Any
from utils.logger import get_logger

logger = get_logger("perception.calibration")


class BaselineCalibrator:
    def __init__(self, calibration_duration: float = 12.0):
        self.calibration_duration = calibration_duration
        self.start_time: Optional[float] = None
        self.is_complete = False
        
        # Buffer raw metric history collected during calibration
        self._facial_samples: Dict[str, list] = {
            "blink_rate": [],
            "eye_openness": [],
            "brow_gap": [],
            "inner_brow_gap": [],
            "lip_gap": [],
            "lip_thickness": [],
            "jaw_span": [],
            "head_movement": [],
            "forehead_gap": [],
            "lip_corner_droop": [],
        }
        
        self._audio_samples: Dict[str, list] = {
            "voice_pitch": [],
            "pitch_variance": [],
            "voice_energy": [],
            "speech_rate": [],
            "spectral_centroid": [],
            "jitter": [],
            "shimmer": [],
        }

        # Computed baseline means (mu) and standard deviations (sigma)
        self.facial_baselines: Dict[str, float] = {}
        self.audio_baselines: Dict[str, float] = {}

    def start(self, now: Optional[float] = None) -> None:
        if self.start_time is None:
            self.start_time = now if now is not None else time.time()
            logger.info(f"Started baseline calibration stage ({self.calibration_duration}s duration).")

    def is_calibrating(self, now: Optional[float] = None) -> bool:
        if self.is_complete:
            return False
        if self.start_time is None:
            self.start(now)
            return True
        current_time = now if now is not None else time.time()
        elapsed = current_time - self.start_time
        if elapsed >= self.calibration_duration:
            self._finalize_calibration()
            return False
        return True

    def progress(self, now: Optional[float] = None) -> float:
        if self.is_complete:
            return 1.0
        if self.start_time is None:
            return 0.0
        current_time = now if now is not None else time.time()
        elapsed = current_time - self.start_time
        return max(0.0, min(1.0, elapsed / self.calibration_duration))

    def add_facial_sample(self, sample: dict) -> None:
        if self.is_complete:
            return
        for key in self._facial_samples.keys():
            val = sample.get(key)
            if val is not None and isinstance(val, (int, float)) and not np.isnan(val):
                self._facial_samples[key].append(float(val))

    def add_audio_sample(self, sample: dict) -> None:
        if self.is_complete:
            return
        if sample.get("silence_detected", True):
            return
        for key in self._audio_samples.keys():
            val = sample.get(key)
            if val is not None and isinstance(val, (int, float)) and not np.isnan(val):
                self._audio_samples[key].append(float(val))

    def _finalize_calibration(self) -> None:
        if self.is_complete:
            return

        # Calculate facial baselines
        for key, values in self._facial_samples.items():
            if len(values) >= 5:
                # Use trimmed mean to ignore initial positioning artifacts
                clean_vals = np.sort(values)[int(len(values)*0.1): max(1, int(len(values)*0.9))]
                self.facial_baselines[key] = float(np.mean(clean_vals))
            else:
                self.facial_baselines[key] = self._default_facial_baseline(key)

        # Calculate audio baselines
        for key, values in self._audio_samples.items():
            if len(values) >= 5:
                clean_vals = np.sort(values)[int(len(values)*0.1): max(1, int(len(values)*0.9))]
                self.audio_baselines[key] = float(np.mean(clean_vals))
            else:
                self.audio_baselines[key] = self._default_audio_baseline(key)

        self.is_complete = True
        logger.info(f"Baseline calibration complete! Facial baselines: {self.facial_baselines}")
        logger.info(f"Audio baselines: {self.audio_baselines}")

    @staticmethod
    def _default_facial_baseline(key: str) -> float:
        defaults = {
            "blink_rate": 17.0,
            "eye_openness": 0.65,
            "brow_gap": 0.055,
            "inner_brow_gap": 0.09,
            "lip_gap": 0.02,
            "lip_thickness": 0.035,
            "jaw_span": 0.45,
            "head_movement": 0.1,
            "forehead_gap": 0.12,
            "lip_corner_droop": 0.04,
        }
        return defaults.get(key, 0.5)

    @staticmethod
    def _default_audio_baseline(key: str) -> float:
        defaults = {
            "voice_pitch": 160.0,
            "pitch_variance": 15.0,
            "voice_energy": 0.4,
            "speech_rate": 120.0,
            "spectral_centroid": 1800.0,
            "jitter": 0.015,
            "shimmer": 0.05,
        }
        return defaults.get(key, 0.0)

    def get_facial_baseline(self, key: str) -> float:
        return self.facial_baselines.get(key, self._default_facial_baseline(key))

    def get_audio_baseline(self, key: str) -> float:
        return self.audio_baselines.get(key, self._default_audio_baseline(key))
