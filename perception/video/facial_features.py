"""
Computes calibrated, temporal facial features from MediaPipe Face Mesh landmarks.

Tracks temporal behavior (EAR blink duration, brow furrowing, lip compression,
jaw clenching, facial rigidity, head pose) across frames. Integrates with
BaselineCalibrator to measure relative deviations from the user's resting state.
"""
import math
import time
from collections import deque
from typing import Optional, Dict, Any, List

import numpy as np

from perception.config import (
    EAR_BLINK_THRESHOLD,
    BLINK_MIN_FRAMES_CLOSED,
    BLINK_RATE_WINDOW_SECONDS,
    DEFAULT_BLINK_RATE,
    HEAD_MOVEMENT_HISTORY,
    FACE_CONFIDENCE_DECAY,
    MEDIAN_FILTER_WINDOW,
    BROW_FURROW_STRESS_SCALE,
    LIP_COMPRESSION_STRESS_SCALE,
    JAW_CLENCH_STRESS_SCALE,
    BLINK_RATE_STRESS_SCALE,
)
from perception.calibration import BaselineCalibrator

# --- MediaPipe Face Mesh landmark indices ---
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

# Brow position & furrowing
RIGHT_BROW_CENTER = 105
RIGHT_BROW_EYE = 159
LEFT_BROW_CENTER = 334
LEFT_BROW_EYE = 386
INNER_BROW_LEFT = 55
INNER_BROW_RIGHT = 285

# Mouth, lips, jaw
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
UPPER_LIP_TOP = 0
LOWER_LIP_BOTTOM = 17
CHIN = 152

# Face width reference & head pose
NOSE_TIP = 1
LEFT_FACE_EDGE = 234
RIGHT_FACE_EDGE = 454

# Forehead tension
FOREHEAD_UPPER_L = 10
FOREHEAD_UPPER_R = 338
FOREHEAD_LOWER_L = 108
FOREHEAD_LOWER_R = 337


def _dist2d(p1, p2) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _dist3d(p1, p2) -> float:
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)


def _eye_aspect_ratio(landmarks, eye_indices) -> float:
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]
    vertical = _dist2d(p2, p6) + _dist2d(p3, p5)
    horizontal = 2.0 * _dist2d(p1, p4)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


class FacialFeatureExtractor:
    """Stateful facial feature extractor with baseline calibration, temporal
    EAR blink tracking, facial rigidity, and outlier median filtering."""

    def __init__(self, calibrator: Optional[BaselineCalibrator] = None):
        self.calibrator = calibrator if calibrator is not None else BaselineCalibrator()
        
        # Blink tracking
        self._blink_timestamps: deque = deque()
        self._consecutive_closed_frames = 0
        self._eye_was_closed = False
        
        # Motion & history tracking
        self._nose_history: deque = deque(maxlen=HEAD_MOVEMENT_HISTORY)
        self._prev_landmarks: Optional[List] = None
        self._face_confidence = 0.0

        # Temporal median filter buffers
        self._metric_buffers: Dict[str, deque] = {
            "eye_openness": deque(maxlen=MEDIAN_FILTER_WINDOW),
            "brow_tension": deque(maxlen=MEDIAN_FILTER_WINDOW),
            "mouth_tension": deque(maxlen=MEDIAN_FILTER_WINDOW),
            "head_movement": deque(maxlen=MEDIAN_FILTER_WINDOW),
            "forehead_tension": deque(maxlen=MEDIAN_FILTER_WINDOW),
            "lip_depression": deque(maxlen=MEDIAN_FILTER_WINDOW),
            "facial_rigidity": deque(maxlen=MEDIAN_FILTER_WINDOW),
        }

        # Last known smoothed metrics
        self._last_metrics: Dict[str, float] = {
            "blink_rate": DEFAULT_BLINK_RATE,
            "eye_openness": 0.65,
            "brow_tension": 0.0,
            "mouth_tension": 0.0,
            "head_movement": 0.0,
            "face_confidence": 0.0,
            "forehead_tension": 0.0,
            "lip_depression": 0.0,
            "facial_rigidity": 0.0,
        }

    def update(self, landmarks: Optional[list], now: Optional[float] = None) -> dict:
        now = now if now is not None else time.time()

        if landmarks is None:
            self._face_confidence *= FACE_CONFIDENCE_DECAY
            out = dict(self._last_metrics)
            out["face_confidence"] = round(self._face_confidence, 3)
            out["is_calibrating"] = self.calibrator.is_calibrating(now)
            out["calibration_progress"] = round(self.calibrator.progress(now), 2)
            return out

        self._face_confidence = 1.0

        try:
            face_width = _dist2d(landmarks[LEFT_FACE_EDGE], landmarks[RIGHT_FACE_EDGE]) or 1.0

            # 1. EAR & Eye Openness
            ear_right = _eye_aspect_ratio(landmarks, RIGHT_EYE)
            ear_left = _eye_aspect_ratio(landmarks, LEFT_EYE)
            ear = (ear_right + ear_left) / 2.0
            eye_openness = max(0.0, min(1.0, ear / 0.32))

            # Blink temporal detection
            blink_rate = self._update_blink_temporal(ear, now)

            # 2. Brow Position & Furrowing
            brow_gap_r = _dist2d(landmarks[RIGHT_BROW_CENTER], landmarks[RIGHT_BROW_EYE]) / face_width
            brow_gap_l = _dist2d(landmarks[LEFT_BROW_CENTER], landmarks[LEFT_BROW_EYE]) / face_width
            brow_gap = (brow_gap_r + brow_gap_l) / 2.0
            inner_brow_gap = _dist2d(landmarks[INNER_BROW_LEFT], landmarks[INNER_BROW_RIGHT]) / face_width

            # 3. Mouth, Lips, Jaw
            mouth_width = _dist2d(landmarks[MOUTH_LEFT], landmarks[MOUTH_RIGHT]) / face_width
            lip_gap = _dist2d(landmarks[MOUTH_TOP], landmarks[MOUTH_BOTTOM]) / face_width
            lip_thickness = _dist2d(landmarks[UPPER_LIP_TOP], landmarks[LOWER_LIP_BOTTOM]) / face_width
            jaw_span = _dist2d(landmarks[MOUTH_TOP], landmarks[CHIN]) / face_width

            # 4. Forehead Tension
            fh_l = _dist2d(landmarks[FOREHEAD_UPPER_L], landmarks[FOREHEAD_LOWER_L]) / face_width
            fh_r = _dist2d(landmarks[FOREHEAD_UPPER_R], landmarks[FOREHEAD_LOWER_R]) / face_width
            forehead_gap = (fh_l + fh_r) / 2.0

            # 5. Lip Corner Depression (Frown / Grimace)
            nose_y = landmarks[NOSE_TIP][1]
            lip_center_y = landmarks[UPPER_LIP_TOP][1]
            corner_avg_y = (landmarks[MOUTH_LEFT][1] + landmarks[MOUTH_RIGHT][1]) / 2.0
            lip_corner_droop = (corner_avg_y - lip_center_y) / face_width

            # 6. Head Movement & Motion Velocity (Facial Rigidity)
            nose = landmarks[NOSE_TIP]
            self._nose_history.append(nose)
            head_movement = self._compute_head_movement()

            landmark_velocity = self._compute_landmark_velocity(landmarks, face_width)
            self._prev_landmarks = landmarks

            # Calibration updates
            raw_sample = {
                "blink_rate": blink_rate,
                "eye_openness": eye_openness,
                "brow_gap": brow_gap,
                "inner_brow_gap": inner_brow_gap,
                "lip_gap": lip_gap,
                "lip_thickness": lip_thickness,
                "jaw_span": jaw_span,
                "head_movement": head_movement,
                "forehead_gap": forehead_gap,
                "lip_corner_droop": lip_corner_droop,
            }

            is_calibrating = self.calibrator.is_calibrating(now)
            if is_calibrating:
                self.calibrator.add_facial_sample(raw_sample)
                brow_tension = 0.0
                mouth_tension = 0.0
                forehead_tension = 0.0
                lip_depression = 0.0
                facial_rigidity = 0.0
            else:
                # Compute calibrated relative stress metrics
                base_brow = self.calibrator.get_facial_baseline("brow_gap")
                base_inner_brow = self.calibrator.get_facial_baseline("inner_brow_gap")
                base_lip_thick = self.calibrator.get_facial_baseline("lip_thickness")
                base_jaw = self.calibrator.get_facial_baseline("jaw_span")
                base_forehead = self.calibrator.get_facial_baseline("forehead_gap")
                base_droop = self.calibrator.get_facial_baseline("lip_corner_droop")

                # Brow tension: brow pulled down or furrowed inwards
                brow_pulled_down = max(0.0, (base_brow - brow_gap) / (base_brow * BROW_FURROW_STRESS_SCALE))
                brow_furrowed = max(0.0, (base_inner_brow - inner_brow_gap) / (base_inner_brow * BROW_FURROW_STRESS_SCALE))
                brow_tension = max(0.0, min(1.0, max(brow_pulled_down, brow_furrowed)))

                # Mouth tension: lip compression (thinning lips pressed together) or jaw clenching
                lip_compressed = max(0.0, (base_lip_thick - lip_thickness) / (base_lip_thick * LIP_COMPRESSION_STRESS_SCALE))
                jaw_clenched = max(0.0, (base_jaw - jaw_span) / (base_jaw * JAW_CLENCH_STRESS_SCALE)) if lip_gap < 0.03 else 0.0
                mouth_tension = max(0.0, min(1.0, 0.6 * lip_compressed + 0.4 * jaw_clenched))

                # Forehead tension: compressed forehead gap
                forehead_tension = max(0.0, min(1.0, (base_forehead - forehead_gap) / (base_forehead * 0.35)))

                # Lip corner depression
                lip_depression = max(0.0, min(1.0, (lip_corner_droop - base_droop) / 0.05))

                # Facial Rigidity: when facial velocity is unnaturally low despite elevated tension
                facial_rigidity = max(0.0, min(1.0, (0.05 - landmark_velocity) / 0.05)) if brow_tension > 0.3 or mouth_tension > 0.3 else 0.0

            # Push to median filter buffers to reject outliers
            self._metric_buffers["eye_openness"].append(eye_openness)
            self._metric_buffers["brow_tension"].append(brow_tension)
            self._metric_buffers["mouth_tension"].append(mouth_tension)
            self._metric_buffers["head_movement"].append(head_movement)
            self._metric_buffers["forehead_tension"].append(forehead_tension)
            self._metric_buffers["lip_depression"].append(lip_depression)
            self._metric_buffers["facial_rigidity"].append(facial_rigidity)

            # Compute median values
            med_eye = float(np.median(self._metric_buffers["eye_openness"]))
            med_brow = float(np.median(self._metric_buffers["brow_tension"]))
            med_mouth = float(np.median(self._metric_buffers["mouth_tension"]))
            med_head = float(np.median(self._metric_buffers["head_movement"]))
            med_fh = float(np.median(self._metric_buffers["forehead_tension"]))
            med_droop = float(np.median(self._metric_buffers["lip_depression"]))
            med_rigid = float(np.median(self._metric_buffers["facial_rigidity"]))

            self._last_metrics = {
                "blink_rate": round(blink_rate, 1),
                "eye_openness": round(med_eye, 3),
                "brow_tension": round(med_brow, 3),
                "mouth_tension": round(med_mouth, 3),
                "head_movement": round(med_head, 3),
                "face_confidence": round(self._face_confidence, 3),
                "forehead_tension": round(med_fh, 3),
                "lip_depression": round(med_droop, 3),
                "facial_rigidity": round(med_rigid, 3),
                "is_calibrating": is_calibrating,
                "calibration_progress": round(self.calibrator.progress(now), 2),
            }
            return self._last_metrics

        except (IndexError, ZeroDivisionError, ValueError) as e:
            self._face_confidence *= FACE_CONFIDENCE_DECAY
            out = dict(self._last_metrics)
            out["face_confidence"] = round(self._face_confidence, 3)
            return out

    def _update_blink_temporal(self, ear: float, now: float) -> float:
        is_closed = ear < EAR_BLINK_THRESHOLD
        if is_closed:
            self._consecutive_closed_frames += 1
        else:
            if self._consecutive_closed_frames >= BLINK_MIN_FRAMES_CLOSED:
                # Valid blink re-opened after enough closed frames
                self._blink_timestamps.append(now)
            self._consecutive_closed_frames = 0

        # Purge timestamps outside the rolling window
        cutoff = now - BLINK_RATE_WINDOW_SECONDS
        while self._blink_timestamps and self._blink_timestamps[0] < cutoff:
            self._blink_timestamps.popleft()

        if not self._blink_timestamps:
            return DEFAULT_BLINK_RATE

        elapsed = max(now - self._blink_timestamps[0], 5.0)
        return len(self._blink_timestamps) * (60.0 / elapsed)

    def _compute_head_movement(self) -> float:
        if len(self._nose_history) < 2:
            return 0.0
        deltas = [
            _dist2d(self._nose_history[i], self._nose_history[i - 1])
            for i in range(1, len(self._nose_history))
        ]
        avg_delta = sum(deltas) / len(deltas)
        return max(0.0, min(1.0, avg_delta * 16.0))

    def _compute_landmark_velocity(self, landmarks: list, face_width: float) -> float:
        if self._prev_landmarks is None:
            return 0.0
        # Compute mean 2D displacement of key face center landmarks
        key_indices = [NOSE_TIP, MOUTH_LEFT, MOUTH_RIGHT, RIGHT_BROW_CENTER, LEFT_BROW_CENTER]
        disp = sum(_dist2d(landmarks[i], self._prev_landmarks[i]) for i in key_indices) / (len(key_indices) * face_width)
        return float(disp)
