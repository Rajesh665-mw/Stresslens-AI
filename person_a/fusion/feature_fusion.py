"""
Fuses facial and audio metrics into a calibrated, highly accurate stress reading.

Applies baseline relative deviation scoring, multi-modality conflict resolution,
confidence estimation, temporal exponential moving average (EMA) smoothing,
and factor breakdown logging for transparent debugging.
"""
import time
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

from person_a.config_a import (
    EMA_ALPHA,
    STRESS_LEVEL_LOW_MAX,
    STRESS_LEVEL_MODERATE_MAX,
    FACIAL_WEIGHT,
    AUDIO_WEIGHT,
    BLINK_RATE_STRESS_SCALE,
    PITCH_RISE_STRESS_SCALE,
    PITCH_VAR_STRESS_SCALE,
    ENERGY_RISE_STRESS_SCALE,
    SPEECH_RATE_STRESS_SCALE,
    JITTER_STRESS_RANGE,
    SHIMMER_STRESS_RANGE,
    SPECTRAL_CENTROID_STRESS_SCALE,
)
from person_a.calibration import BaselineCalibrator
from utils.logger import get_logger

logger = get_logger("person_a.fusion")


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _stress_level_for_score(score: Optional[float], confidence: float) -> str:
    if score is None or confidence < 0.10:
        return "Insufficient Data"
    if score < STRESS_LEVEL_LOW_MAX:
        return "Low"
    if score < STRESS_LEVEL_MODERATE_MAX:
        return "Moderate"
    return "High"


class StressFusionEngine:
    """Multimodal fusion engine supporting calibration, modality conflict resolution,
    temporal stability, and factor breakdown debugging."""

    def __init__(self, calibrator: Optional[BaselineCalibrator] = None):
        self.calibrator = calibrator if calibrator is not None else BaselineCalibrator()
        self._smoothed: Optional[dict] = None
        self._prev_fused_stress: Optional[float] = None

    def fuse(
        self,
        facial: Optional[dict],
        audio: Optional[dict],
        camera_active: bool,
        mic_active: bool,
        now: Optional[float] = None,
    ) -> dict:
        now = now if now is not None else time.time()

        # Check calibration status
        is_calibrating = (facial or {}).get("is_calibrating", False) or (audio or {}).get("is_calibrating", False) or self.calibrator.is_calibrating(now)
        cal_prog = round(self.calibrator.progress(now), 2)

        if is_calibrating:
            return self._calibration_response(facial, audio, camera_active, mic_active, cal_prog)

        stress_score, confidence, factor_breakdown = self._score_and_confidence(facial, audio, camera_active, mic_active)

        raw = {
            "stress_score": stress_score,
            "blink_rate": (facial or {}).get("blink_rate"),
            "eye_openness": (facial or {}).get("eye_openness"),
            "brow_tension": (facial or {}).get("brow_tension"),
            "mouth_tension": (facial or {}).get("mouth_tension"),
            "head_movement": (facial or {}).get("head_movement"),
            "face_confidence": (facial or {}).get("face_confidence", 0.0),
            "forehead_tension": (facial or {}).get("forehead_tension"),
            "lip_depression": (facial or {}).get("lip_depression"),
            "facial_rigidity": (facial or {}).get("facial_rigidity"),
            "voice_pitch": (audio or {}).get("voice_pitch"),
            "pitch_variance": (audio or {}).get("pitch_variance"),
            "voice_energy": (audio or {}).get("voice_energy"),
            "speech_rate": (audio or {}).get("speech_rate"),
            "jitter": (audio or {}).get("jitter"),
            "shimmer": (audio or {}).get("shimmer"),
            "spectral_centroid": (audio or {}).get("spectral_centroid"),
            "confidence": confidence,
        }

        smoothed = self._smooth(raw)
        final_score = smoothed["stress_score"]
        final_conf = smoothed["confidence"]

        stress_level = _stress_level_for_score(final_score, final_conf)

        if final_score is not None:
            logger.debug(
                f"Fusion Breakdown -> Final Score: {round(final_score, 1)} ({stress_level}, Conf: {round(final_conf, 2)}) | Factors: {factor_breakdown}"
            )

        return {
            "timestamp": datetime.now().isoformat(),
            "stress_score": round(final_score) if final_score is not None else 0,
            "stress_level": stress_level,
            "blink_rate": smoothed["blink_rate"],
            "eye_openness": smoothed["eye_openness"],
            "mouth_tension": smoothed["mouth_tension"],
            "brow_tension": smoothed["brow_tension"],
            "head_movement": smoothed["head_movement"],
            "face_confidence": smoothed["face_confidence"],
            "forehead_tension": smoothed["forehead_tension"],
            "lip_depression": smoothed["lip_depression"],
            "voice_pitch": smoothed["voice_pitch"],
            "pitch_variance": smoothed["pitch_variance"],
            "voice_energy": smoothed["voice_energy"],
            "speech_rate": smoothed["speech_rate"],
            "jitter": smoothed["jitter"],
            "shimmer": smoothed["shimmer"],
            "spectral_centroid": smoothed["spectral_centroid"],
            "silence_detected": (audio or {}).get("silence_detected", True),
            "audio_level": (audio or {}).get("audio_level", 0.0),
            "camera_active": camera_active,
            "mic_active": mic_active,
            "confidence": round(final_conf, 3),
            "is_calibrating": False,
            "calibration_progress": 1.0,
            "status_message": "Insufficient data for stress estimation" if stress_level == "Insufficient Data" else "Active",
        }

    def _calibration_response(
        self, facial: Optional[dict], audio: Optional[dict], camera_active: bool, mic_active: bool, progress: float
    ) -> dict:
        return {
            "timestamp": datetime.now().isoformat(),
            "stress_score": 15,
            "stress_level": "Calibrating",
            "blink_rate": (facial or {}).get("blink_rate"),
            "eye_openness": (facial or {}).get("eye_openness"),
            "mouth_tension": (facial or {}).get("mouth_tension"),
            "brow_tension": (facial or {}).get("brow_tension"),
            "head_movement": (facial or {}).get("head_movement"),
            "face_confidence": (facial or {}).get("face_confidence", 0.0),
            "forehead_tension": (facial or {}).get("forehead_tension"),
            "lip_depression": (facial or {}).get("lip_depression"),
            "voice_pitch": (audio or {}).get("voice_pitch"),
            "pitch_variance": (audio or {}).get("pitch_variance"),
            "voice_energy": (audio or {}).get("voice_energy"),
            "speech_rate": (audio or {}).get("speech_rate"),
            "jitter": (audio or {}).get("jitter"),
            "shimmer": (audio or {}).get("shimmer"),
            "spectral_centroid": (audio or {}).get("spectral_centroid"),
            "silence_detected": (audio or {}).get("silence_detected", True),
            "audio_level": (audio or {}).get("audio_level", 0.0),
            "camera_active": camera_active,
            "mic_active": mic_active,
            "confidence": 0.50,
            "is_calibrating": True,
            "calibration_progress": progress,
            "status_message": f"Calibrating baseline... ({int(progress * 100)}%)",
        }

    def _score_and_confidence(
        self, facial: Optional[dict], audio: Optional[dict], camera_active: bool, mic_active: bool
    ) -> Tuple[Optional[float], float, Dict[str, float]]:
        
        face_conf = (facial or {}).get("face_confidence", 0.0)
        face_valid = facial is not None and face_conf > 0.15 and camera_active

        silence = (audio or {}).get("silence_detected", True)
        audio_valid = audio is not None and not silence and mic_active
        voice_conf = 0.85 if audio_valid else 0.0

        factor_breakdown: Dict[str, float] = {}

        # NO VALID INPUT HANDLER
        if not face_valid and not audio_valid:
            return None, 0.0, {"no_data": 0.0}

        contributions = []

        # 1. Facial Stress Analysis
        if face_valid:
            blink_rate = facial.get("blink_rate", 17.0)
            base_blink = self.calibrator.get_facial_baseline("blink_rate")
            blink_dev = max(0.0, (blink_rate - base_blink) / (base_blink * BLINK_RATE_STRESS_SCALE))
            blink_stress = _clamp01(blink_dev)

            brow_tension = _clamp01(facial.get("brow_tension", 0.0))
            mouth_tension = _clamp01(facial.get("mouth_tension", 0.0))
            forehead_tension = _clamp01(facial.get("forehead_tension", 0.0))
            lip_depression = _clamp01(facial.get("lip_depression", 0.0))
            facial_rigidity = _clamp01(facial.get("facial_rigidity", 0.0))
            head_movement = _clamp01(facial.get("head_movement", 0.0))

            # Weighted combination: brow, mouth, and forehead are dominant markers
            facial_stress = (
                brow_tension * 0.28 +
                mouth_tension * 0.24 +
                forehead_tension * 0.18 +
                blink_stress * 0.10 +
                lip_depression * 0.10 +
                facial_rigidity * 0.05 +
                head_movement * 0.05
            )

            factor_breakdown["Brow_Tension"] = round(brow_tension * 28, 1)
            factor_breakdown["Mouth_Tension"] = round(mouth_tension * 24, 1)
            factor_breakdown["Forehead_Tension"] = round(forehead_tension * 18, 1)
            factor_breakdown["Blink_Rate"] = round(blink_stress * 10, 1)
            factor_breakdown["Lip_Depression"] = round(lip_depression * 10, 1)

            contributions.append(("facial", facial_stress, FACIAL_WEIGHT * face_conf))

        # 2. Audio Stress Analysis
        if audio_valid:
            pitch = audio.get("voice_pitch")
            base_pitch = self.calibrator.get_audio_baseline("voice_pitch")
            pitch_dev = max(0.0, (pitch - base_pitch) / (base_pitch * PITCH_RISE_STRESS_SCALE)) if pitch else 0.0
            pitch_stress = _clamp01(pitch_dev)

            variance = audio.get("pitch_variance")
            base_var = self.calibrator.get_audio_baseline("pitch_variance")
            var_dev = max(0.0, (variance - base_var) / (max(base_var, 5.0) * PITCH_VAR_STRESS_SCALE)) if variance else 0.0
            var_stress = _clamp01(var_dev)

            energy = audio.get("voice_energy", 0.0)
            base_energy = self.calibrator.get_audio_baseline("voice_energy")
            energy_dev = max(0.0, (energy - base_energy) / (max(base_energy, 0.1) * ENERGY_RISE_STRESS_SCALE))
            energy_stress = _clamp01(energy_dev)

            rate = audio.get("speech_rate")
            base_rate = self.calibrator.get_audio_baseline("speech_rate")
            rate_dev = max(0.0, (rate - base_rate) / (max(base_rate, 40.0) * SPEECH_RATE_STRESS_SCALE)) if rate else 0.0
            rate_stress = _clamp01(rate_dev)

            jitter = audio.get("jitter")
            jitter_stress = _clamp01(jitter / JITTER_STRESS_RANGE) if jitter else 0.0

            shimmer = audio.get("shimmer")
            shimmer_stress = _clamp01(shimmer / SHIMMER_STRESS_RANGE) if shimmer else 0.0

            centroid = audio.get("spectral_centroid")
            base_cent = self.calibrator.get_audio_baseline("spectral_centroid")
            cent_dev = max(0.0, (centroid - base_cent) / (base_cent * SPECTRAL_CENTROID_STRESS_SCALE)) if centroid else 0.0
            cent_stress = _clamp01(cent_dev)

            # Audio stress requires multiple signals (never pitch alone!)
            audio_stress = (
                pitch_stress * 0.18 +
                energy_stress * 0.16 +
                rate_stress * 0.16 +
                var_stress * 0.16 +
                jitter_stress * 0.14 +
                shimmer_stress * 0.10 +
                cent_stress * 0.10
            )

            factor_breakdown["Pitch_Rise"] = round(pitch_stress * 18, 1)
            factor_breakdown["Voice_Energy"] = round(energy_stress * 16, 1)
            factor_breakdown["Speech_Rate"] = round(rate_stress * 16, 1)
            factor_breakdown["Jitter_Shimmer"] = round((jitter_stress + shimmer_stress) * 12, 1)

            contributions.append(("audio", audio_stress, AUDIO_WEIGHT))

        # 3. Multimodal Fusion & Conflict Resolution
        if len(contributions) == 1:
            mod_name, stress_val, weight = contributions[0]
            stress_score = stress_val * 100.0
            confidence = 0.82 if mod_name == "facial" else 0.75
            return stress_score, confidence, factor_breakdown

        # Both face and voice are active
        _, f_stress, f_weight = contributions[0]
        _, a_stress, a_weight = contributions[1]

        total_weight = f_weight + a_weight
        weighted_stress = (f_stress * f_weight + a_stress * a_weight) / total_weight
        stress_score = weighted_stress * 100.0

        # Conflict resolution check
        stress_diff = abs(f_stress - a_stress)
        if stress_diff > 0.35:
            # Modality conflict: face & voice disagree (e.g. calm face, stressed voice)
            # Penalize confidence and bound score toward Moderate (35-65)
            confidence = 0.55
            stress_score = 35.0 + (weighted_stress * 30.0)
            factor_breakdown["Modality_Conflict_Penalty"] = round(-stress_diff * 15, 1)
        else:
            # Modalities agree! High confidence bonus.
            confidence = min(0.96, (total_weight / (FACIAL_WEIGHT + AUDIO_WEIGHT)) + 0.15)

        return stress_score, confidence, factor_breakdown

    def _smooth(self, raw: dict) -> dict:
        if self._smoothed is None:
            self._smoothed = dict(raw)
            return self._smoothed

        out = {}
        for key, new_val in raw.items():
            prev_val = self._smoothed.get(key)
            if new_val is None:
                out[key] = prev_val
            elif prev_val is None:
                out[key] = new_val
            elif isinstance(new_val, (int, float)):
                out[key] = EMA_ALPHA * float(new_val) + (1.0 - EMA_ALPHA) * float(prev_val)
            else:
                out[key] = new_val
        self._smoothed = out
        return out
