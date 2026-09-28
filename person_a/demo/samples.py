"""
Predefined stress profiles for demo mode -- used when there's no camera
or microphone available (or you just want a reliable, repeatable demo).

Each profile is a function that returns one full feature reading, given
elapsed time `t` (seconds since the demo started), so values drift
smoothly and slightly randomly rather than looking robotic.
"""
import random
from datetime import datetime


def _jitter(value: float, amount: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value + random.uniform(-amount, amount)))


def _reading(
    stress_score, blink_rate, eye_openness, brow_tension, mouth_tension,
    head_movement, face_confidence, voice_pitch, pitch_variance,
    voice_energy, speech_rate, silence_detected, confidence,
) -> dict:
    stress_level = "Low" if stress_score < 40 else ("Moderate" if stress_score < 70 else "High")
    return {
        "timestamp": datetime.now().isoformat(),
        "stress_score": round(stress_score),
        "stress_level": stress_level,
        "blink_rate": round(blink_rate, 1),
        "eye_openness": round(eye_openness, 3),
        "brow_tension": round(brow_tension, 3),
        "mouth_tension": round(mouth_tension, 3),
        "head_movement": round(head_movement, 3),
        "face_confidence": round(face_confidence, 3),
        "voice_pitch": round(voice_pitch, 1),
        "pitch_variance": round(pitch_variance, 2),
        "voice_energy": round(voice_energy, 3),
        "speech_rate": round(speech_rate, 1),
        "silence_detected": silence_detected,
        "confidence": round(confidence, 3),
    }


def calm_sample(t: float) -> dict:
    return _reading(
        stress_score=_jitter(20, 6, 5, 38),
        blink_rate=_jitter(14, 2, 8, 20),
        eye_openness=_jitter(0.9, 0.05, 0.75, 1.0),
        brow_tension=_jitter(0.12, 0.05, 0.0, 0.3),
        mouth_tension=_jitter(0.15, 0.05, 0.0, 0.3),
        head_movement=_jitter(0.08, 0.04, 0.0, 0.2),
        face_confidence=_jitter(0.95, 0.03, 0.85, 1.0),
        voice_pitch=_jitter(160, 8, 130, 190),
        pitch_variance=_jitter(8, 3, 2, 15),
        voice_energy=_jitter(0.25, 0.08, 0.05, 0.45),
        speech_rate=_jitter(105, 8, 85, 125),
        silence_detected=random.random() < 0.2,
        confidence=_jitter(0.93, 0.03, 0.85, 0.98),
    )


def medium_sample(t: float) -> dict:
    return _reading(
        stress_score=_jitter(52, 8, 40, 68),
        blink_rate=_jitter(19, 3, 14, 26),
        eye_openness=_jitter(0.75, 0.07, 0.55, 0.9),
        brow_tension=_jitter(0.42, 0.08, 0.25, 0.6),
        mouth_tension=_jitter(0.45, 0.08, 0.25, 0.6),
        head_movement=_jitter(0.28, 0.08, 0.1, 0.45),
        face_confidence=_jitter(0.88, 0.05, 0.7, 0.97),
        voice_pitch=_jitter(200, 12, 170, 230),
        pitch_variance=_jitter(20, 6, 10, 32),
        voice_energy=_jitter(0.5, 0.1, 0.3, 0.7),
        speech_rate=_jitter(125, 10, 105, 145),
        silence_detected=random.random() < 0.1,
        confidence=_jitter(0.8, 0.06, 0.6, 0.92),
    )


def high_sample(t: float) -> dict:
    return _reading(
        stress_score=_jitter(82, 8, 70, 96),
        blink_rate=_jitter(27, 4, 20, 35),
        eye_openness=_jitter(0.6, 0.08, 0.4, 0.75),
        brow_tension=_jitter(0.75, 0.1, 0.55, 0.92),
        mouth_tension=_jitter(0.72, 0.1, 0.55, 0.9),
        head_movement=_jitter(0.5, 0.12, 0.3, 0.75),
        face_confidence=_jitter(0.9, 0.05, 0.75, 0.98),
        voice_pitch=_jitter(240, 15, 210, 280),
        pitch_variance=_jitter(38, 8, 25, 55),
        voice_energy=_jitter(0.78, 0.1, 0.55, 0.95),
        speech_rate=_jitter(148, 10, 128, 168),
        silence_detected=random.random() < 0.05,
        confidence=_jitter(0.9, 0.05, 0.75, 0.98),
    )


PROFILES = {
    "calm": calm_sample,
    "medium": medium_sample,
    "high": high_sample,
}
