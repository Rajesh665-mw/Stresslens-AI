"""
Data models shared across the backend and UI.

FeatureData mirrors the JSON contract produced by Person A's
webcam/microphone pipeline. SLMResponse mirrors the strict JSON
contract we require from the local model.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class FeatureData:
    timestamp: str
    stress_score: float
    stress_level: str
    blink_rate: Optional[float] = None
    eye_openness: Optional[float] = None
    mouth_tension: Optional[float] = None   # jaw/mouth tension proxy
    head_movement: Optional[float] = None
    voice_pitch: Optional[float] = None
    voice_energy: Optional[float] = None
    speech_rate: Optional[float] = None
    confidence: Optional[float] = None
    # --- Added for the Person A live-capture pipeline. All optional with
    # default None so the original simulator and any older recorded JSON
    # keep working without changes (backward compatible). ---
    brow_tension: Optional[float] = None      # facial brow tension proxy
    face_confidence: Optional[float] = None   # face-detection quality, separate from overall `confidence`
    pitch_variance: Optional[float] = None    # rolling variance of voice pitch
    silence_detected: Optional[bool] = None   # True if the current audio window is silence
    camera_active: Optional[bool] = None      # True ONLY when webcam hardware is open and producing frames
    mic_active: Optional[bool] = None         # True ONLY when microphone hardware stream is open
    audio_level: Optional[float] = None       # Live audio RMS level (0.0 to 1.0)
    # --- New micro-expression and voice quality metrics ---
    forehead_tension: Optional[float] = None  # forehead wrinkle/compression proxy (0-1)
    lip_depression: Optional[float] = None    # lip corner depression / frown proxy (0-1)
    jitter: Optional[float] = None            # cycle-to-cycle pitch variation (voice stress marker)
    shimmer: Optional[float] = None           # amplitude variation across pitch cycles (voice stress)
    spectral_centroid: Optional[float] = None # spectral brightness of voice (Hz, rises under stress)

    @staticmethod
    def from_dict(d: dict) -> "FeatureData":
        """Build a FeatureData object defensively -- missing keys become None
        rather than raising, so a partial reading from Person A never crashes
        the pipeline. Unknown/extra keys in `d` are simply ignored."""
        return FeatureData(
            timestamp=d.get("timestamp", datetime.now().isoformat()),
            stress_score=float(d.get("stress_score", 0)),
            stress_level=d.get("stress_level", "Unknown"),
            blink_rate=d.get("blink_rate"),
            eye_openness=d.get("eye_openness"),
            mouth_tension=d.get("mouth_tension"),
            head_movement=d.get("head_movement"),
            voice_pitch=d.get("voice_pitch"),
            voice_energy=d.get("voice_energy"),
            speech_rate=d.get("speech_rate"),
            confidence=d.get("confidence"),
            brow_tension=d.get("brow_tension"),
            face_confidence=d.get("face_confidence"),
            pitch_variance=d.get("pitch_variance"),
            silence_detected=d.get("silence_detected"),
            camera_active=d.get("camera_active"),
            mic_active=d.get("mic_active"),
            audio_level=d.get("audio_level"),
            forehead_tension=d.get("forehead_tension"),
            lip_depression=d.get("lip_depression"),
            jitter=d.get("jitter"),
            shimmer=d.get("shimmer"),
            spectral_centroid=d.get("spectral_centroid"),
        )


@dataclass
class SLMResponse:
    stress_level: str
    reasoning: str
    intervention: str
    confidence: str  # "Low" / "Medium" / "High" -- kept as text per spec
    is_fallback: bool = False   # True when the model output could not be parsed
    raw_text: str = ""          # kept for optional debugging in the UI

    @staticmethod
    def fallback(reason: str, raw_text: str = "") -> "SLMResponse":
        """Used whenever the model's output is missing, malformed, or Ollama
        is unreachable -- keeps the UI alive instead of crashing."""
        return SLMResponse(
            stress_level="Unknown",
            reasoning=f"Could not get a reliable interpretation ({reason}).",
            intervention="Take a slow breath and check back in a moment.",
            confidence="Low",
            is_fallback=True,
            raw_text=raw_text,
        )
