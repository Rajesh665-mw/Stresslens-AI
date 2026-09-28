"""
Prompt template for the local SLM.

Design goals:
- Never invent a medical diagnosis.
- Only interpret, explain, and suggest -- grounded strictly in the
  supplied features.
- Always return STRICT JSON matching a fixed schema so the backend
  can parse it reliably.
"""
from models.schemas import FeatureData

SYSTEM_INSTRUCTIONS = """You are a wellness assistant that interprets physiological stress signals.
You are NOT a doctor and must NEVER provide a medical diagnosis, disease name, or clinical claim.

Rules:
1. Only use the numeric/categorical features provided below. Do not invent data.
2. Interpret the overall stress level in plain language.
3. Give a short, plausible explanation referencing the specific features that stand out.
4. Recommend one short, practical wellness action (breathing, posture, hydration, a short break, etc).
5. If the "confidence" value provided is low (below 0.5), explicitly say the reading is uncertain.
6. Respond with STRICT JSON only. No markdown, no code fences, no extra commentary, no text before or after the JSON.

Return JSON with EXACTLY these keys:
{
  "stress_level": "Low" | "Moderate" | "High",
  "reasoning": "one or two sentences explaining likely reasons based only on the given features",
  "intervention": "one short, actionable wellness suggestion",
  "confidence": "Low" | "Medium" | "High"
}
"""

EXAMPLE = """Example input features:
stress_score: 74, stress_level: High, blink_rate: 23, eye_openness: 0.63, mouth_tension: 0.71,
head_movement: 0.24, voice_pitch: 228, voice_energy: 0.77, speech_rate: 138, confidence: 0.91

Example output:
{"stress_level":"High","reasoning":"Elevated voice pitch, faster speech and increased facial tension suggest increased stress.","intervention":"Take a short break, drink water, and practice slow breathing for two minutes.","confidence":"High"}
"""


def _fmt(value) -> str:
    """Renders a feature value for the prompt, or 'N/A' when Person A's
    pipeline didn't supply it (e.g. an older client, or a modality that's
    currently unavailable such as a muted mic)."""
    return "N/A" if value is None else str(value)


def build_prompt(features: FeatureData) -> str:
    """Compose the full prompt sent to the local model for one reading."""
    feature_block = (
        f"timestamp: {features.timestamp}\n"
        f"stress_score: {features.stress_score}\n"
        f"stress_level: {features.stress_level}\n"
        f"blink_rate: {_fmt(features.blink_rate)}\n"
        f"eye_openness: {_fmt(features.eye_openness)}\n"
        f"mouth_tension (jaw/mouth tension proxy): {_fmt(features.mouth_tension)}\n"
        f"brow_tension (brow tension proxy): {_fmt(features.brow_tension)}\n"
        f"head_movement: {_fmt(features.head_movement)}\n"
        f"face_confidence (face detection quality): {_fmt(features.face_confidence)}\n"
        f"voice_pitch: {_fmt(features.voice_pitch)}\n"
        f"pitch_variance: {_fmt(features.pitch_variance)}\n"
        f"voice_energy: {_fmt(features.voice_energy)}\n"
        f"speech_rate: {_fmt(features.speech_rate)}\n"
        f"silence_detected: {_fmt(features.silence_detected)}\n"
        f"confidence: {_fmt(features.confidence)}\n"
    )

    return (
        f"{SYSTEM_INSTRUCTIONS}\n"
        f"{EXAMPLE}\n"
        f"Now interpret this new reading:\n{feature_block}\n"
        f"Respond with JSON only:"
    )
