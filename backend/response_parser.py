"""
Parses and validates the SLM's raw text response into a strict
SLMResponse object. Ollama's "format": "json" mode makes this mostly
straightforward, but models occasionally wrap output in markdown
fences or omit a key -- this module is the single place that
tolerates that mess and guarantees a safe, usable object comes out.
"""
import json
import re
from typing import Optional

from models.schemas import SLMResponse
from utils.logger import get_logger

logger = get_logger("response_parser")

REQUIRED_KEYS = {"stress_level", "reasoning", "intervention", "confidence"}
VALID_LEVELS = {"Low", "Moderate", "High", "Unknown"}
VALID_CONFIDENCE = {"Low", "Medium", "High"}


def _extract_json_block(text: str) -> str:
    """Strips markdown code fences and grabs the outermost {...} block,
    in case the model added stray text around the JSON."""
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text


def parse_slm_response(raw_text: Optional[str]) -> SLMResponse:
    """Converts raw model output into a validated SLMResponse.
    Never raises -- always returns something the UI can render safely."""
    if not raw_text:
        return SLMResponse.fallback("empty response from model")

    json_str = _extract_json_block(raw_text)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        logger.warning(f"Malformed JSON from model: {raw_text[:200]!r}")
        return SLMResponse.fallback("malformed JSON", raw_text=raw_text)

    if not REQUIRED_KEYS.issubset(data.keys()):
        missing = REQUIRED_KEYS - data.keys()
        logger.warning(f"Model response missing keys: {missing}")
        return SLMResponse.fallback(f"missing keys: {missing}", raw_text=raw_text)

    stress_level = str(data.get("stress_level", "Unknown")).strip().title()
    confidence = str(data.get("confidence", "Low")).strip().title()

    if stress_level not in VALID_LEVELS:
        stress_level = "Unknown"
    if confidence not in VALID_CONFIDENCE:
        confidence = "Low"

    return SLMResponse(
        stress_level=stress_level,
        reasoning=str(data.get("reasoning", "")).strip() or "No reasoning provided.",
        intervention=str(data.get("intervention", "")).strip() or "Take a short mindful break.",
        confidence=confidence,
        is_fallback=False,
        raw_text=raw_text,
    )
