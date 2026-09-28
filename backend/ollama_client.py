"""
Thin wrapper around the local Ollama HTTP API.
Keeps all networking + model-selection logic in one place so the rest
of the backend never talks to Ollama directly.
"""
import json
from typing import Optional

import requests

from utils.config import (
    OLLAMA_HOST,
    PREFERRED_MODELS,
    OLLAMA_REQUEST_TIMEOUT,
    OLLAMA_TEMPERATURE,
)
from utils.logger import get_logger

logger = get_logger("ollama_client")


class OllamaClient:
    def __init__(self, host: str = OLLAMA_HOST):
        self.host = host.rstrip("/")
        self._model: Optional[str] = None

    def get_installed_models(self) -> list:
        """Returns model names currently pulled in the local Ollama instance."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=OLLAMA_REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except requests.RequestException as e:
            logger.error(f"Could not reach Ollama at {self.host}: {e}")
            return []

    def resolve_model(self) -> Optional[str]:
        """Picks the first preferred model that is actually installed.
        Caches the result so we don't hit /api/tags on every request."""
        if self._model:
            return self._model

        installed = self.get_installed_models()
        if not installed:
            return None

        for candidate in PREFERRED_MODELS:
            for name in installed:
                if name == candidate or name.startswith(candidate.split(":")[0]):
                    self._model = name
                    logger.info(f"Using local model: {name}")
                    return name

        # Nothing on our preference list is installed -- fall back to whatever exists.
        self._model = installed[0]
        logger.warning(f"No preferred model found; falling back to '{installed[0]}'.")
        return self._model

    def is_available(self) -> bool:
        return self.resolve_model() is not None

    def generate(self, prompt: str) -> Optional[str]:
        """Sends the prompt to the local model and returns the raw text response.
        Returns None on any failure -- callers must handle that gracefully."""
        model = self.resolve_model()
        if not model:
            logger.error("No local Ollama model available.")
            return None

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",  # ask Ollama to constrain output to valid JSON
            "options": {"temperature": OLLAMA_TEMPERATURE},
        }

        try:
            resp = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=OLLAMA_REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")
        except requests.RequestException as e:
            logger.error(f"Ollama request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Ollama returned a non-JSON HTTP body: {e}")
            return None
