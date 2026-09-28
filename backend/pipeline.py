"""
Orchestrates the end-to-end flow:
Person A's JSON -> Ollama -> parsed SLMResponse -> DataStore.

Runs as a background thread so the Streamlit UI stays responsive while
this polls for new readings.
"""
import threading
import time
from typing import Optional

from backend.data_store import store
from backend.ollama_client import OllamaClient
from backend.response_parser import parse_slm_response
from backend.watcher import FeatureWatcher
from models.schemas import FeatureData, SLMResponse
from prompts.stress_prompt import build_prompt
from utils.config import POLL_INTERVAL_SECONDS
from utils.logger import get_logger

logger = get_logger("pipeline")

_client = OllamaClient()
_watcher = FeatureWatcher()


def run_inference(features: FeatureData) -> SLMResponse:
    """Runs one SLM inference for a given reading and writes the result
    to the shared DataStore. Used both by the polling loop and by the
    dashboard's 'Generate New Intervention' button."""
    prompt = build_prompt(features)
    raw = _client.generate(prompt)
    response = parse_slm_response(raw)
    store.update(features, response)
    return response


def get_latest_features() -> Optional[FeatureData]:
    return _watcher.last_known()


_is_inferring = False


def _async_inference(features: FeatureData):
    global _is_inferring
    try:
        run_inference(features)
    finally:
        _is_inferring = False


def _poll_loop(stop_event: threading.Event):
    global _is_inferring
    logger.info("Pipeline started -- watching for features from Person A's pipeline.")
    while not stop_event.is_set():
        try:
            if _watcher.has_new_data():
                features = _watcher.read_latest()
                if features:
                    store.update_features(features)
                    if not _is_inferring:
                        _is_inferring = True
                        threading.Thread(target=_async_inference, args=(features,), daemon=True).start()
        except Exception as e:  # pipeline must never die on an unexpected error
            logger.error(f"Unexpected error in pipeline loop: {e}")
        time.sleep(POLL_INTERVAL_SECONDS)
    logger.info("Pipeline stopped.")


_stop_event = threading.Event()
_thread: Optional[threading.Thread] = None


def start_background_pipeline():
    """Idempotent: safe to call on every Streamlit rerun -- only ever
    starts one background thread per process."""
    global _thread
    if _thread is None or not _thread.is_alive():
        _stop_event.clear()
        _thread = threading.Thread(target=_poll_loop, args=(_stop_event,), daemon=True)
        _thread.start()
