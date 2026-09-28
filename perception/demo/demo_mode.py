"""
Demo mode: replays a canned stress profile (calm / medium / high) into
the shared features.json on a fixed interval, without needing a webcam
or microphone at all. Used for reliable demos and for
testing the backend/UI stack independently of live hardware.
"""
import time

from perception.demo.samples import PROFILES
from perception.feature_writer import write_features
from utils.logger import get_logger

logger = get_logger("perception.demo_mode")


def run_demo(profile: str = "calm", interval_seconds: float = 2.0):
    """Runs forever (until Ctrl+C), writing one reading from the chosen
    profile every `interval_seconds`."""
    if profile not in PROFILES:
        valid = ", ".join(PROFILES.keys())
        raise ValueError(f"Unknown demo profile '{profile}'. Valid options: {valid}")

    sample_fn = PROFILES[profile]
    logger.info(f"Demo mode running with profile='{profile}' (writing every {interval_seconds}s). Ctrl+C to stop.")

    start = time.time()
    try:
        while True:
            t = time.time() - start
            reading = sample_fn(t)
            if write_features(reading):
                logger.info(
                    f"[{profile}] wrote stress_score={reading['stress_score']} "
                    f"level={reading['stress_level']} confidence={reading['confidence']}"
                )
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        logger.info("Demo mode stopped.")
