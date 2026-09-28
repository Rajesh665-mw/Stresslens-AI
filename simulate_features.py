"""
Standalone simulator that mimics the perception pipeline output.
Useful for developing and demoing the backend and UI stack before the real
webcam/microphone pipeline is wired in.

Run with (from the project root):
    python simulate_features.py
"""
import json
import os
import random
import time
from datetime import datetime

from utils.config import FEATURES_FILE


def random_reading() -> dict:
    stress_score = random.randint(10, 95)
    if stress_score < 40:
        level = "Low"
    elif stress_score < 70:
        level = "Moderate"
    else:
        level = "High"

    return {
        "timestamp": datetime.now().isoformat(),
        "stress_score": stress_score,
        "stress_level": level,
        "blink_rate": round(random.uniform(10, 30), 1),
        "eye_openness": round(random.uniform(0.4, 1.0), 2),
        "mouth_tension": round(random.uniform(0.1, 0.9), 2),
        "head_movement": round(random.uniform(0.0, 0.5), 2),
        "voice_pitch": round(random.uniform(150, 260), 1),
        "voice_energy": round(random.uniform(0.3, 0.9), 2),
        "speech_rate": round(random.uniform(100, 160), 1),
        "confidence": round(random.uniform(0.5, 0.98), 2),
    }


def main():
    os.makedirs(os.path.dirname(FEATURES_FILE), exist_ok=True)
    print(f"Simulating perception feature pipeline -> {FEATURES_FILE}")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            reading = random_reading()
            with open(FEATURES_FILE, "w") as f:
                json.dump(reading, f, indent=2)
            print(f"Wrote reading: score={reading['stress_score']} level={reading['stress_level']}")
            time.sleep(15)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
