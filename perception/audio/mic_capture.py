"""
Thin wrapper around sounddevice for microphone capture. The import is
guarded against both ImportError (package not installed) and OSError
(package installed but the native PortAudio library is missing) --
both are common on fresh machines and neither should crash the app.

Auto-detects the best working input device if the OS default
captures near-zero audio (e.g. headset mic selected but not plugged in).

Includes automatic gain normalization and overflow protection so quiet
microphones produce usable signal levels without manual tuning.
"""
import threading
import time
from collections import deque
from typing import Optional

import numpy as np

from perception.config import AUDIO_SAMPLE_RATE, AUDIO_BLOCK_SECONDS, AUDIO_BUFFER_SECONDS
from utils.logger import get_logger

logger = get_logger("perception.mic_capture")

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError) as e:
    SOUNDDEVICE_AVAILABLE = False
    _import_error = e


class MicrophoneUnavailableError(RuntimeError):
    """Raised when the microphone can't be opened or sounddevice isn't usable."""


def _probe_device(device_index: int, sample_rate: int, duration: float = 0.2) -> float:
    """Records a short burst from a device and returns its RMS.
    Returns 0.0 on any failure, including inf/nan results (loopback artifacts)."""
    try:
        samples = sd.rec(
            int(sample_rate * duration),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            device=device_index,
        )
        sd.wait()
        rms = float(np.sqrt(np.mean(np.square(samples))))
        # Reject inf/nan (can happen with loopback/virtual devices)
        if not np.isfinite(rms):
            return 0.0
        return rms
    except Exception:
        return 0.0


def _is_real_mic(name: str) -> bool:
    """Heuristic: true for devices that are likely actual microphones,
    false for speakers/loopback/virtual devices."""
    lower = name.lower()
    # Negative signals -- these are loopback/output devices masquerading as inputs
    if any(kw in lower for kw in ("speaker", "pc speaker", "stereo mix", "loopback",
                                   "headphone", "output", "sound mapper")):
        return False
    # Positive signals
    if any(kw in lower for kw in ("microphone", "mic array", "mic input", "hands-free")):
        return True
    return True


def _find_best_input_device(sample_rate: int) -> Optional[int]:
    """Probes available input devices and picks the best working microphone.
    Strictly avoids loopback/speaker devices."""
    try:
        default_input = sd.default.device[0]
        devices = sd.query_devices()
    except Exception:
        return None

    # Collect real mic candidates
    mic_candidates = []
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0 and _is_real_mic(d["name"]):
            mic_candidates.append(i)

    if not mic_candidates:
        return default_input if default_input >= 0 else None

    # If default input is a real mic, use it or probe mic candidates
    best_idx = mic_candidates[0]
    best_rms = -1.0

    for idx in mic_candidates:
        rms = _probe_device(idx, sample_rate)
        name = devices[idx]["name"]
        logger.info(f"  Mic candidate [{idx}] '{name}' probe RMS={rms:.6f}")
        # Prefer Intel Smart Sound / internal mic array or whatever has highest non-zero RMS
        if rms > best_rms:
            best_rms = rms
            best_idx = idx

    selected_name = devices[best_idx]["name"]
    logger.info(f"Selected real mic device [{best_idx}] '{selected_name}' (RMS={best_rms:.6f})")
    return best_idx




class MicCapture:
    """Continuously records audio into a rolling ring buffer via a
    background callback stream. `read_recent(seconds)` gives the caller
    a thread-safe snapshot of the most recent audio for feature extraction.

    Includes automatic gain normalization: tracks the rolling peak amplitude
    and scales incoming samples so quiet microphones produce usable levels.
    Also clamps samples to ±1.0 to prevent distortion from USB mic overflow."""

    def __init__(self, sample_rate: int = AUDIO_SAMPLE_RATE):
        if not SOUNDDEVICE_AVAILABLE:
            raise MicrophoneUnavailableError(
                f"sounddevice/PortAudio is not available ({_import_error}). "
                "Run: pip install -r requirements-perception.txt and ensure "
                "PortAudio is installed on your system."
            )
        self.sample_rate = sample_rate
        self._lock = threading.Lock()
        max_samples = int(sample_rate * AUDIO_BUFFER_SECONDS)
        self._buffer: deque = deque(maxlen=max_samples)
        self._stream: Optional["sd.InputStream"] = None
        self._available = False
        self.device_name: str = "Unknown"
        # Automatic gain normalization state
        self._peak_tracker = 0.01  # rolling peak amplitude (start small, adapt up)
        self._gain_decay = 0.9995  # slow decay so peak doesn't drop too fast

    def _callback(self, indata, frames, time_info, status):
        if status:
            logger.warning(f"Audio stream status: {status}")

        samples = indata[:, 0].copy()

        # Overflow protection: clamp to ±1.0
        np.clip(samples, -1.0, 1.0, out=samples)

        # Auto-gain normalization: track peak and scale up quiet mics
        peak = float(np.max(np.abs(samples)))
        if peak > self._peak_tracker:
            self._peak_tracker = peak  # instant rise
        else:
            self._peak_tracker *= self._gain_decay  # slow decay
            self._peak_tracker = max(self._peak_tracker, 0.001)  # floor

        # Scale so that the rolling peak maps to ~0.7 amplitude
        gain = min(0.7 / self._peak_tracker, 10.0)  # cap gain at 10x to prevent noise explosion
        samples = samples * gain
        np.clip(samples, -1.0, 1.0, out=samples)

        with self._lock:
            self._buffer.extend(samples.tolist())

    def start(self) -> bool:
        """Opens the input stream. Auto-detects the best working mic.
        Returns False (without raising) if no microphone is available."""
        try:
            device_idx = _find_best_input_device(self.sample_rate)
            block_size = max(1, int(self.sample_rate * AUDIO_BLOCK_SECONDS))
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                blocksize=block_size,
                dtype="float32",
                callback=self._callback,
                device=device_idx,
            )
            self._stream.start()
            self._available = True
            # Store the resolved device name
            try:
                info = sd.query_devices(device_idx)
                self.device_name = info["name"]
            except Exception:
                self.device_name = f"Device {device_idx}"
            logger.info(f"Microphone opened: [{device_idx}] {self.device_name}")
            return True
        except Exception as e:
            logger.warning(f"Could not open microphone: {e}")
            self._stream = None
            self._available = False
            return False

    def is_available(self) -> bool:
        return self._available

    def read_recent(self, seconds: float) -> np.ndarray:
        """Returns up to the last `seconds` of audio as a float32 numpy array.
        Returns an empty array if nothing has been captured yet."""
        with self._lock:
            n = int(self.sample_rate * seconds)
            data = list(self._buffer)[-n:] if n > 0 else list(self._buffer)
        return np.array(data, dtype=np.float32)

    def stop(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.warning(f"Error closing audio stream: {e}")
            self._stream = None
            self._available = False
