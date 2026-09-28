"""
Calibrated, multi-feature audio extraction with robust Voice Activity Detection (VAD).

Combines RMS energy, Zero Crossing Rate (ZCR), and spectral centroid to ensure
silence and ambient room noise are never analyzed. Extracts Pitch (F0), Pitch Variance,
RMS Energy, Speech Rate, Jitter, Shimmer, Spectral Centroid, and Harmonic-to-Noise Ratio (HNR).
"""
import time
from collections import deque
from typing import Optional, Dict, Any

import numpy as np

from person_a.config_a import (
    AUDIO_SAMPLE_RATE,
    SILENCE_RMS_THRESHOLD,
    VAD_ZCR_MAX_SILENCE,
    PITCH_MIN_HZ,
    PITCH_MAX_HZ,
    PITCH_HISTORY_SECONDS,
    SPEECH_RATE_WINDOW_SECONDS,
    MEDIAN_FILTER_WINDOW,
)
from person_a.calibration import BaselineCalibrator


def _rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples))))


def _zero_crossing_rate(samples: np.ndarray) -> float:
    if samples.size < 2:
        return 0.0
    zero_crossings = np.nonzero(np.diff(samples > 0))[0]
    return float(len(zero_crossings) / float(len(samples)))


def _parabolic_interpolation(corr: np.ndarray, peak_idx: int) -> float:
    if peak_idx <= 0 or peak_idx >= len(corr) - 1:
        return float(peak_idx)
    a = corr[peak_idx - 1]
    b = corr[peak_idx]
    c = corr[peak_idx + 1]
    denom = 2.0 * (2.0 * b - a - c)
    if abs(denom) < 1e-12:
        return float(peak_idx)
    offset = (a - c) / denom
    return peak_idx + offset


def estimate_pitch(samples: np.ndarray, sample_rate: int = AUDIO_SAMPLE_RATE) -> Optional[float]:
    if samples.size < sample_rate // 20:  # need at least 50ms
        return None
    if _rms(samples) < SILENCE_RMS_THRESHOLD:
        return None

    windowed = (samples - np.mean(samples)) * np.hamming(len(samples))
    corr = np.correlate(windowed, windowed, mode="full")
    corr = corr[len(corr) // 2:]

    min_lag = int(sample_rate / PITCH_MAX_HZ)
    max_lag = min(int(sample_rate / PITCH_MIN_HZ), len(corr) - 1)
    if max_lag <= min_lag:
        return None

    search_region = corr[min_lag:max_lag]
    if search_region.size == 0 or np.max(search_region) <= 0:
        return None

    peak_val = np.max(search_region)
    if corr[0] > 0 and (peak_val / corr[0]) < 0.25:
        return None

    peak_lag_int = int(np.argmax(search_region)) + min_lag
    refined_lag = _parabolic_interpolation(corr, peak_lag_int)
    if refined_lag <= 0:
        return None

    pitch = float(sample_rate / refined_lag)
    if pitch < PITCH_MIN_HZ or pitch > PITCH_MAX_HZ:
        return None
    return pitch


def _compute_jitter(samples: np.ndarray, pitch_hz: float, sample_rate: int = AUDIO_SAMPLE_RATE) -> Optional[float]:
    if pitch_hz is None or pitch_hz < PITCH_MIN_HZ:
        return None
    period_samples = int(sample_rate / pitch_hz)
    if period_samples < 4 or len(samples) < period_samples * 3:
        return None

    num_periods = len(samples) // period_samples
    if num_periods < 3:
        return None

    period_lengths = []
    for i in range(num_periods - 1):
        start = i * period_samples
        end = min(start + period_samples * 2, len(samples))
        segment = samples[start:end]
        half = len(segment) // 2
        second_half = np.abs(segment[half:])
        if second_half.size == 0:
            break
        local_peak = half + int(np.argmax(second_half))
        period_lengths.append(local_peak)

    if len(period_lengths) < 2:
        return None

    periods = np.array(period_lengths, dtype=np.float64)
    avg_period = np.mean(periods)
    if avg_period < 1:
        return None

    diffs = np.abs(np.diff(periods))
    return float(min(np.mean(diffs) / avg_period, 0.15))


def _compute_shimmer(samples: np.ndarray, pitch_hz: float, sample_rate: int = AUDIO_SAMPLE_RATE) -> Optional[float]:
    if pitch_hz is None or pitch_hz < PITCH_MIN_HZ:
        return None
    period_samples = int(sample_rate / pitch_hz)
    if period_samples < 4 or len(samples) < period_samples * 3:
        return None

    num_periods = len(samples) // period_samples
    if num_periods < 3:
        return None

    peak_amps = []
    for i in range(num_periods):
        start = i * period_samples
        end = min(start + period_samples, len(samples))
        segment = samples[start:end]
        if segment.size > 0:
            peak_amps.append(float(np.max(np.abs(segment))))

    if len(peak_amps) < 2:
        return None

    amps = np.array(peak_amps, dtype=np.float64)
    avg_amp = np.mean(amps)
    if avg_amp < 1e-6:
        return None

    diffs = np.abs(np.diff(amps))
    return float(min(np.mean(diffs) / avg_amp, 0.40))


def _spectral_centroid(samples: np.ndarray, sample_rate: int = AUDIO_SAMPLE_RATE) -> Optional[float]:
    if samples.size < 256 or _rms(samples) < SILENCE_RMS_THRESHOLD:
        return None
    windowed = samples * np.hamming(len(samples))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(windowed), d=1.0 / sample_rate)

    total_energy = np.sum(spectrum)
    if total_energy < 1e-10:
        return None

    centroid = float(np.sum(freqs * spectrum) / total_energy)
    if centroid < 150 or centroid > 7500:
        return None
    return centroid


def _compute_hnr(samples: np.ndarray, pitch_hz: Optional[float], sample_rate: int = AUDIO_SAMPLE_RATE) -> Optional[float]:
    """Harmonic-to-Noise Ratio (HNR) in dB."""
    if pitch_hz is None or samples.size < 256:
        return None
    corr = np.correlate(samples, samples, mode="full")
    corr = corr[len(corr) // 2:]
    period_lag = int(sample_rate / pitch_hz)
    if period_lag >= len(corr) or corr[0] <= 0:
        return None
    r0 = corr[0]
    r_tau = corr[period_lag]
    if r0 <= r_tau or r_tau <= 0:
        return None
    hnr = 10.0 * np.log10(r_tau / (r0 - r_tau))
    return float(max(-10.0, min(hnr, 35.0)))


class AudioFeatureExtractor:
    """Stateful audio feature extractor with robust Voice Activity Detection (VAD),
    baseline calibration, and multi-metric speech DSP."""

    def __init__(self, calibrator: Optional[BaselineCalibrator] = None, sample_rate: int = AUDIO_SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.calibrator = calibrator if calibrator is not None else BaselineCalibrator()
        
        self._pitch_history: deque = deque()
        self._peak_timestamps: deque = deque()
        self._peak_energy = 0.3
        self._energy_decay = 0.998

        self._audio_buffers: Dict[str, deque] = {
            "voice_pitch": deque(maxlen=MEDIAN_FILTER_WINDOW),
            "voice_energy": deque(maxlen=MEDIAN_FILTER_WINDOW),
            "speech_rate": deque(maxlen=MEDIAN_FILTER_WINDOW),
            "jitter": deque(maxlen=MEDIAN_FILTER_WINDOW),
            "shimmer": deque(maxlen=MEDIAN_FILTER_WINDOW),
            "spectral_centroid": deque(maxlen=MEDIAN_FILTER_WINDOW),
        }

    def update(self, samples: np.ndarray, now: Optional[float] = None) -> dict:
        now = now if now is not None else time.time()

        if samples.size == 0:
            return self._empty_response(silence=True)

        energy_raw = _rms(samples)
        zcr = _zero_crossing_rate(samples)

        # Multi-condition VAD (Voice Activity Detection)
        is_silence = energy_raw < SILENCE_RMS_THRESHOLD or (zcr > VAD_ZCR_MAX_SILENCE and energy_raw < SILENCE_RMS_THRESHOLD * 2.0)

        if is_silence:
            return self._empty_response(silence=True, audio_level=round(energy_raw, 4))

        # Peak energy tracking for adaptive normalization
        if energy_raw > self._peak_energy:
            self._peak_energy = energy_raw
        else:
            self._peak_energy *= self._energy_decay
            self._peak_energy = max(self._peak_energy, 0.01)

        voice_energy = max(0.0, min(1.0, energy_raw / self._peak_energy))

        pitch = estimate_pitch(samples, self.sample_rate)
        centroid = _spectral_centroid(samples, self.sample_rate)
        jitter = _compute_jitter(samples, pitch, self.sample_rate) if pitch else None
        shimmer = _compute_shimmer(samples, pitch, self.sample_rate) if pitch else None
        hnr = _compute_hnr(samples, pitch, self.sample_rate) if pitch else None

        if pitch is not None:
            self._pitch_history.append((now, pitch))

        # Purge pitch history outside window
        cutoff = now - PITCH_HISTORY_SECONDS
        while self._pitch_history and self._pitch_history[0][0] < cutoff:
            self._pitch_history.popleft()

        pitch_values = [p for _, p in self._pitch_history]
        pitch_variance = float(np.std(pitch_values)) if len(pitch_values) >= 2 else None

        speech_rate = self._update_speech_rate(samples, now)

        raw_sample = {
            "voice_pitch": pitch,
            "pitch_variance": pitch_variance,
            "voice_energy": voice_energy,
            "speech_rate": speech_rate,
            "spectral_centroid": centroid,
            "jitter": jitter,
            "shimmer": shimmer,
            "silence_detected": False,
        }

        is_calibrating = self.calibrator.is_calibrating(now)
        if is_calibrating:
            self.calibrator.add_audio_sample(raw_sample)

        return {
            "voice_pitch": round(pitch, 1) if pitch else None,
            "pitch_variance": round(pitch_variance, 1) if pitch_variance else None,
            "voice_energy": round(voice_energy, 3),
            "audio_level": round(energy_raw, 4),
            "speech_rate": round(speech_rate, 1) if speech_rate else None,
            "silence_detected": False,
            "jitter": round(jitter, 4) if jitter else None,
            "shimmer": round(shimmer, 4) if shimmer else None,
            "spectral_centroid": round(centroid, 1) if centroid else None,
            "hnr": round(hnr, 1) if hnr else None,
            "is_calibrating": is_calibrating,
        }

    def _update_speech_rate(self, samples: np.ndarray, now: float) -> Optional[float]:
        if samples.size == 0:
            return None
        env = np.abs(samples)
        if env.size > 10:
            smoothed = np.convolve(env, np.ones(10) / 10.0, mode="same")
            peaks = np.where((smoothed[1:-1] > smoothed[:-2]) & (smoothed[1:-1] > smoothed[2:]) & (smoothed[1:-1] > 0.05))[0]
            if len(peaks) > 0:
                self._peak_timestamps.append(now)

        cutoff = now - SPEECH_RATE_WINDOW_SECONDS
        while self._peak_timestamps and self._peak_timestamps[0] < cutoff:
            self._peak_timestamps.popleft()

        if not self._peak_timestamps:
            return None

        elapsed = max(now - self._peak_timestamps[0], 2.0)
        rate = len(self._peak_timestamps) * (60.0 / elapsed)
        return min(rate, 250.0)

    def _empty_response(self, silence: bool = True, audio_level: float = 0.0) -> dict:
        return {
            "voice_pitch": None,
            "pitch_variance": None,
            "voice_energy": 0.0,
            "audio_level": audio_level,
            "speech_rate": None,
            "silence_detected": silence,
            "jitter": None,
            "shimmer": None,
            "spectral_centroid": None,
            "hnr": None,
            "is_calibrating": False,
        }
