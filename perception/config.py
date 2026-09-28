"""
Configuration for the perception signal-acquisition pipeline.

Kept modular from utils/config.py (backend/UI config) so perception thresholds,
weights, and audio/video device settings can be tuned cleanly. Both modules share
the same FEATURES_FILE path from utils/config.py as the integration contract.
"""

# ---- Devices ----
CAMERA_INDEX = 0                 # OpenCV camera index; change if you have multiple webcams
AUDIO_SAMPLE_RATE = 16000        # Hz -- fine for pitch/energy work, keeps CPU load low
AUDIO_BLOCK_SECONDS = 0.1        # ~100ms audio blocks – more signal per chunk, fewer empty reads
AUDIO_BUFFER_SECONDS = 5.0       # how much recent audio we keep around for rate/variance calcs

# ---- Timing & Calibration ----
FUSION_INTERVAL_SECONDS = 15.0   # fuse face & voice metrics and update meter every 15 seconds
VIDEO_TARGET_FPS = 24            # smoother webcam preview, faster facial feature refresh
CALIBRATION_DURATION_SECONDS = 15.0 # initial baseline calibration phase (seconds)

# ---- Facial feature thresholds ----
EAR_BLINK_THRESHOLD = 0.19       # eye-aspect-ratio below this counts as "eye closed"
BLINK_MIN_FRAMES_CLOSED = 2      # minimum consecutive closed-eye frames to register a blink (avoids noise)
BLINK_RATE_WINDOW_SECONDS = 60.0 # rolling window used to estimate blinks-per-minute
DEFAULT_BLINK_RATE = 17.0        # typical adult resting blink rate (~17 bpm)
HEAD_MOVEMENT_HISTORY = 8        # nose history depth for smoothing jitter
FACE_CONFIDENCE_DECAY = 0.70     # decay factor per missing frame when face is lost

# ---- Audio feature thresholds ----
SILENCE_RMS_THRESHOLD = 0.008    # RMS energy below this is treated as silence/unvoiced noise
VAD_ZCR_MAX_SILENCE = 0.35       # High ZCR with low energy = noise/hiss, not voice
PITCH_MIN_HZ = 70                # plausible human voice pitch range for autocorrelation search
PITCH_MAX_HZ = 400
PITCH_HISTORY_SECONDS = 10.0     # rolling window for pitch variance
SPEECH_RATE_WINDOW_SECONDS = 5.0 # rolling window used to estimate a syllable-rate proxy

# ---- Smoothing ----
EMA_ALPHA = 0.35                 # smooth EMA factor to prevent rapid erratic score jumps
MEDIAN_FILTER_WINDOW = 5         # median filter window to eliminate single-frame glitches

# ---- Stress scoring levels ----
STRESS_LEVEL_LOW_MAX = 35.0      # score < 35  -> Low
STRESS_LEVEL_MODERATE_MAX = 68.0 # score < 68 -> Moderate, else High

# Relative weights of each modality's contribution to the fused stress score.
FACIAL_WEIGHT = 0.55
AUDIO_WEIGHT = 0.45

# ---- Empirical Deviation Ranges (Inspired by WESAD / RAVDESS / UBFC-Phys) ----
# Relative fractional change relative to personal baseline (e.g., +0.50 = +50% above baseline)
BLINK_RATE_STRESS_SCALE = 0.80     # +80% blink rate increase -> max blink stress
BROW_FURROW_STRESS_SCALE = 0.35    # -35% brow gap reduction -> max brow stress
LIP_COMPRESSION_STRESS_SCALE = 0.40 # +40% lip compression -> max mouth stress
JAW_CLENCH_STRESS_SCALE = 0.30      # -30% jaw span reduction -> max jaw stress
HEAD_RIGIDITY_STRESS_SCALE = 0.60  # rigidity / movement deviation scale
PITCH_RISE_STRESS_SCALE = 0.30     # +30% voice pitch increase -> max pitch stress
PITCH_VAR_STRESS_SCALE = 1.00      # +100% pitch variance rise -> max variance stress
ENERGY_RISE_STRESS_SCALE = 0.60    # +60% RMS energy rise -> max energy stress
SPEECH_RATE_STRESS_SCALE = 0.50    # +50% speech acceleration -> max rate stress
JITTER_STRESS_RANGE = 0.035        # absolute jitter stress scale
SHIMMER_STRESS_RANGE = 0.12        # absolute shimmer stress scale
SPECTRAL_CENTROID_STRESS_SCALE = 0.45 # +45% spectral centroid rise -> max centroid stress
