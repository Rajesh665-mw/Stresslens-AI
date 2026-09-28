"""
Entry point for the multimodal perception & signal-acquisition pipeline.

Usage (from the project root):
    python -m perception.main --mode demo --profile calm
    python -m perception.main --mode demo --profile medium
    python -m perception.main --mode demo --profile high
    python -m perception.main --mode live
    python -m perception.main --mode live --no-audio
    python -m perception.main --mode live --no-video

Live mode runs a video thread and an audio thread, each updating shared
state, while the main thread periodically fuses whatever is currently
available and writes it to the shared features.json.
"""
import argparse
import sys
import threading
import time
from typing import Optional

from perception.config import (
    CAMERA_INDEX,
    FUSION_INTERVAL_SECONDS,
    VIDEO_TARGET_FPS,
    AUDIO_BLOCK_SECONDS,
    CALIBRATION_DURATION_SECONDS,
)
from perception.calibration import BaselineCalibrator
from perception.fusion.feature_fusion import StressFusionEngine
from perception.feature_writer import write_features
from utils.logger import get_logger

logger = get_logger("perception.main")


class LiveState:
    """Thread-safe shared state between the video/audio threads and the
    main fusion loop. Each side only ever writes its own half."""

    def __init__(self):
        self._lock = threading.Lock()
        self.facial: Optional[dict] = None
        self.audio: Optional[dict] = None
        self.camera_active = False
        self.mic_active = False

    def set_facial(self, value: dict):
        with self._lock:
            self.facial = value

    def set_audio(self, value: dict):
        with self._lock:
            self.audio = value

    def snapshot(self):
        with self._lock:
            return self.facial, self.audio, self.camera_active, self.mic_active


def _video_loop(state: LiveState, stop_event: threading.Event, camera_index: int, calibrator: BaselineCalibrator):
    """Runs in its own thread: opens the webcam + face mesh, and keeps
    `state.facial` fresh."""
    import cv2
    from perception.video.face_mesh import VideoCapture, FaceMeshProcessor, CameraUnavailableError
    from perception.video.facial_features import FacialFeatureExtractor
    from perception.feature_writer import write_frame

    extractor = FacialFeatureExtractor(calibrator=calibrator)
    frame_interval = 1.0 / max(1, VIDEO_TARGET_FPS)

    try:
        cap = VideoCapture(camera_index)
    except CameraUnavailableError as e:
        logger.warning(f"Video disabled: {e}")
        with state._lock:
            state.camera_active = False
        return

    if not cap.open():
        logger.warning("Camera unavailable -- video features will stay off for this session.")
        with state._lock:
            state.camera_active = False
        return

    try:
        face_mesh = FaceMeshProcessor(max_num_faces=1)
    except CameraUnavailableError as e:
        logger.warning(f"Video disabled: {e}")
        cap.release()
        with state._lock:
            state.camera_active = False
        return

    with state._lock:
        state.camera_active = True
    logger.info("Video thread started.")

    try:
        while not stop_event.is_set():
            loop_start = time.time()
            frame = cap.read_frame()
            if frame is None:
                with state._lock:
                    state.camera_active = False
                time.sleep(0.5)
                continue

            with state._lock:
                state.camera_active = True

            try:
                landmarks = face_mesh.process(frame)
            except Exception as e:
                logger.warning(f"Face mesh processing error: {e}")
                landmarks = None

            try:
                annotated = face_mesh.draw_landmarks(frame, landmarks)
                ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if ok:
                    write_frame(buf.tobytes())
            except Exception as e:
                logger.warning(f"Error drawing or encoding frame preview: {e}")

            facial_metrics = extractor.update(landmarks)
            state.set_facial(facial_metrics)

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, frame_interval - elapsed))
    finally:
        cap.release()
        face_mesh.close()
        with state._lock:
            state.camera_active = False
        logger.info("Video thread stopped.")


def _audio_loop(state: LiveState, stop_event: threading.Event, calibrator: BaselineCalibrator):
    """Runs in its own thread: opens the mic and keeps `state.audio` fresh."""
    from perception.audio.mic_capture import MicCapture, MicrophoneUnavailableError
    from perception.audio.audio_features import AudioFeatureExtractor

    try:
        mic = MicCapture()
    except MicrophoneUnavailableError as e:
        logger.warning(f"Audio disabled: {e}")
        return

    if not mic.start():
        logger.warning("Microphone unavailable -- audio features will stay off for this session.")
        with state._lock:
            state.mic_active = False
        return

    with state._lock:
        state.mic_active = True
    logger.info("Audio thread started.")

    extractor = AudioFeatureExtractor(calibrator=calibrator)
    try:
        while not stop_event.is_set():
            time.sleep(AUDIO_BLOCK_SECONDS)
            with state._lock:
                state.mic_active = mic.is_available()
            samples = mic.read_recent(seconds=1.0)
            audio_metrics = extractor.update(samples)
            state.set_audio(audio_metrics)
    finally:
        mic.stop()
        with state._lock:
            state.mic_active = False
        logger.info("Audio thread stopped.")


def run_live(use_video: bool = True, use_audio: bool = True, camera_index: int = CAMERA_INDEX):
    state = LiveState()
    stop_event = threading.Event()
    
    calibrator = BaselineCalibrator(calibration_duration=CALIBRATION_DURATION_SECONDS)
    fusion = StressFusionEngine(calibrator=calibrator)
    threads = []

    if use_video:
        t = threading.Thread(target=_video_loop, args=(state, stop_event, camera_index, calibrator), daemon=True)
        t.start()
        threads.append(t)
    if use_audio:
        t = threading.Thread(target=_audio_loop, args=(state, stop_event, calibrator), daemon=True)
        t.start()
        threads.append(t)

    logger.info(f"Fusion loop running every {FUSION_INTERVAL_SECONDS}s. Ctrl+C to stop.")
    try:
        while True:
            facial, audio, camera_active, mic_active = state.snapshot()
            reading = fusion.fuse(facial, audio, camera_active, mic_active)
            if write_features(reading):
                logger.info(
                    f"stress_score={reading['stress_score']} level={reading['stress_level']} "
                    f"confidence={reading['confidence']} camera={camera_active} mic={mic_active}"
                )
            time.sleep(FUSION_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=2.0)


def main():
    parser = argparse.ArgumentParser(description="Multimodal physiological signal acquisition & perception pipeline")
    parser.add_argument("--mode", choices=["live", "demo"], default="demo", help="Capture mode")
    parser.add_argument("--profile", choices=["calm", "medium", "high"], default="calm", help="Demo profile (demo mode only)")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between demo writes")
    parser.add_argument("--camera-index", type=int, default=CAMERA_INDEX, help="OpenCV camera index (live mode only)")
    parser.add_argument("--no-video", action="store_true", help="Disable webcam capture (live mode only)")
    parser.add_argument("--no-audio", action="store_true", help="Disable microphone capture (live mode only)")
    args = parser.parse_args()

    if args.mode == "demo":
        from perception.demo.demo_mode import run_demo
        run_demo(profile=args.profile, interval_seconds=args.interval)
    else:
        if args.no_video and args.no_audio:
            logger.error("Live mode with both --no-video and --no-audio has nothing to capture. Exiting.")
            sys.exit(1)
        run_live(use_video=not args.no_video, use_audio=not args.no_audio, camera_index=args.camera_index)


if __name__ == "__main__":
    main()
