"""
Thin wrappers around OpenCV (camera capture) and MediaPipe (face mesh
landmark detection). Both imports are guarded so the rest of the project
-- including demo mode and every test -- works on a machine that doesn't
have opencv-python / mediapipe installed. The guarded flags let callers
fail fast with a clear message only when live video is actually requested.
"""
from typing import Optional

from utils.logger import get_logger

logger = get_logger("perception.face_mesh")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = hasattr(mp, "solutions")
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class CameraUnavailableError(RuntimeError):
    """Raised when the webcam can't be opened or opencv-python isn't installed."""


class VideoCapture:
    """Wraps cv2.VideoCapture with defensive open/read semantics.
    Sets explicit resolution and disables autofocus to reduce per-frame
    processing latency on Windows webcam drivers."""

    def __init__(self, camera_index: int = 0):
        if not CV2_AVAILABLE:
            raise CameraUnavailableError(
                "opencv-python is not installed. Run: pip install -r requirements-perception.txt"
            )
        self.camera_index = camera_index
        self._cap = None

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            # Fallback without CAP_DSHOW
            self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            logger.warning(f"Could not open camera index {self.camera_index}.")
            self._cap = None
            return False
        # Set explicit resolution to reduce driver buffer lag
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap.set(cv2.CAP_PROP_FPS, 30)
        # Minimize driver-level buffering (reduces webcam preview lag)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Disable autofocus if supported (reduces per-frame latency)
        self._cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        return True

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def read_frame(self):
        """Returns a BGR frame (numpy array) or None if a frame couldn't be read."""
        if not self.is_open():
            return None
        ok, frame = self._cap.read()
        if not ok:
            return None
        return frame

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class FaceMeshProcessor:
    """Wraps MediaPipe's Face Mesh solution. Configured for a single
    primary face (max_num_faces=1) per the requirement to ignore any
    secondary faces in frame.

    Detection/tracking thresholds are lowered vs. defaults (0.4/0.35)
    so the mesh sustains tracking through subtle micro-expressions and
    partial face occlusions instead of dropping the face."""

    def __init__(self, max_num_faces: int = 1):
        if not MEDIAPIPE_AVAILABLE:
            raise CameraUnavailableError(
                "mediapipe is not installed. Run: pip install -r requirements-perception.txt"
            )
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.35,
        )

    def process(self, bgr_frame) -> Optional[list]:
        """Runs face-mesh on one BGR frame. Returns a list of (x, y, z)
        normalized landmark tuples for the primary face, or None if no
        face was detected."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        # max_num_faces=1 guarantees at most one entry; this is our primary face.
        landmarks = results.multi_face_landmarks[0].landmark
        return [(lm.x, lm.y, lm.z) for lm in landmarks]

    def draw_landmarks(self, bgr_frame, landmarks: Optional[list]):
        """Draws real-time Face Mesh wireframe and landmark overlays on the frame.
        Includes eye contours, brow arcs, mouth outline, and nose bridge."""
        if not landmarks or not CV2_AVAILABLE:
            return bgr_frame
        import numpy as np
        annotated = bgr_frame.copy()
        h, w, _ = annotated.shape

        # Draw all landmarks as subtle cyan dots
        for x, y, z in landmarks:
            px, py = int(x * w), int(y * h)
            cv2.circle(annotated, (px, py), 1, (235, 206, 0), -1)

        # Highlight eye contours with bright cyan wireframe
        from perception.video.facial_features import RIGHT_EYE, LEFT_EYE, MOUTH_OUTLINE, BROW_LEFT_ARC, BROW_RIGHT_ARC
        for eye in [RIGHT_EYE, LEFT_EYE]:
            pts = np.array([[int(landmarks[i][0] * w), int(landmarks[i][1] * h)] for i in eye], dtype=np.int32)
            cv2.polylines(annotated, [pts], True, (0, 255, 255), 1)

        # Draw mouth outline
        pts = np.array([[int(landmarks[i][0] * w), int(landmarks[i][1] * h)] for i in MOUTH_OUTLINE], dtype=np.int32)
        cv2.polylines(annotated, [pts], True, (180, 105, 255), 1)

        # Draw brow arcs
        for brow in [BROW_LEFT_ARC, BROW_RIGHT_ARC]:
            pts = np.array([[int(landmarks[i][0] * w), int(landmarks[i][1] * h)] for i in brow], dtype=np.int32)
            cv2.polylines(annotated, [pts], False, (100, 255, 100), 1)

        return annotated

    def close(self):
        self._face_mesh.close()

