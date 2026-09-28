"""
Automated Verification Suite for the High-Accuracy Calibrated Stress Detector.

Validates all 9 test conditions specified in the requirements:
1. Neutral face + Normal speech -> Low Stress
2. Relaxed smile -> Low Stress
3. Serious face + Brow/Jaw tension -> Moderate Stress
4. Fast blinking + Lip compression + Rigid face + Raised voice -> High Stress
5. No face -> Face features ignored
6. Silence -> Voice features ignored
7. Face calm + Voice stressed -> Moderate Stress with reduced confidence
8. Face stressed + Voice calm -> Moderate Stress with reduced confidence
9. Both inputs missing -> Insufficient data for stress estimation
"""
import sys
import unittest
import numpy as np

from person_a.calibration import BaselineCalibrator
from person_a.fusion.feature_fusion import StressFusionEngine
from person_a.video.facial_features import FacialFeatureExtractor
from person_a.audio.audio_features import AudioFeatureExtractor, estimate_pitch


class TestStressDetectorAccuracy(unittest.TestCase):

    def setUp(self):
        # Create shared calibrator and bypass duration for instant calibration testing
        self.calibrator = BaselineCalibrator(calibration_duration=0.1)
        self.fusion = StressFusionEngine(calibrator=self.calibrator)
        self.facial_extractor = FacialFeatureExtractor(calibrator=self.calibrator)
        self.audio_extractor = AudioFeatureExtractor(calibrator=self.calibrator)

        # Seed calibration data for resting state
        for _ in range(10):
            self.calibrator.add_facial_sample({
                "blink_rate": 17.0,
                "eye_openness": 0.65,
                "brow_gap": 0.055,
                "inner_brow_gap": 0.09,
                "lip_gap": 0.02,
                "lip_thickness": 0.035,
                "jaw_span": 0.45,
                "head_movement": 0.05,
                "forehead_gap": 0.12,
                "lip_corner_droop": 0.04,
            })
            self.calibrator.add_audio_sample({
                "voice_pitch": 150.0,
                "pitch_variance": 12.0,
                "voice_energy": 0.35,
                "speech_rate": 110.0,
                "spectral_centroid": 1700.0,
                "jitter": 0.012,
                "shimmer": 0.04,
                "silence_detected": False,
            })
        self.calibrator._finalize_calibration()

    def test_1_neutral_face_normal_speech(self):
        """Test 1: Neutral face + Normal speech -> Low Stress"""
        facial = {
            "blink_rate": 17.0,
            "eye_openness": 0.65,
            "brow_tension": 0.05,
            "mouth_tension": 0.05,
            "head_movement": 0.05,
            "face_confidence": 0.95,
            "forehead_tension": 0.05,
            "lip_depression": 0.0,
            "is_calibrating": False,
        }
        audio = {
            "voice_pitch": 152.0,
            "pitch_variance": 13.0,
            "voice_energy": 0.36,
            "speech_rate": 112.0,
            "jitter": 0.013,
            "shimmer": 0.042,
            "spectral_centroid": 1720.0,
            "silence_detected": False,
            "is_calibrating": False,
        }
        res = self.fusion.fuse(facial, audio, camera_active=True, mic_active=True)
        print("\nTest 1 Result:", res["stress_score"], res["stress_level"], f"Conf: {res['confidence']}")
        self.assertLess(res["stress_score"], 35.0)
        self.assertEqual(res["stress_level"], "Low")

    def test_2_relaxed_smile(self):
        """Test 2: Relaxed smile -> Low Stress"""
        facial = {
            "blink_rate": 16.0,
            "eye_openness": 0.60,
            "brow_tension": 0.0,
            "mouth_tension": 0.08,
            "head_movement": 0.04,
            "face_confidence": 0.98,
            "forehead_tension": 0.0,
            "lip_depression": 0.0,
            "is_calibrating": False,
        }
        res = self.fusion.fuse(facial, None, camera_active=True, mic_active=False)
        print("Test 2 Result:", res["stress_score"], res["stress_level"], f"Conf: {res['confidence']}")
        self.assertLess(res["stress_score"], 35.0)
        self.assertEqual(res["stress_level"], "Low")

    def test_3_serious_face_brow_jaw_tension(self):
        """Test 3: Serious face + Raised eyebrows + Jaw tension -> Moderate Stress"""
        facial = {
            "blink_rate": 22.0,
            "eye_openness": 0.70,
            "brow_tension": 0.55,
            "mouth_tension": 0.50,
            "head_movement": 0.15,
            "face_confidence": 0.95,
            "forehead_tension": 0.45,
            "lip_depression": 0.30,
            "is_calibrating": False,
        }
        res = self.fusion.fuse(facial, None, camera_active=True, mic_active=False)
        print("Test 3 Result:", res["stress_score"], res["stress_level"], f"Conf: {res['confidence']}")
        self.assertTrue(35.0 <= res["stress_score"] <= 68.0, f"Score {res['stress_score']} not Moderate!")
        self.assertEqual(res["stress_level"], "Moderate")

    def test_4_fast_blinking_lip_compression_raised_voice(self):
        """Test 4: Fast blinking + Lip compression + Rigid face + Raised voice -> High Stress"""
        facial = {
            "blink_rate": 38.0,
            "eye_openness": 0.55,
            "brow_tension": 0.85,
            "mouth_tension": 0.82,
            "head_movement": 0.35,
            "face_confidence": 0.95,
            "forehead_tension": 0.80,
            "lip_depression": 0.75,
            "facial_rigidity": 0.70,
            "is_calibrating": False,
        }
        audio = {
            "voice_pitch": 215.0,  # +43% pitch rise
            "pitch_variance": 35.0,
            "voice_energy": 0.75,
            "speech_rate": 180.0,
            "jitter": 0.045,
            "shimmer": 0.14,
            "spectral_centroid": 2800.0,
            "silence_detected": False,
            "is_calibrating": False,
        }
        res = self.fusion.fuse(facial, audio, camera_active=True, mic_active=True)
        print("Test 4 Result:", res["stress_score"], res["stress_level"], f"Conf: {res['confidence']}")
        self.assertGreaterEqual(res["stress_score"], 68.0)
        self.assertEqual(res["stress_level"], "High")

    def test_5_no_face(self):
        """Test 5: No face -> Face features ignored"""
        facial = {"face_confidence": 0.0, "is_calibrating": False}
        audio = {
            "voice_pitch": 150.0,
            "pitch_variance": 12.0,
            "voice_energy": 0.35,
            "speech_rate": 110.0,
            "silence_detected": False,
            "is_calibrating": False,
        }
        res = self.fusion.fuse(facial, audio, camera_active=False, mic_active=True)
        print("Test 5 Result:", res["stress_score"], res["stress_level"], f"Conf: {res['confidence']}")
        self.assertLess(res["stress_score"], 35.0)

    def test_6_silence(self):
        """Test 6: Silence -> Voice features ignored"""
        facial = {
            "blink_rate": 17.0,
            "eye_openness": 0.65,
            "brow_tension": 0.0,
            "mouth_tension": 0.0,
            "face_confidence": 0.95,
            "is_calibrating": False,
        }
        audio = {"silence_detected": True, "is_calibrating": False}
        res = self.fusion.fuse(facial, audio, camera_active=True, mic_active=True)
        print("Test 6 Result:", res["stress_score"], res["stress_level"], f"Conf: {res['confidence']}")
        self.assertLess(res["stress_score"], 35.0)

    def test_7_face_calm_voice_stressed(self):
        """Test 7: Face calm + Voice stressed -> Moderate Stress with reduced confidence"""
        facial = {
            "blink_rate": 17.0,
            "brow_tension": 0.05,
            "mouth_tension": 0.05,
            "face_confidence": 0.95,
            "is_calibrating": False,
        }
        audio = {
            "voice_pitch": 220.0,
            "pitch_variance": 38.0,
            "voice_energy": 0.80,
            "speech_rate": 185.0,
            "jitter": 0.048,
            "silence_detected": False,
            "is_calibrating": False,
        }
        res = self.fusion.fuse(facial, audio, camera_active=True, mic_active=True)
        print("Test 7 Result:", res["stress_score"], res["stress_level"], f"Conf: {res['confidence']}")
        self.assertTrue(35.0 <= res["stress_score"] <= 68.0)
        self.assertEqual(res["stress_level"], "Moderate")
        self.assertLessEqual(res["confidence"], 0.65)  # reduced confidence due to conflict!

    def test_8_face_stressed_voice_calm(self):
        """Test 8: Face stressed + Voice calm -> Moderate Stress with reduced confidence"""
        facial = {
            "blink_rate": 35.0,
            "brow_tension": 0.85,
            "mouth_tension": 0.80,
            "forehead_tension": 0.75,
            "face_confidence": 0.95,
            "is_calibrating": False,
        }
        audio = {
            "voice_pitch": 150.0,
            "pitch_variance": 10.0,
            "voice_energy": 0.30,
            "speech_rate": 105.0,
            "silence_detected": False,
            "is_calibrating": False,
        }
        res = self.fusion.fuse(facial, audio, camera_active=True, mic_active=True)
        print("Test 8 Result:", res["stress_score"], res["stress_level"], f"Conf: {res['confidence']}")
        self.assertTrue(35.0 <= res["stress_score"] <= 68.0)
        self.assertEqual(res["stress_level"], "Moderate")
        self.assertLessEqual(res["confidence"], 0.65)  # reduced confidence!

    def test_9_both_unavailable(self):
        """Test 9: Both inputs missing -> Insufficient data for stress estimation"""
        facial = {"face_confidence": 0.0, "is_calibrating": False}
        audio = {"silence_detected": True, "is_calibrating": False}
        res = self.fusion.fuse(facial, audio, camera_active=False, mic_active=False)
        print("Test 9 Result:", res["stress_score"], res["stress_level"], f"Conf: {res['confidence']}")
        self.assertEqual(res["stress_level"], "Insufficient Data")
        self.assertEqual(res["confidence"], 0.0)
        self.assertEqual(res["status_message"], "Insufficient data for stress estimation")


if __name__ == "__main__":
    unittest.main()
