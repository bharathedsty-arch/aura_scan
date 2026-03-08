import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class PoseDetector:
    def __init__(self, pose_model='pose_landmarker.task', hand_model='hand_landmarker.task'):
        # Check and download models if missing
        self._ensure_model(pose_model, "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task")
        self._ensure_model(hand_model, "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

        # 1. Initialize Pose Landmarker
        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=pose_model),
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

        # 2. Initialize Hand Landmarker
        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=hand_model),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

        self.pose_results = None
        self.hand_results = None
        self._frame_timestamp_ms = 0

    def _ensure_model(self, path, url):
        if not os.path.exists(path):
            print(f"Downloading model to {path}...")
            urllib.request.urlretrieve(url, path)

    def find_pose(self, img):
        """Holistic detection: Pose + Hands"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        self._frame_timestamp_ms += 1
        
        # Detect Pose
        self.pose_results = self.pose_landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)
        
        # Detect Hands
        self.hand_results = self.hand_landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)
        
        return {"pose": self.pose_results, "hands": self.hand_results}

    def get_landmarks(self, img):
        """Returns pose landmarks with hand data as extended indices (optional or separate)."""
        landmarks = []
        if self.pose_results and self.pose_results.pose_landmarks:
            for id, lm in enumerate(self.pose_results.pose_landmarks[0]):
                h, w, _ = img.shape
                cx, cy, cz = int(lm.x * w), int(lm.y * h), lm.z
                landmarks.append([id, cx, cy, cz])
        return landmarks

    def get_hand_landmarks(self, img):
        """Returns hand landmarks (usually 21 points per hand)."""
        hands_data = []
        if self.hand_results and self.hand_results.hand_landmarks:
            for hand_landmarks in self.hand_results.hand_landmarks:
                points = []
                for lm in hand_landmarks:
                    h, w, _ = img.shape
                    cx, cy, cz = int(lm.x * w), int(lm.y * h), lm.z
                    points.append([cx, cy, cz])
                hands_data.append(points)
        return hands_data
    
    def __del__(self):
        if hasattr(self, 'pose_landmarker'):
            self.pose_landmarker.close()
        if hasattr(self, 'hand_landmarker'):
            self.hand_landmarker.close()
