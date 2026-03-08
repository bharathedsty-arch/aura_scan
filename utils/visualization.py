import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from mediapipe.tasks.python.vision import drawing_utils as mp_drawing

class Visualizer:
    def __init__(self):
        # Manually define POSE_CONNECTIONS as legacy solutions is missing
        self.pose_connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
            (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (15, 17), (16, 18),
            (17, 19), (18, 20), (15, 21), (16, 22), (27, 29), (28, 30), (29, 31), (30, 32)
        ]
        self.hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]
        self.frame_count = 0
        self.history = {} # landmark_id: deque of (x, y)
        self.trail_length = 8

    def draw_skeleton(self, img, results, risk_components=None):
        self.frame_count += 1
        
        # 0. Draw Scanning Overlay (Sci-Fi Feel)
        self._draw_scanning_overlay(img)

        # 1. Holistic Drawing (Pose + Hands)
        pose_results = results.get("pose")
        hand_results = results.get("hands")

        if pose_results and pose_results.pose_landmarks:
            for pose_landmarks in pose_results.pose_landmarks:
                # 1a. Draw Skeleton Lines
                for connection in self.pose_connections:
                    start_idx, end_idx = connection
                    if start_idx < len(pose_landmarks) and end_idx < len(pose_landmarks):
                        h, w, _ = img.shape
                        p1 = pose_landmarks[start_idx]
                        p2 = pose_landmarks[end_idx]
                        
                        color = (0, 255, 156) # Neon Green
                        if risk_components:
                            if connection[0] in [11, 12, 23, 24]: # Spine
                                color = self._get_color(risk_components.get('spine_risk', 0))
                            elif connection[0] in [25, 26, 27, 28]: # Knee
                                color = self._get_color(risk_components.get('knee_risk', 0))

                        cv2.line(img, (int(p1.x * w), int(p1.y * h)), 
                                 (int(p2.x * w), int(p2.y * h)), color, 2, cv2.LINE_AA)
                
                # 1b. Motion Trails (Dynamic Visual)
                self._draw_motion_trails(img, pose_landmarks)

        if hand_results and hand_results.hand_landmarks:
            for hand_landmarks in hand_results.hand_landmarks:
                for connection in self.hand_connections:
                    h, w, _ = img.shape
                    p1 = hand_landmarks[connection[0]]
                    p2 = hand_landmarks[connection[1]]
                    cv2.line(img, (int(p1.x * w), int(p1.y * h)), 
                             (int(p2.x * w), int(p2.y * h)), (0, 229, 255), 1, cv2.LINE_AA)
                for lm in hand_landmarks:
                    cv2.circle(img, (int(lm.x * w), int(lm.y * h)), 2, (122, 92, 255), -1)

        return img

    def highlight_stress(self, img, landmarks, risk_components):
        """Heatmap Glow effect for stressed joints."""
        spine_score = risk_components.get('spine_risk', 0)
        knee_score = risk_components.get('knee_risk', 0)
        
        # Highlight Spine
        for idx in [11, 12, 23, 24]:
            self._draw_heatmap_point(img, landmarks, idx, spine_score, "SPINE" if idx == 24 else None)
        
        # Highlight Knees
        for idx in [25, 26]:
            self._draw_heatmap_point(img, landmarks, idx, knee_score, "KNEE" if idx == 26 else None)

        return img

    def _get_color(self, score):
        if score > 70: return (92, 59, 255) # Neon Red/Alert (using BGR - 255, 59, 92 is RGB)
        if score > 40: return (0, 255, 255) # Yellow
        return (156, 255, 0) # Neon Green

    def _draw_motion_trails(self, img, landmarks):
        h, w, _ = img.shape
        tracked_indices = [15, 16, 27, 28] # Hands and Feet for trails
        
        for idx in tracked_indices:
            lm = landmarks[idx]
            pos = (int(lm.x * w), int(lm.y * h))
            
            if idx not in self.history:
                self.history[idx] = deque(maxlen=self.trail_length)
            self.history[idx].append(pos)
            
            # Draw trail
            points = list(self.history[idx])
            for i in range(1, len(points)):
                thickness = int(np.sqrt(self.trail_length / float(i + 1)) * 2)
                alpha = int(255 * (i / self.trail_length))
                color = (0, 229, 255) # Neon Blue
                cv2.line(img, points[i-1], points[i], color, thickness, cv2.LINE_AA)

    def _draw_heatmap_point(self, img, landmarks, idx, score, label=None):
        point = next((lm[1:3] for lm in landmarks if lm[0] == idx), None)
        if not point or score < 20: return

        color = self._get_color(score)
        
        # Blinking for high stress
        if score > 70 and (self.frame_count // 5) % 2 == 0:
            color = (255, 255, 255)

        # Sci-Fi Heat Glow (Concentric circles with fading alpha)
        overlay = img.copy()
        for r in [15, 25, 35]:
            cv2.circle(overlay, (point[0], point[1]), r, color, -1 if r == 15 else 2, cv2.LINE_AA)
        
        alpha = np.clip(score / 100.0, 0.2, 0.6)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        if label and score > 40:
            cv2.putText(img, label, (point[0] + 15, point[1] - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_scanning_overlay(self, img):
        h, w, _ = img.shape
        # Moving Scan Line
        line_y = (self.frame_count * 5) % h
        cv2.line(img, (0, line_y), (w, line_y), (122, 92, 255), 1, cv2.LINE_AA)
        
        # Tech corner text
        cv2.putText(img, "SCANNING BIO-SIGNATURE...", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 229, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"FRAME_ID: {self.frame_count:06d}", (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 229, 255), 1, cv2.LINE_AA)
        
        # Target HUD corners
        d = 40
        # TL
        cv2.line(img, (10, 10), (10+d, 10), (0, 229, 255), 2)
        cv2.line(img, (10, 10), (10, 10+d), (0, 229, 255), 2)
        # BR
        cv2.line(img, (w-10, h-10), (w-10-d, h-10), (0, 229, 255), 2)
        cv2.line(img, (w-10, h-10), (w-10, h-10-d), (0, 229, 255), 2)

    def overlay_metrics(self, img, risk_score, risk_components):
        h, w, _ = img.shape
        
        # Sci-Fi Side Panel
        overlay = img.copy()
        cv2.rectangle(overlay, (w - 260, 0), (w, 200), (11, 15, 26), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # Border
        cv2.rectangle(img, (w - 260, 0), (w, 200), (122, 92, 255), 1)
        
        status_text = "OPTIMAL" if risk_score < 35 else "WARNING" if risk_score < 70 else "CRITICAL"
        score_color = self._get_color(risk_score)
        
        cv2.putText(img, f"MSD RISK: {risk_score:.1f}%", (w - 240, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"STATUS: {status_text}", (w - 240, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, score_color, 2, cv2.LINE_AA)
        
        # Mini bars
        y_start = 100
        for i, (name, val) in enumerate(risk_components.items()):
            if i > 2: break
            cv2.putText(img, name.replace("_risk", "").upper(), (w - 240, y_start + i*30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            # Bar background
            cv2.rectangle(img, (w - 120, y_start + i*30 - 10), (w - 20, y_start + i*30), (50, 50, 50), -1)
            # Bar fill
            cv2.rectangle(img, (w - 120, y_start + i*30 - 10), (w - 120 + int(val), y_start + i*30), self._get_color(val), -1)

        return img
