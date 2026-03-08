import numpy as np
import time
from collections import deque

class MotionAnalyzer:
    def __init__(self, window_size=30): # Increased window for gait analysis
        self.window_size = window_size
        self.history = {} # landmark_id: deque of (x, y, z, timestamp)
        self.step_history = {"left": [], "right": []}

    def get_body_velocity(self, landmarks):
        """Calculates approximate body velocity and acceleration using hip landmarks."""
        timestamp = time.time()
        # Use average of hips (23, 24) for overall body motion
        try:
            l_hip = next(lm for lm in landmarks if lm[0] == 23)[1:4]
            r_hip = next(lm for lm in landmarks if lm[0] == 24)[1:4]
            current_pos = (np.array(l_hip) + np.array(r_hip)) / 2.0
        except StopIteration:
            return np.zeros(3), np.zeros(3)

        lm_id = "body_center"
        if lm_id not in self.history:
            self.history[lm_id] = deque(maxlen=self.window_size)
        
        self.history[lm_id].append((current_pos, timestamp))
        
        velocity = np.zeros(3)
        acceleration = np.zeros(3)
        
        if len(self.history[lm_id]) >= 2:
            p2, t2 = self.history[lm_id][-1]
            p1, t1 = self.history[lm_id][-2]
            dt = t2 - t1
            if dt > 0:
                velocity = (p2 - p1) / dt
                
                if len(self.history[lm_id]) >= 3:
                    p0, t0 = self.history[lm_id][-3]
                    dt1 = t1 - t0
                    v1 = (p1 - p0) / dt1
                    acceleration = (velocity - v1) / dt
        
        return velocity, acceleration

    def analyze_gait_symmetry(self, landmarks):
        """
        Performs advanced gait analysis including step cycles, symmetry, hip sway, and stability.
        """
        # MediaPipe indices: 11=L Shoulder, 12=R Shoulder, 23=L Hip, 24=R Hip, 25=L Knee, 26=R Knee, 27=L Ankle, 28=R Ankle
        try:
            l_ankle = next(lm for lm in landmarks if lm[0] == 27)[1:3]
            r_ankle = next(lm for lm in landmarks if lm[0] == 28)[1:3]
            l_hip = next(lm for lm in landmarks if lm[0] == 23)[1:3]
            r_hip = next(lm for lm in landmarks if lm[0] == 24)[1:3]
            
            # --- 1. Step Length & Symmetry ---
            l_step_len = np.linalg.norm(np.array(l_ankle) - np.array(l_hip))
            r_step_len = np.linalg.norm(np.array(r_ankle) - np.array(r_hip))
            
            # Store step lengths for averaging
            self.step_history["left"].append(l_step_len)
            self.step_history["right"].append(r_step_len)
            if len(self.step_history["left"]) > 100:
                self.step_history["left"].pop(0)
                self.step_history["right"].pop(0)

            avg_l = np.mean(self.step_history["left"])
            avg_r = np.mean(self.step_history["right"])
            
            symmetry_index = 1.0 - abs(avg_l - avg_r) / (max(avg_l, avg_r) + 1e-6)
            
            # --- 2. Hip Sway Amplitude ---
            # Horizontal variation of the midpoint between hips
            hip_mid_x = (l_hip[0] + r_hip[0]) / 2
            if "hip_mid_x" not in self.history:
                self.history["hip_mid_x"] = deque(maxlen=self.window_size)
            self.history["hip_mid_x"].append(hip_mid_x)
            
            hip_sway = np.std(self.history["hip_mid_x"]) if len(self.history["hip_mid_x"]) > 5 else 0
            
            # --- 3. Walking Stability Score ---
            # Based on hip sway and symmetry
            # Ideally sway should be low but not zero, and symmetry high.
            sway_penalty = np.clip(hip_sway / 50.0, 0, 1) # Normalizing sway
            stability_score = (symmetry_index * 0.7 + (1.0 - sway_penalty) * 0.3) * 100
            
            return {
                "left_stride_length": float(avg_l),
                "right_stride_length": float(avg_r),
                "symmetry_index": float(symmetry_index),
                "hip_sway": float(hip_sway),
                "stability_score": float(np.clip(stability_score, 0, 100))
            }
        except (StopIteration, ZeroDivisionError, ValueError):
            return {
                "left_stride_length": 0.0,
                "right_stride_length": 0.0,
                "symmetry_index": 1.0,
                "hip_sway": 0.0,
                "stability_score": 100.0
            }
