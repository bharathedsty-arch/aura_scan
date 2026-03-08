import numpy as np
from utils.angles import calculate_angle_2d, calculate_angle_3d

class BiomechanicsEngine:
    def __init__(self, user_weight_kg=75):
        self.user_weight = user_weight_kg

    def calculate_metrics(self, landmarks):
        metrics = {}
        
        try:
            # Landmark IDs
            L_SHOULDER, R_SHOULDER = 11, 12
            L_HIP, R_HIP = 23, 24
            L_KNEE, R_KNEE = 25, 26
            L_ANKLE, R_ANKLE = 27, 28
            
            # Helper to get coords
            get_xyz = lambda idx: next(lm[1:4] for lm in landmarks if lm[0] == idx)
            
            # --- 1. Joint Angles ---
            metrics['l_knee_angle'] = self.calculate_knee_angle(get_xyz(L_HIP), get_xyz(L_KNEE), get_xyz(L_ANKLE))
            metrics['r_knee_angle'] = self.calculate_knee_angle(get_xyz(R_HIP), get_xyz(R_KNEE), get_xyz(R_ANKLE))
            
            metrics['l_hip_angle'] = self.calculate_hip_angle(get_xyz(L_SHOULDER), get_xyz(L_HIP), get_xyz(L_KNEE))
            metrics['r_hip_angle'] = self.calculate_hip_angle(get_xyz(R_SHOULDER), get_xyz(R_HIP), get_xyz(R_KNEE))
            
            metrics['spine_angle'] = self.calculate_spine_angle(landmarks)
            
            # --- 2. Center of Mass (CoM) ---
            metrics['com'] = self.estimate_center_of_mass(landmarks)
            
            # --- 3. Torque & Load Estimation ---
            metrics['l_knee_torque'] = self.estimate_knee_torque(get_xyz(L_KNEE), get_xyz(L_ANKLE), weight_fraction=0.5)
            metrics['r_knee_torque'] = self.estimate_knee_torque(get_xyz(R_KNEE), get_xyz(R_ANKLE), weight_fraction=0.5)
            metrics['spine_load'] = self.estimate_spine_load(landmarks, metrics['com'])
            
        except (StopIteration, IndexError):
            pass
            
        return metrics

    def calculate_knee_angle(self, hip, knee, ankle):
        return calculate_angle_2d(hip, knee, ankle)

    def calculate_hip_angle(self, shoulder, hip, knee):
        return calculate_angle_2d(shoulder, hip, knee)

    def calculate_spine_angle(self, landmarks):
        get_xyz = lambda idx: next(lm[1:4] for lm in landmarks if lm[0] == idx)
        l_shoulder = get_xyz(11)
        r_shoulder = get_xyz(12)
        l_hip = get_xyz(23)
        r_hip = get_xyz(24)
        
        shoulder_mid = (np.array(l_shoulder) + np.array(r_shoulder)) / 2
        hip_mid = (np.array(l_hip) + np.array(r_hip)) / 2
        
        # Angle relative to vertical
        vertical_ref = hip_mid + np.array([0, -100, 0])
        return calculate_angle_2d(shoulder_mid, hip_mid, vertical_ref)

    def estimate_center_of_mass(self, landmarks):
        """Weighted CoM estimation based on major body segments."""
        get_xyz = lambda idx: np.array(next(lm[1:4] for lm in landmarks if lm[0] == idx))
        
        try:
            shoulder_mid = (get_xyz(11) + get_xyz(12)) / 2
            hip_mid = (get_xyz(23) + get_xyz(24)) / 2
            knee_mid = (get_xyz(25) + get_xyz(26)) / 2
            
            # Weights: Torso/Hips (0.6), Shoulders (0.2), Knees (0.2)
            com = (hip_mid * 0.6) + (shoulder_mid * 0.2) + (knee_mid * 0.2)
            return com
        except StopIteration:
            return np.mean([lm[1:4] for lm in landmarks], axis=0)

    def estimate_knee_torque(self, knee, ankle, weight_fraction=0.5):
        """Simplified torque: T = r x F. r is horizontal distance from joint to support."""
        g = 9.81
        weight_force = self.user_weight * g * weight_fraction
        lever_arm = abs(knee[0] - ankle[0]) / 100.0 # px to meters approx
        return lever_arm * weight_force

    def estimate_spine_load(self, landmarks, com):
        """Estimate load on lower spine based on CoM distance from hip center."""
        get_xyz = lambda idx: np.array(next(lm[1:4] for lm in landmarks if lm[0] == idx))
        try:
            hip_mid = (get_xyz(23) + get_xyz(24)) / 2
            g = 9.81
            upper_body_weight = self.user_weight * g * 0.6 # Approx 60% weight
            lever_arm = abs(hip_mid[0] - com[0]) / 100.0
            return lever_arm * upper_body_weight
        except StopIteration:
            return 0.0
