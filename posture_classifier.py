import numpy as np

class PostureClassifier:
    def __init__(self):
        self.posture_types = [
            "Normal", 
            "Forward Bend", 
            "Knee Stress", 
            "Asymmetric Walk", 
            "Leaning"
        ]

    def classify_posture(self, biomech_metrics, gait_analysis):
        """
        Classifies worker posture based on biomechanics and gait metrics.
        Returns posture type and confidence.
        """
        spine_angle = biomech_metrics.get('spine_angle', 0)
        l_knee = biomech_metrics.get('l_knee_angle', 180)
        r_knee = biomech_metrics.get('r_knee_angle', 180)
        symmetry = gait_analysis.get('symmetry_index', 1.0)
        hip_sway = gait_analysis.get('hip_sway', 0)
        
        # Classification Logic
        scores = {
            "Forward Bend": np.clip(spine_angle / 45.0, 0, 1),
            "Knee Stress": np.clip((180 - min(l_knee, r_knee)) / 60.0, 0, 1),
            "Asymmetric Walk": np.clip((1.0 - symmetry) * 2.0, 0, 1),
            "Leaning": np.clip(hip_sway / 100.0, 0, 1),
            "Normal": 1.0
        }
        
        # Calculate Normal score as inverse of other risks
        other_risks = [scores["Forward Bend"], scores["Knee Stress"], scores["Asymmetric Walk"]]
        scores["Normal"] = max(0, 1.0 - max(other_risks))
        
        # Find the max score
        posture_type = max(scores, key=scores.get)
        confidence = float(scores[posture_type])
        
        return {
            "posture_type": posture_type,
            "confidence": confidence
        }
