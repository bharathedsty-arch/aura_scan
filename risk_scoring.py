import numpy as np

from collections import deque

class RiskScoringEngine:
    def __init__(self):
        self.risk_history = deque(maxlen=60) # 2 seconds at 30fps
        self.industrial_mode = False

    def toggle_industrial_mode(self, enabled):
        self.industrial_mode = enabled

    def calculate_risk(self, biomech_metrics, motion_analysis_results):
        """
        Computes final MSD risk score (0-100) using weighted formula.
        risk_score = 0.35 * gait_asymmetry + 0.30 * spinal_strain + 0.20 * knee_load + 0.15 * balance_instability
        """
        # Thresholds change based on mode
        SPINE_THRESHOLD = 20.0 if self.industrial_mode else 30.0
        KNEE_STRESS_MULT = 1.5 if self.industrial_mode else 1.2
        
        risk_components = {}
        alerts = []
        
        # 1. Spinal Strain
        spine_angle = biomech_metrics.get('spine_angle', 0)
        spine_load = biomech_metrics.get('spine_load', 0)
        spinal_strain_score = np.clip((spine_angle / SPINE_THRESHOLD * 50.0) + (spine_load * 0.5), 0, 100)
        risk_components['spinal_strain'] = float(spinal_strain_score)
        if spine_angle > SPINE_THRESHOLD:
            alerts.append(f"High spine bending detected ({spine_angle:.1f}°)")

        # 2. Gait Asymmetry
        symmetry_index = motion_analysis_results.get('symmetry_index', 1.0)
        gait_asymmetry_score = (1.0 - symmetry_index) * 100
        risk_components['gait_asymmetry'] = float(gait_asymmetry_score)
        if symmetry_index < 0.85:
            alerts.append("Unstable walking pattern detected")

        # 3. Knee Load
        l_knee = biomech_metrics.get('l_knee_angle', 180)
        r_knee = biomech_metrics.get('r_knee_angle', 180)
        knee_torque = max(biomech_metrics.get('l_knee_torque', 0), biomech_metrics.get('r_knee_torque', 0))
        knee_angle_stress = max(0, 180 - min(l_knee, r_knee))
        knee_load_score = np.clip((knee_angle_stress * KNEE_STRESS_MULT) + (knee_torque * 0.8), 0, 100)
        risk_components['knee_stress'] = float(knee_load_score)
        if knee_load_score > 60:
            alerts.append("Significant knee load detected")

        # 4. Balance Instability
        stability_score = motion_analysis_results.get('stability_score', 100)
        balance_instability_score = 100.0 - stability_score
        risk_components['balance_instability'] = float(balance_instability_score)
        if stability_score < 70:
            alerts.append("Walking imbalance detected")

        # --- Weighted Final Score ---
        total_score = (0.35 * gait_asymmetry_score + 
                       0.30 * spinal_strain_score + 
                       0.20 * knee_load_score + 
                       0.15 * balance_instability_score)
        
        self.risk_history.append(total_score)
        trend_info = self.analyze_risk_trend()

        # AI Explanation
        explanation = []
        if total_score > 40:
            if spinal_strain_score > 50: explanation.append(f"Excessive spine bending ({spine_angle:.0f}°)")
            if knee_load_score > 50: explanation.append("High knee joint stress")
            if gait_asymmetry_score > 40: explanation.append("Asymmetric gait pattern")

        output = {
            "risk_score": float(np.clip(total_score, 0, 100)),
            "spine_risk": float(spinal_strain_score),
            "knee_risk": float(knee_load_score),
            "gait_risk": float(gait_asymmetry_score),
            "balance_risk": float(balance_instability_score),
            "alerts": alerts,
            "trend": trend_info["trend"],
            "risk_velocity": trend_info["risk_velocity"],
            "explanation": "High risk detected due to:\n* " + "\n* ".join(explanation) if explanation else ""
        }
        
        return output["risk_score"], output

    def analyze_risk_trend(self):
        """Analyzes if risk is increasing, stable or decreasing."""
        if len(self.risk_history) < 10:
            return {"trend": "stable", "risk_velocity": 0.0}
        
        # Compare first half vs second half
        mid = len(self.risk_history) // 2
        first_half = list(self.risk_history)[:mid]
        second_half = list(self.risk_history)[mid:]
        
        v1 = np.mean(first_half)
        v2 = np.mean(second_half)
        diff = v2 - v1
        
        trend = "stable"
        if diff > 5.0: trend = "increasing"
        elif diff < -5.0: trend = "decreasing"
        
        return {"trend": trend, "risk_velocity": float(diff)}
