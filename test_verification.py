import numpy as np
from biomechanics import BiomechanicsEngine

def test_biomechanics():
    engine = BiomechanicsEngine(user_weight_kg=70)
    
    # Mock landmarks: Standing straight
    # 11: L Shoulder (100, 100, 0), 12: R Shoulder (200, 100, 0)
    # 23: L Hip (100, 300, 0), 24: R Hip (200, 300, 0)
    # 25: L Knee (100, 500, 0), 26: R Knee (200, 500, 0)
    # 27: L Ankle (100, 700, 0), 28: R Ankle (200, 700, 0)
    landmarks = [
        [11, 100, 100, 0], [12, 200, 100, 0],
        [23, 100, 300, 0], [24, 200, 300, 0],
        [25, 100, 500, 0], [26, 200, 500, 0],
        [27, 100, 700, 0], [28, 200, 700, 0]
    ]
    
    metrics = engine.calculate_metrics(landmarks)
    
    print("Biomechanics Metrics (Standing Straight):")
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v:.2f}")

    # Mock landmarks: Squatting/Leaning
    # Move hip back, bend knee
    landmarks_squat = [
        [11, 150, 150, 0], [12, 250, 150, 0],
        [23, 50, 400, 0], [24, 150, 400, 0],
        [25, 150, 600, 0], [26, 250, 600, 0],
        [27, 150, 800, 0], [28, 250, 800, 0]
    ]
    
    metrics_squat = engine.calculate_metrics(landmarks_squat)
    print("\nBiomechanics Metrics (Squat/Lean):")
    for k, v in metrics_squat.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v:.2f}")

if __name__ == "__main__":
    test_biomechanics()
