# 🧬 AURA-SCAN AI
### Advanced Industrial Biomechanics & Ergonomic HUD

Aura-Scan is an Edge AI prototype that predicts early musculoskeletal disorder (MSD) risks in real-time using pose estimation and high-fidelity biomechanics modeling.

---

## ✨ Features
- **[NEW] Holistic Scanning**: 75-point body and hand tracking.
- **Sci-Fi HUD**: Real-time motion trails, heatmap glows, and dynamic scanning effects.
- **Biomechanics Node**: Live calculation of torque, angles, and gait symmetry.
- **Industrial Protocols**: One-click safety mode for factory-floor compliance.
- **Explainable AI**: Real-time reasoning panels for decision transparency.

> [!TIP]
> **Dive Deeper**: Check out [project_overview.md](project_overview.md) for a full technical breakdown, future roadmaps (Thermal Detection), and system architecture.

---

## 🚀 Setup Instructions

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Application:**
   ```bash
   python -m streamlit run app.py
   ```

---

## 📁 System Architecture
- `app.py`: Futuristic Streamlit HUD implementation.
- `pose_detector.py`: Holistic (Pose + Hand) MediaPipe integration.
- `motion_analysis.py`: Spatio-temporal gait tracking.
- `biomechanics.py`: Vector-based skeletal physics engine.
- `risk_scoring.py`: Ergonomic risk aggregation & trending.
- `posture_classifier.py`: AI-driven ergonomic state classification.
- `utils/visualization.py`: Sci-Fi visual effects & HUD logic.

---
**Built for Industrial Safety // Empowering the Modern Worker**
