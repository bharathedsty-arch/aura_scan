import streamlit as st
import cv2
import tempfile
import time
import numpy as np
from pose_detector import PoseDetector
from motion_analysis import MotionAnalyzer
from biomechanics import BiomechanicsEngine
from risk_scoring import RiskScoringEngine
from posture_classifier import PostureClassifier
from utils.visualization import Visualizer

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AURA-SCAN AI | BIOMECH HUD",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THEME STATE ---
if 'industrial_theme' not in st.session_state:
    st.session_state.industrial_theme = False

def update_theme():
    st.session_state.industrial_theme = st.session_state.ind_toggle

# --- SCI-FI HUD CSS ---
theme_color = "#FF9800" if st.session_state.industrial_theme else "#00E5FF"
bg_gradient = "radial-gradient(circle, #2C1E0A 0%, #0B0F1A 100%)" if st.session_state.industrial_theme else "radial-gradient(circle, #1A1C24 0%, #0B0F1A 100%)"

st.markdown(f"""
<style>
    .main {{ background-color: #0B0F1A; color: #ffffff; font-family: 'Courier New', Courier, monospace; }}
    .stApp {{ background: {bg_gradient}; }}
    
    /* Neon Glass Panels */
    .stMetric {{ 
        background: rgba(11, 15, 26, 0.8); 
        padding: 20px; 
        border-radius: 0px; 
        border: 1px solid {theme_color};
        box-shadow: 0 0 10px {theme_color}44;
    }}
    
    .alert-box {{ 
        padding: 15px; 
        border: 1px solid #FF3B5C; 
        background: rgba(255, 59, 92, 0.1); 
        color: #FF3B5C;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: bold;
    }}
    
    .summary-card {{ 
        background: rgba(0, 229, 255, 0.05); 
        padding: 25px; 
        border-left: 4px solid {theme_color};
        border-right: 1px solid {theme_color}33;
        border-top: 1px solid {theme_color}33;
        border-bottom: 1px solid {theme_color}33;
    }}
    
    h1, h2, h3 {{ color: {theme_color} !important; text-transform: uppercase; letter-spacing: 3px; }}
    
    .fps-text {{ font-size: 0.8em; color: {theme_color}; opacity: 0.7; float: right; }}
</style>
""", unsafe_allow_html=True)

@st.cache_resource # v2
def get_engines():
    return (
        PoseDetector(),
        MotionAnalyzer(),
        BiomechanicsEngine(),
        RiskScoringEngine(),
        PostureClassifier(),
        Visualizer()
    )

def main():
    detector, motion, biomech, risk_engine, classifier, visualizer = get_engines()
    
    # Header HUD
    cols = st.columns([2, 1])
    with cols[0]:
        st.title("🧬 AURA-SCAN // BIOMECH HUD")
        st.caption("SYSTEM STATUS: ONLINE // AI CORE: ACTIVE")
    with cols[1]:
        fps_placeholder = st.empty()

    # --- SIDEBAR HUD ---
    with st.sidebar:
        st.header("⚡ NODE_SETTINGS")
        industrial_mode = st.toggle("🏭 INDUSTRIAL_PROTOCOLS", key="ind_toggle", on_change=update_theme)
        risk_engine.toggle_industrial_mode(industrial_mode)
        
        st.divider()
        source = st.radio("DATA_INPUT_STREAM", ["LIVE_WEBCAM", "ARCHIVE_FILE"])
        uploaded_file = None
        if source == "ARCHIVE_FILE":
            uploaded_file = st.file_uploader("UPLOAD_BYTE_STREAM", type=["mp4", "mov", "avi"])
        
        st.divider()
        if industrial_mode:
            st.warning("SAFETY_MODE: ENFORCED // THRESHOLDS: CRITICAL")
        else:
            st.info("NORMAL_OPS: ACTIVE // THRESHOLDS: STANDARD")

    # --- MAIN HUD LAYOUT ---
    main_container = st.container()
    
    with main_container:
        st.write("### 🖥️ PRIMARY_VIDEO_ANALYSIS")
        placeholder = st.empty()
        alert_placeholder = st.empty()

    st.divider()
    
    st.write("### 📊 DECISION_intelligence_GRID")
    dash_col1, dash_col2 = st.columns([1, 1])
    
    with dash_col1:
        score_placeholder = st.empty()
        posture_placeholder = st.empty()

    with dash_col2:
        metrics_placeholder = st.empty()
        explanation_placeholder = st.empty()
        
    summary_placeholder = st.empty()

    # --- EXECUTION LOOP ---
    btn_label = "⚡ INITIATE BIOMECHANICAL SCAN"
    if st.button(btn_label, key="scan_btn", width="stretch"):
        cap = None
        if source == "LIVE_WEBCAM":
            cap = cv2.VideoCapture(0)
        elif source == "ARCHIVE_FILE" and uploaded_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
        
        if cap and cap.isOpened():
            prev_time = time.time()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # Calc FPS
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time)
                prev_time = curr_time
                fps_placeholder.markdown(f'<p class="fps-text">FPS: {fps:.1f} // CPU_LOAD: OPTIMAL</p>', unsafe_allow_html=True)

                frame = cv2.resize(frame, (640, 480))
                
                # 1. Detection (Holistic)
                results = detector.find_pose(frame)
                landmarks = detector.get_landmarks(frame)
                hand_landmarks = detector.get_hand_landmarks(frame)
                
                if landmarks:
                    # 2. Analysis
                    velocity, _ = motion.get_body_velocity(landmarks)
                    gait_analysis = motion.analyze_gait_symmetry(landmarks)
                    biomech_metrics = biomech.calculate_metrics(landmarks)
                    
                    # 3. Posture Classification
                    posture_info = classifier.classify_posture(biomech_metrics, gait_analysis)
                    
                    # 4. Risk Scoring
                    risk_score, risk_data = risk_engine.calculate_risk(biomech_metrics, gait_analysis)
                    
                    # 5. Advanced Visualization
                    frame = visualizer.draw_skeleton(frame, results, risk_data)
                    frame = visualizer.highlight_stress(frame, landmarks, risk_data)
                    frame = visualizer.overlay_metrics(frame, risk_score, risk_data)
                    
                    # Display HUD
                    placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                    
                    # --- HUD UPDATES ---
                    # 1. Alerts (Futuristic)
                    if risk_data['alerts']:
                        alert_html = "".join([f'<div class="alert-box">CRITICAL: {a}</div>' for a in risk_data['alerts'][:2]])
                        alert_placeholder.markdown(alert_html, unsafe_allow_html=True)
                    else:
                        alert_placeholder.empty()

                    # 2. Score Meter
                    score_color = '#FF3B5C' if risk_score > 70 else '#FFD600' if risk_score > 35 else '#00E5FF'
                    score_placeholder.markdown(f"""
                        <div class="stMetric" style="border-top: 5px solid {score_color}">
                            <p style='margin: 0; opacity: 0.6;'>CUMULATIVE_RISK_INDEX</p>
                            <h1 style='color: {score_color}; margin: 10px 0;'>{risk_score:.1f}%</h1>
                            <progress value="{risk_score}" max="100" style="width: 100%;"></progress>
                        </div>
                    """, unsafe_allow_html=True)

                    # 3. Posture HUD
                    posture_placeholder.markdown(f"""
                        <div class="stMetric">
                            <p style='margin: 0; opacity: 0.6;'>POSTURE_CLASSIFICATION</p>
                            <h3 style='margin: 10px 0;'>{posture_info['posture_type'].upper()}</h3>
                            <p style='font-size: 0.8em;'>CONFIDENCE: {posture_info['confidence']*100:.1f}%</p>
                        </div>
                    """, unsafe_allow_html=True)

                    # 4. AI Explanation (Decision Logic)
                    with explanation_placeholder.container():
                        st.markdown(f"""
                        <div class="summary-card">
                            <h4 style='margin-top:0;'>🧠 AI_DECISION_REASONING</h4>
                            <p style='font-size: 0.9em; line-height: 1.4;'>{risk_data['explanation']}</p>
                            <ul style='font-size: 0.8em; opacity: 0.8;'>
                                <li>Spine Angle Performance: {'OPTIMAL' if biomech_metrics['spine_angle'] < 20 else 'STRESSED'}</li>
                                <li>Gait Symmetry Index: {gait_analysis['symmetry_index']:.2f}</li>
                                <li>Joint Load Distribution: {'BALANCED' if risk_data['knee_risk'] < 50 else 'ASYMMETRIC'}</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)

                    # 5. Dashboard Grid
                    with metrics_placeholder.container():
                        m_col1, m_col2 = st.columns(2)
                        m_col1.metric("SPINE_DEG", f"{biomech_metrics['spine_angle']:.0f}°")
                        m_col2.metric("SYM_INDEX", f"{gait_analysis['symmetry_index']:.2f}")
                        m_col1.metric("MAX_TORQUE", f"{max(biomech_metrics['l_knee_torque'], biomech_metrics['r_knee_torque']):.1f}Nm")
                        m_col2.metric("STABILITY", f"{gait_analysis['stability_score']:.0f}%")

                    # 6. Global Summary
                    summary_placeholder.markdown(f"""
                        <div style='background: {theme_color}22; padding: 15px; border: 1px dashed {theme_color}; margin-top: 20px;'>
                            <h5 style='margin: 0; color: {theme_color}'>BIOMECH_ANALYTICS_SUMMARY</h5>
                            <p style='font-size: 0.8em; margin: 5px 0;'>Automated health assessment suggests: <b>{ "URGENT INTERVENTION REQUIRED" if risk_score > 70 else "MONITORING CONTINUED" if risk_score > 35 else "ALL SYSTEMS NOMINAL"}</b></p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                    alert_placeholder.warning("NO_ENTITY_DETECTED // SCANNING_ENVIRONMENT...")

            cap.release()
        else:
            st.error("STREAM_UNAVAILABLE // CHECK_HARDWARE_NODE")

if __name__ == "__main__":
    main()
