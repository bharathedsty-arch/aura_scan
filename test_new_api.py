import cv2
import numpy as np
from pose_detector import PoseDetector
from utils.visualization import Visualizer

def test_new_api():
    print("Initializing PoseDetector (New API)...")
    try:
        detector = PoseDetector()
        visualizer = Visualizer()
        
        # Create a blank image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        print("Running detection on blank frame...")
        results = detector.find_pose(img)
        landmarks = detector.get_landmarks(img)
        hands = detector.get_hand_landmarks(img)
        
        print(f"Detection ran. Landmarks found: {len(landmarks)}")
        print(f"Hands detected: {len(hands)}")
        
        img = visualizer.draw_skeleton(img, results)
        print("Skeleton drawing complete.")
        
        print("SUCCESS: New API is working correctly.")
    except Exception as e:
        print(f"FAILURE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_new_api()
