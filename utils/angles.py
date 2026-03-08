import numpy as np

def calculate_angle_2d(a, b, c):
    """Calculates the 2D angle between three points (a, b, c)."""
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def calculate_angle_3d(a, b, c):
    """Calculates the 3D angle between three points (a, b, c)."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    v1 = a - b
    v2 = c - b
    
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)
