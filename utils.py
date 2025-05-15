# utils.py
import cv2
import mediapipe as mp
import numpy as np
import os
import json
from datetime import datetime

# Initialize MediaPipe solutions
mp_holistic = mp.solutions.holistic
# mp_drawing is not used in this file directly for processing, but good to have if needed elsewhere
# mp_drawing = mp.solutions.drawing_utils

# Define the number of landmarks to extract for consistency
# Pose: 33 landmarks * 3 coordinates (x, y, z) = 99
# Left Hand: 21 landmarks * 3 coordinates = 63
# Right Hand: 21 landmarks * 3 coordinates = 63
# Total = 99 + 63 + 63 = 225 features per frame
NUM_POSE_LANDMARKS = 33
NUM_HAND_LANDMARKS = 21 # For each hand

# Updated TOTAL_FEATURES calculation
POSE_FEATURES = NUM_POSE_LANDMARKS * 3
HAND_FEATURES = NUM_HAND_LANDMARKS * 3
TOTAL_FEATURES = POSE_FEATURES + (2 * HAND_FEATURES) # Pose + LeftHand + RightHand

def extract_landmarks(results):
    """
    Extracts Pose, Left Hand, and Right Hand landmarks into a flat numpy array.
    Face landmarks are ignored.
    """
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(POSE_FEATURES)

    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(HAND_FEATURES)

    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(HAND_FEATURES)

    # Concatenate in the order: Pose, Left Hand, Right Hand
    combined = np.concatenate([pose, lh, rh])

    # Double check size
    if len(combined) != TOTAL_FEATURES:
         print(f"Warning: Landmark feature count mismatch. Expected {TOTAL_FEATURES}, got {len(combined)}")
         # Pad with zeros if too short (shouldn't happen with np.zeros fallback)
         if len(combined) < TOTAL_FEATURES:
             combined = np.pad(combined, (0, TOTAL_FEATURES - len(combined)), 'constant', constant_values=0)
         # Truncate if too long (indicates a definition issue with constants)
         elif len(combined) > TOTAL_FEATURES:
              combined = combined[:TOTAL_FEATURES]
    return combined

def process_video_file(video_path, output_dir="data", holistic_model_instance=None):
    """
    Processes a video file, extracts selected landmarks (Pose, Hands), and saves as JSON.
    Optionally uses a pre-initialized holistic_model_instance.
    """
    sequence_data = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None

    # Use a passed-in model instance or create one if not provided
    # This is to allow app.py to manage a single holistic model instance
    if holistic_model_instance:
        holistic = holistic_model_instance
        close_holistic_on_exit = False # Don't close if it was passed in
    else:
        holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        close_holistic_on_exit = True

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break # End of video

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False # Optimize
            results = holistic.process(image_rgb)
            image_rgb.flags.writeable = True

            landmarks = extract_landmarks(results) # This now extracts 225 features
            sequence_data.append(landmarks.tolist())
    finally:
        cap.release()
        if close_holistic_on_exit:
            holistic.close()


    if not sequence_data:
        print(f"Warning: No landmarks extracted from {video_path}")
        return None

    base_filename = os.path.splitext(os.path.basename(video_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{base_filename}_{timestamp}_landmarks_{TOTAL_FEATURES}features.json" # Indicate feature count
    output_path = os.path.join(output_dir, output_filename)

    os.makedirs(output_dir, exist_ok=True)
    try:
        with open(output_path, 'w') as f:
            # Using indent=None or indent=0 makes smaller files, 
            # but indent=2 or 4 is more readable for debugging.
            # For production, consider no indent for smaller file size.
            json.dump(sequence_data, f) # No indent for compactness
        print(f"Landmark data ({TOTAL_FEATURES} features/frame) saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error saving data to {output_path}: {e}")
        return None