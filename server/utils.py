# Moved from root utils.py to server/utils.py without changes
import cv2
import mediapipe as mp
import numpy as np
import os
import json
from datetime import datetime

mp_holistic = mp.solutions.holistic

NUM_POSE_LANDMARKS = 33
NUM_HAND_LANDMARKS = 21

POSE_FEATURES = NUM_POSE_LANDMARKS * 3
HAND_FEATURES = NUM_HAND_LANDMARKS * 3
TOTAL_FEATURES = POSE_FEATURES + (2 * HAND_FEATURES)

def extract_landmarks(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(POSE_FEATURES)

    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(HAND_FEATURES)

    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(HAND_FEATURES)

    combined = np.concatenate([pose, lh, rh])

    if len(combined) != TOTAL_FEATURES:
         print(f"Warning: Landmark feature count mismatch. Expected {TOTAL_FEATURES}, got {len(combined)}")
         if len(combined) < TOTAL_FEATURES:
             combined = np.pad(combined, (0, TOTAL_FEATURES - len(combined)), 'constant', constant_values=0)
         elif len(combined) > TOTAL_FEATURES:
              combined = combined[:TOTAL_FEATURES]
    return combined

def process_video_file(video_path, output_dir="data", holistic_model_instance=None):
    sequence_data = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None

    if holistic_model_instance:
        holistic = holistic_model_instance
        close_holistic_on_exit = False
    else:
        holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        close_holistic_on_exit = True

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = holistic.process(image_rgb)
            image_rgb.flags.writeable = True

            landmarks = extract_landmarks(results)
            sequence_data.append(landmarks.tolist())
    finally:
        cap.release()
        if close_holistic_on_exit:
            holistic.close()

    if not sequence_data:
        print(f"Warning: No landmarks extracted from {video_path}")
        return None

    base_filename = os.path.splitext(os.path.basename(video_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_filename = f"{base_filename}_{timestamp}_landmarks_{TOTAL_FEATURES}features.json"
    output_path = os.path.join(output_dir, output_filename)

    os.makedirs(output_dir, exist_ok=True)
    try:
        with open(output_path, 'w') as f:
            json.dump(sequence_data, f)
        print(f"Landmark data ({TOTAL_FEATURES} features/frame) saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error saving data to {output_path}: {e}")
        return None


