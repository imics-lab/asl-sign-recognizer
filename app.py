# app.py
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import os
import cv2
import numpy as np
import base64
import json # For loading/saving JSON and for mock model
import random # For mock model
from datetime import datetime # For unique filenames

# Import functions from our utils.py
# Ensure utils.py is in the same directory or accessible via PYTHONPATH
from utils import extract_landmarks, process_video_file, TOTAL_FEATURES # TOTAL_FEATURES is now 225

import mediapipe as mp

# --- Initialization ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_very_secret_key_please_change_me!' # IMPORTANT: Change this!
app.config['UPLOAD_FOLDER'] = 'uploads' # Temporary storage for uploaded videos
app.config['PROCESSED_DATA_FOLDER'] = 'data' # For JSON landmark files
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB max upload size

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_DATA_FOLDER'], exist_ok=True)

# Initialize SocketIO
socketio = SocketIO(app, async_mode='eventlet') # Using eventlet

# Initialize MediaPipe Holistic Model ONCE for the entire application
# This is much more efficient than re-initializing for each request/frame.
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
print(f"MediaPipe Holistic model initialized. Expecting {TOTAL_FEATURES} features per frame.")

# --- WLASL Class List Loading ---
WLASL_CLASSES = []
def load_wlasl_class_list(filepath="resources/wlasl_class_list.txt"):
    global WLASL_CLASSES
    try:
        with open(filepath, 'r', encoding='utf-8') as f: # Added encoding
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    WLASL_CLASSES.append(parts[1]) # Store only the name
        if WLASL_CLASSES:
            print(f"Successfully loaded {len(WLASL_CLASSES)} classes from {filepath}")
        else:
            print(f"Warning: {filepath} was read, but no classes were loaded. Check file format.")
            WLASL_CLASSES = [f"Sign_{i}" for i in range(2000)] # Fallback
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Mock model will use generic labels.")
        WLASL_CLASSES = [f"Sign_{i}" for i in range(2000)] # Fallback
    except Exception as e:
        print(f"An error occurred while loading {filepath}: {e}")
        WLASL_CLASSES = [f"Sign_{i}" for i in range(2000)] # Fallback

load_wlasl_class_list() # Load at application startup

# --- Model Related Constants & Functions ---
MAX_SEQ_LENGTH = 100 # Number of frames the model expects

def detect_neutral_pose_and_trim(sequence_data, min_frames_neutral=10, y_threshold_normalized=0.7, velocity_threshold_normalized=0.03):
    """
    Analyzes the end of a sequence for a neutral pose (hands down, low velocity) and trims it.
    sequence_data: list of landmark lists (each landmark list has TOTAL_FEATURES numbers).
    y_threshold_normalized: Normalized y-coordinate (0=top, 1=bottom). Wrists should be below this.
    velocity_threshold_normalized: Max average change in normalized x,y for wrist landmarks.
    min_frames_neutral: How many consecutive frames must be neutral.
    """
    if not sequence_data or len(sequence_data) < min_frames_neutral:
        return sequence_data

    # Indices for Pose landmarks (from MediaPipe Holistic documentation):
    # LEFT_WRIST = 15, RIGHT_WRIST = 16
    # In our flat array (Pose, LH, RH), Pose landmarks are first.
    # x, y, z for each.
    LW_X_IDX, LW_Y_IDX = 15 * 3, 15 * 3 + 1
    RW_X_IDX, RW_Y_IDX = 16 * 3, 16 * 3 + 1

    # Iterate backwards from a point that allows a full neutral segment to be checked
    for i in range(len(sequence_data) - min_frames_neutral, -1, -1):
        is_segment_neutral = True
        for j in range(min_frames_neutral):
            current_frame_idx_in_segment = i + j
            frame = sequence_data[current_frame_idx_in_segment]
            # Use current frame as prev_frame if it's the first frame of the segment AND sequence
            prev_frame = sequence_data[current_frame_idx_in_segment - 1] if current_frame_idx_in_segment > 0 else frame

            # Check Y position of wrists (assuming normalized 0.0 at top, 1.0 at bottom of frame)
            left_wrist_y = frame[LW_Y_IDX]
            right_wrist_y = frame[RW_Y_IDX]

            # Hands DOWN: y-coordinate should be larger (further down the screen)
            if left_wrist_y < y_threshold_normalized or right_wrist_y < y_threshold_normalized:
                is_segment_neutral = False
                break
            
            # Check velocity (simplified: change from previous frame for wrists)
            lw_vel = abs(frame[LW_X_IDX] - prev_frame[LW_X_IDX]) + abs(frame[LW_Y_IDX] - prev_frame[LW_Y_IDX])
            rw_vel = abs(frame[RW_X_IDX] - prev_frame[RW_X_IDX]) + abs(frame[RW_Y_IDX] - prev_frame[RW_Y_IDX])

            if lw_vel > velocity_threshold_normalized or rw_vel > velocity_threshold_normalized:
                is_segment_neutral = False
                break
        
        if is_segment_neutral:
            print(f"Neutral pose detected starting at original frame index {i}. Trimming sequence.")
            return sequence_data[:i] # Trim up to the start of the neutral segment

    print("No clear neutral pose detected at the end. Using full (or previously trimmed) sequence.")
    return sequence_data


def preprocess_landmarks_for_model(raw_sequence_data, max_len=MAX_SEQ_LENGTH, features_per_frame=TOTAL_FEATURES):
    """
    Preprocesses a list of landmark lists:
    1. Trims based on neutral pose detection at the end.
    2. Pads or truncates the sequence to max_len.
    Returns a list of lists, ready for the model.
    """
    # 1. Trim based on neutral pose detection (if enabled and configured)
    # For now, let's assume it's always active.
    trimmed_sequence = detect_neutral_pose_and_trim(raw_sequence_data)
    
    # 2. Pad or Truncate to max_len
    processed_sequence_np = np.zeros((max_len, features_per_frame), dtype=np.float32)
    
    num_frames = len(trimmed_sequence)
    
    if num_frames > 0:
        actual_frames_to_copy = min(num_frames, max_len)
        
        # Convert to numpy array for slicing and assignment
        # Ensure the inner lists are consistently `features_per_frame` long
        # This should be guaranteed by extract_landmarks
        try:
            trimmed_array = np.array(trimmed_sequence[:actual_frames_to_copy], dtype=np.float32)
            if trimmed_array.shape[1] != features_per_frame:
                # This case should ideally not happen if extract_landmarks is correct
                print(f"Warning: Feature count mismatch in trimmed_array. Expected {features_per_frame}, got {trimmed_array.shape[1]}. Adjusting.")
                # Create a correctly shaped temp array and copy what we can
                temp_array = np.zeros((trimmed_array.shape[0], features_per_frame), dtype=np.float32)
                copy_cols = min(trimmed_array.shape[1], features_per_frame)
                temp_array[:, :copy_cols] = trimmed_array[:, :copy_cols]
                trimmed_array = temp_array

            processed_sequence_np[:actual_frames_to_copy, :] = trimmed_array
            
            if num_frames > max_len:
                print(f"Sequence truncated from {num_frames} (after trim) to {max_len} frames.")
            elif num_frames < max_len:
                print(f"Sequence padded from {num_frames} (after trim) to {max_len} frames.")
            # else: sequence length matched max_len after trimming

        except ValueError as ve:
            print(f"ValueError during numpy array conversion or assignment: {ve}")
            print("This might be due to inconsistent feature counts in frames of raw_sequence_data.")
            # Fallback to returning zeros, or handle error more gracefully
            return processed_sequence_np.tolist()
    else:
        print("Sequence is empty after trimming (or was empty initially). Returning zero-padded sequence.")

    return processed_sequence_np.tolist()


def mock_pytorch_model(landmark_sequence_processed):
    """
    Simulates a PyTorch model prediction.
    landmark_sequence_processed: A list of lists (max_len x TOTAL_FEATURES).
    Returns a list of top-5 predictions (dict with "label" and "confidence").
    """
    # print(f"Mock model received processed sequence with {len(landmark_sequence_processed)} frames.")
    
    num_classes = len(WLASL_CLASSES)
    if num_classes == 0:
        return [{"label": "Error: Class list not loaded", "confidence": 0.0}]

    # Simulate some processing or check if sequence is mostly zeros
    # For a more "realistic" mock, check if there's any non-zero data
    is_empty_sequence = not np.any(landmark_sequence_processed) # True if all zeros
    
    predictions = []
    
    if is_empty_sequence:
        # If the sequence is empty (all zeros after padding), return low confidence
        predictions.append({
            "label": "No significant motion detected",
            "confidence": 5.0
        })
        # Fill remaining with random low confidence
        chosen_indices = random.sample(range(num_classes), min(4, num_classes))
        for class_idx in chosen_indices:
            predictions.append({
                "label": WLASL_CLASSES[class_idx],
                "confidence": round(random.uniform(0.1, 2.0), 2)
            })
        return predictions


    # Generate 5 random predictions for non-empty sequences
    chosen_indices = random.sample(range(num_classes), min(5, num_classes))
    confidences = sorted([random.random() for _ in range(len(chosen_indices))], reverse=True)
    
    for i, class_idx in enumerate(chosen_indices):
        predictions.append({
            "label": WLASL_CLASSES[class_idx],
            "confidence": round(confidences[i] * 100, 2) # As percentage
        })
    return predictions

# --- Routes ---
@app.route('/')
def index():
    """Serves the main sign recognition page."""
    return render_template('index.html')

@app.route('/landmark_extractor')
def landmark_extractor_page():
    """Serves the dedicated landmark extraction page."""
    return render_template('landmark_extractor.html')

@app.route('/playback')
def playback_page():
    """Serves the landmark playback page."""
    return render_template('playback.html')

@app.route('/upload_and_predict', methods=['POST'])
def upload_and_predict_video():
    """Handles video file uploads, processes them, and returns predictions."""
    if 'video' not in request.files:
        return jsonify({"error": "No video file part"}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Secure filename and create path
        # filename = secure_filename(file.filename) # Good practice
        filename = f"upload_{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        raw_landmark_data_for_playback = None
        playback_filename = None

        try:
            file.save(video_path)
            print(f"Video file saved temporarily to {video_path}")

            # Process the video to get raw landmark sequence (list of lists)
            # For this, we cannot use process_video_file directly if we want the raw list
            # Let's extract sequence data here.
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"Could not open video file {video_path} for processing.")

            current_video_sequence_data = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results = holistic_model.process(image_rgb)
                image_rgb.flags.writeable = True
                landmarks = extract_landmarks(results)
                current_video_sequence_data.append(landmarks.tolist())
            cap.release()

            if not current_video_sequence_data:
                os.remove(video_path) # Clean up
                return jsonify({"error": "No landmarks extracted from the video"}), 500

            # Save the raw landmarks for playback
            raw_landmark_data_for_playback = current_video_sequence_data
            playback_base_filename = os.path.splitext(filename)[0]
            playback_filename_stem = f"{playback_base_filename}_raw_landmarks_{TOTAL_FEATURES}f.json"
            playback_filepath = os.path.join(app.config['PROCESSED_DATA_FOLDER'], playback_filename_stem)
            with open(playback_filepath, 'w') as f_pb:
                json.dump(raw_landmark_data_for_playback, f_pb) # No indent for compactness
            playback_filename = playback_filename_stem # Filename to send to client

            # Preprocess for the model
            processed_landmarks = preprocess_landmarks_for_model(current_video_sequence_data, MAX_SEQ_LENGTH, TOTAL_FEATURES)
            
            # Get predictions
            predictions = mock_pytorch_model(processed_landmarks)
            
            os.remove(video_path) # Clean up uploaded file

            return jsonify({
                "message": "Video processed successfully!",
                "predictions": predictions,
                "playback_file": playback_filename # URL will be /data/<playback_filename>
            }), 200

        except Exception as e:
            if os.path.exists(video_path):
                os.remove(video_path)
            print(f"Error during video upload/processing for prediction: {e}")
            return jsonify({"error": f"An error occurred: {str(e)}"}), 500

    return jsonify({"error": "Invalid file"}), 400


@app.route('/upload_for_extraction', methods=['POST'])
def upload_for_extraction():
    """Handles video file uploads FOR THE LANDMARK EXTRACTOR PAGE."""
    if 'video' not in request.files:
        return jsonify({"error": "No video file part"}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # filename = secure_filename(file.filename)
        filename = f"extract_{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(video_path)
            # Process the video using the utility function, passing the global holistic model
            output_json_path = process_video_file(video_path, output_dir=app.config['PROCESSED_DATA_FOLDER'], holistic_model_instance=holistic_model)
            os.remove(video_path) # Clean up uploaded file

            if output_json_path:
                return jsonify({
                    "message": "Video processed successfully for landmark extraction!",
                    "filename": os.path.basename(output_json_path) # Client will use this for download link
                }), 200
            else:
                return jsonify({"error": "Failed to process video or extract landmarks"}), 500
        except Exception as e:
            if os.path.exists(video_path):
                os.remove(video_path)
            print(f"Error during upload/processing for extraction: {e}")
            return jsonify({"error": f"An error occurred: {str(e)}"}), 500
    return jsonify({"error": "Invalid file"}), 400


@app.route('/data/<filename>')
def serve_data(filename):
    """Serves files from the PROCESSED_DATA_FOLDER (e.g., landmark JSONs)."""
    try:
        return send_from_directory(app.config['PROCESSED_DATA_FOLDER'], filename, as_attachment=True)
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404


# --- SocketIO Events for Webcam (for both pages, if needed, or can be namespaced) ---
# For the new index.html (recognition page)
@socketio.on('live_frame_for_recognition')
def handle_live_frame_for_recognition(data):
    """Receives a frame from client, extracts landmarks, sends them back. No prediction here."""
    try:
        img_data = base64.b64decode(data.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            emit('live_landmarks_result', {'landmarks': [], 'error': 'Empty frame received by server'})
            return

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = holistic_model.process(image_rgb)
        image_rgb.flags.writeable = True
        
        landmarks = extract_landmarks(results) # Uses updated extract_landmarks (225 features)
        emit('live_landmarks_result', {'landmarks': landmarks.tolist()})

    except Exception as e:
        print(f"Error processing live frame for recognition: {e}")
        emit('live_landmarks_result', {'landmarks': [], 'error': str(e)})


@socketio.on('predict_webcam_sequence')
def handle_predict_webcam_sequence(landmark_data_list):
    """
    Receives a complete sequence of landmarks collected from webcam.
    Processes it and returns predictions.
    landmark_data_list: A list of landmark lists.
    """
    try:
        if not landmark_data_list or not isinstance(landmark_data_list, list):
            emit('webcam_prediction_result', {"error": "Invalid or empty landmark data received."})
            return

        # Save raw data for playback
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        playback_filename_stem = f"webcam_capture_{timestamp}_raw_landmarks_{TOTAL_FEATURES}f.json"
        playback_filepath = os.path.join(app.config['PROCESSED_DATA_FOLDER'], playback_filename_stem)
        with open(playback_filepath, 'w') as f_pb:
            json.dump(landmark_data_list, f_pb)
        
        # Preprocess for the model
        processed_landmarks = preprocess_landmarks_for_model(landmark_data_list, MAX_SEQ_LENGTH, TOTAL_FEATURES)
        predictions = mock_pytorch_model(processed_landmarks)
        
        emit('webcam_prediction_result', {
            "predictions": predictions,
            "playback_file": playback_filename_stem
        })

    except Exception as e:
        print(f"Error in predict_webcam_sequence: {e}")
        emit('webcam_prediction_result', {"error": f"Server error during prediction: {str(e)}"})


# For landmark_extractor.html (original functionality)
@socketio.on('process_frame_for_extraction') # Renamed from 'process_frame'
def handle_process_frame_for_extraction(data):
    """Receives a frame from the landmark_extractor client, processes it, sends landmarks back."""
    try:
        img_data = base64.b64decode(data.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            emit('extraction_frame_result', {'landmarks': [], 'error': 'Empty frame received'}) # Renamed event
            return

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = holistic_model.process(image_rgb)
        image_rgb.flags.writeable = True
        
        landmarks = extract_landmarks(results) # Uses updated extract_landmarks
        emit('extraction_frame_result', {'landmarks': landmarks.tolist()}) # Renamed event

    except Exception as e:
        print(f"Error processing frame for extraction: {e}")
        emit('extraction_frame_result', {'landmarks': [], 'error': str(e)})


@socketio.on('connect')
def handle_connect():
    print(f'Client connected: {request.sid}')

@socketio.on('disconnect')
def handle_disconnect():
    print(f'Client disconnected: {request.sid}')


# --- Main Execution ---
if __name__ == '__main__':
    print("Starting Flask server with SocketIO support...")
    # Eventlet is often preferred for production SocketIO
    # For development, debug=True is helpful.
    # Ensure eventlet is installed: pip install eventlet
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=True)