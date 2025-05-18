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
import torch

# Import functions from our utils.py
# Ensure utils.py is in the same directory or accessible via PYTHONPATH
from utils import extract_landmarks, process_video_file, TOTAL_FEATURES # TOTAL_FEATURES is now 225

# Import model-related components from our models package
from models import get_model, list_available_models

import mediapipe as mp

# --- Initialization ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_very_secret_key_please_change_me!' # IMPORTANT: Change this!
app.config['UPLOAD_FOLDER'] = 'uploads' # Temporary storage for uploaded videos
app.config['PROCESSED_DATA_FOLDER'] = 'data' # For JSON landmark files
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB max upload size
app.config['MODEL_NAME'] = 'mock'  # Default model to use
app.config['MODEL_PATH'] = 'resources/asl_model.pth'  # Path to model weights
app.config['CLASS_LIST_PATH'] = 'resources/wlasl_class_list.txt'  # Path to class list

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

# Global variable for the ASL model
asl_model = None

# --- Model Related Constants & Functions ---
MAX_SEQ_LENGTH = 80  # Adjusted to match the model's expected sequence length

def initialize_asl_model():
    """Initialize the ASL sign recognition model based on app configuration"""
    global asl_model
    model_name = app.config.get('MODEL_NAME', 'mock')
    class_list_path = app.config.get('CLASS_LIST_PATH', 'resources/wlasl_class_list.txt')
    
    model_kwargs = {
        'class_list_path': class_list_path
    }
    
    # Only pass model_path for the real transformer model
    if model_name == 'transformer':
        model_path = app.config.get('MODEL_PATH', 'resources/asl_model.pth')
        print(f"Initializing {model_name} model with weights from '{model_path}'")
        model_kwargs['model_path'] = model_path
    else:
        print(f"Initializing {model_name} model")
    
    # Use the model registry to get the appropriate model
    asl_model = get_model(model_name, **model_kwargs)
    
    if asl_model:
        print("ASL model loaded successfully and ready for inference.")
    else:
        print("WARNING: ASL model couldn't be loaded. Falling back to mock model.")
        # Fallback to mock model if the requested model fails to load
        asl_model = get_model('mock', class_list_path=class_list_path)

# Initialize the model at startup
initialize_asl_model()
print(f"Available models: {', '.join(list_available_models())}")

def preprocess_landmarks_for_model(raw_sequence_data):
    """Preprocess landmark data to prepare it for model input"""
    if not raw_sequence_data:
        return None
    
    # Convert to numpy array if it's a list
    raw_sequence_data = np.array(raw_sequence_data) if isinstance(raw_sequence_data, list) else raw_sequence_data
    
    # Let the model handle all preprocessing
    return raw_sequence_data

def predict_with_model(landmarks_data, top_n=5):
    """
    Make predictions using the loaded ASL model.
    
    Args:
        landmarks_data: Raw landmark sequence data
        top_n: Number of top predictions to return
    
    Returns:
        List of top predictions with labels and confidence scores
    """
    if asl_model is None:
        print("No ASL model available for prediction.")
        return [{"label": "Model not available", "confidence": 0.0}]
    
    try:
        # The model handles preprocessing, prediction, and postprocessing internally
        predictions = asl_model(landmarks_data, top_n=top_n)
        
        # Check for empty predictions
        if not predictions:
            return [{"label": "No significant motion detected", "confidence": 5.0}]
            
        # Check for empty sequence special case safely without boolean evaluation of arrays
        if isinstance(predictions, list) and len(predictions) > 0:
            # Check if first prediction is the "no motion" label
            first_pred = predictions[0]
            if isinstance(first_pred, tuple) and len(first_pred) > 0:
                if first_pred[0] == "No significant motion detected":
                    return [{"label": "No significant motion detected", "confidence": 5.0}] + [
                        {"label": label, "confidence": round(confidence * 100, 2)} 
                        for label, confidence in predictions[1:]
                    ]
        
        # Convert to the format expected by the frontend
        formatted_predictions = []
        for label, confidence in predictions:
            formatted_predictions.append({
                "label": label, 
                "confidence": round(confidence * 100, 2)  # Convert to percentage with 2 decimal places
            })
            
            print(f"Prediction: {label} with confidence {round(confidence * 100, 2)}%")
        
        if not formatted_predictions:
            return [{"label": "No significant motion detected", "confidence": 5.0}]
            
        return formatted_predictions
    except Exception as e:
        print(f"Error during model prediction: {e}")
        import traceback
        traceback.print_exc()  # Print the full stack trace for debugging
        return [{"label": f"Error during prediction", "confidence": 0.0}]

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

@app.route('/models', methods=['GET'])
def get_available_models():
    """Returns a list of available models and the currently selected model."""
    available_models = list_available_models()
    current_model = app.config.get('MODEL_NAME', 'mock')
    return jsonify({
        "available_models": available_models,
        "current_model": current_model
    })

@app.route('/change_model', methods=['POST'])
def change_model():
    """Changes the current model based on user selection."""
    data = request.get_json()
    if not data or 'model_name' not in data:
        return jsonify({"error": "No model name provided"}), 400
    
    model_name = data['model_name']
    available_models = list_available_models()
    
    if model_name not in available_models:
        return jsonify({"error": f"Model '{model_name}' not found. Available models: {', '.join(available_models)}"}), 400
    
    try:
        # Store the old model name for response
        old_model_name = app.config.get('MODEL_NAME', 'mock')
        
        # Update the app config
        app.config['MODEL_NAME'] = model_name
        
        # Re-initialize the model
        initialize_asl_model()
        
        return jsonify({
            "success": True,
            "message": f"Model changed from '{old_model_name}' to '{model_name}'",
            "previous_model": old_model_name,
            "current_model": model_name
        })
    except Exception as e:
        return jsonify({"error": f"Error changing model: {str(e)}"}), 500

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
            processed_landmarks = preprocess_landmarks_for_model(current_video_sequence_data)
            
            # Get predictions using the model
            predictions = predict_with_model(processed_landmarks)
            
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
        processed_landmarks = preprocess_landmarks_for_model(landmark_data_list)
        predictions = predict_with_model(processed_landmarks)
        
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