import numpy as np
import torch

# --- Landmark Constants ---
MAX_SEQ_LENGTH = 80  # Default sequence length for model input
TOTAL_FEATURES = 225  # Total number of features (3 coordinates * (33 pose + 21 left hand + 21 right hand))

def pad_or_truncate_sequence(sequence, max_len=MAX_SEQ_LENGTH, features_per_frame=TOTAL_FEATURES):
    """
    Pads or truncates a sequence to a fixed length.
    
    Args:
        sequence (np.ndarray): Input sequence of shape (timesteps, features)
        max_len (int): Target sequence length
        features_per_frame (int): Number of features per timestep
    
    Returns:
        np.ndarray: Padded/truncated sequence of shape (max_len, features_per_frame)
    """
    if sequence.size == 0:  # Handle empty sequence
        return np.zeros((max_len, features_per_frame))
    
    if len(sequence) > max_len:
        # Truncate to max_len frames (from the beginning)
        return sequence[:max_len]
    elif len(sequence) < max_len:
        # Pad with zeros to reach max_len
        padding = np.zeros((max_len - len(sequence), features_per_frame))
        return np.vstack([sequence, padding])
    else:
        return sequence

def detect_neutral_pose_and_trim(sequence_data, min_frames_neutral=10, y_threshold_normalized=0.7, velocity_threshold_normalized=0.03):
    """
    Analyzes the end of a sequence for a neutral pose (hands down, low velocity) and trims it.
    
    Args:
        sequence_data: list of landmark lists (each landmark list has TOTAL_FEATURES numbers).
        y_threshold_normalized: Normalized y-coordinate (0=top, 1=bottom). Wrists should be below this.
        velocity_threshold_normalized: Max average change in normalized x,y for wrist landmarks.
        min_frames_neutral: How many consecutive frames must be neutral.
    
    Returns:
        Trimmed sequence data
    """
    # Check if sequence_data is None or empty using .any() for numpy arrays
    if sequence_data is None or not sequence_data.any() or len(sequence_data) < min_frames_neutral:
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
            left_wrist_below_threshold = frame[LW_Y_IDX] > y_threshold_normalized
            right_wrist_below_threshold = frame[RW_Y_IDX] > y_threshold_normalized

            # Check velocity by comparing current frame with previous frame
            left_wrist_x_vel = abs(frame[LW_X_IDX] - prev_frame[LW_X_IDX])
            left_wrist_y_vel = abs(frame[LW_Y_IDX] - prev_frame[LW_Y_IDX])
            right_wrist_x_vel = abs(frame[RW_X_IDX] - prev_frame[RW_X_IDX])
            right_wrist_y_vel = abs(frame[RW_Y_IDX] - prev_frame[RW_Y_IDX])
            
            avg_velocity = (left_wrist_x_vel + left_wrist_y_vel + right_wrist_x_vel + right_wrist_y_vel) / 4
            low_velocity = avg_velocity < velocity_threshold_normalized

            # All conditions must be true to consider this frame as part of a neutral pose
            if not (left_wrist_below_threshold and right_wrist_below_threshold and low_velocity):
                is_segment_neutral = False
                break
        
        # If we found a neutral segment, trim the sequence to end just before it
        if is_segment_neutral:
            return sequence_data[:i]
    
    # No neutral segment found, return original sequence
    return sequence_data

def extract_hand_landmarks(sequence_data):
    """
    Extract hand landmarks from the full sequence data (225 features).
    
    Args:
        sequence_data: Numpy array of shape (frames, 225)
    
    Returns:
        Numpy array of shape (frames, 126) with only hand landmarks
    """
    if len(sequence_data) == 0:
        return np.array([])
    
    # Calculate indices for hand landmarks (99-224)
    hand_start = 99  # 33 pose landmarks * 3 coordinates
    hand_features = 126  # 21 left hand + 21 right hand = 42 landmarks * 3 coordinates
    
    # Extract hand features
    hand_data = np.zeros((len(sequence_data), hand_features))
    for i, frame in enumerate(sequence_data):
        if len(frame) >= hand_start + hand_features:
            hand_data[i] = frame[hand_start:hand_start+hand_features]
    
    return hand_data

def to_tensor(array_data):
    """
    Convert numpy array to PyTorch tensor.
    
    Args:
        array_data: Numpy array of landmark data
    
    Returns:
        PyTorch tensor ready for model input
    """
    if isinstance(array_data, np.ndarray):
        # Add batch dimension if not present
        if len(array_data.shape) == 2:
            array_data = np.expand_dims(array_data, axis=0)
        return torch.tensor(array_data, dtype=torch.float32)
    return array_data 