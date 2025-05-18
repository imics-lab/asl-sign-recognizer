"""
Transformer-based model implementation for ASL sign recognition.

This module provides a transformer-based model for recognizing ASL signs
from sequences of body landmarks.
"""

import torch
import torch.nn as nn
import numpy as np
from .base_model import ASLModel
from .utils import pad_or_truncate_sequence, extract_hand_landmarks, to_tensor, MAX_SEQ_LENGTH

class TransformerClassifier(nn.Module):
    """
    Transformer-based model for ASL sign recognition.
    """
    def __init__(self, input_dim, num_classes, num_heads=8, num_layers=4, hidden_dim=256, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Transformer encoder layer
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=input_dim,  # Input dimension (features per frame)
            nhead=num_heads,    # Number of attention heads
            dim_feedforward=hidden_dim,  # Feedforward hidden layer size
            dropout=dropout
        )
        
        # Stacked transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers
        )
        
        # Classifier head
        self.fc = nn.Linear(input_dim, num_classes)  # Final layer to output class probabilities
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Tensor of shape (batch_size, time_steps, features)
            
        Returns:
            Tensor of shape (batch_size, num_classes)
        """
        # Transformer expects input of shape (sequence_length, batch_size, input_dim)
        x = x.permute(1, 0, 2)  # Shape: (time_steps, batch_size, features)
        
        # Apply transformer encoder
        transformer_out = self.transformer_encoder(x)
        
        # We take the output of the last time step for classification
        x = transformer_out[-1, :, :]  # Shape: (batch_size, features)
        
        # Classifier head to predict the class
        x = self.fc(x)
        return x

class TransformerModel(ASLModel):
    """
    Implementation of the ASLModel interface using a transformer architecture.
    """
    def __init__(self, model_path=None, class_list_path=None):
        """
        Initialize the transformer model.
        
        Args:
            model_path (str): Path to the saved model weights
            class_list_path (str): Path to the file containing class labels
        """
        self.input_dim = 126  # Number of features (only hand landmarks)
        self.num_classes = 2000  # Default number of classes
        self.num_heads = 9  # As specified in training
        
        # Load class list if provided
        self.classes = self._load_class_list(class_list_path)
        
        # Initialize the model
        self.model = TransformerClassifier(
            input_dim=self.input_dim,
            num_classes=len(self.classes),
            num_heads=self.num_heads
        )
        
        # Load model weights if provided
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                print(f"Successfully loaded model from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Using untrained model - predictions will be random")
        
        self.model.eval()  # Set to evaluation mode
    
    def _load_class_list(self, filepath="resources/wlasl_class_list.txt"):
        """
        Load the list of class labels.
        
        Args:
            filepath (str): Path to the file containing class labels
            
        Returns:
            List of class labels
        """
        classes = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        classes.append(parts[1])  # Store only the name
            if classes:
                print(f"Successfully loaded {len(classes)} classes from {filepath}")
            else:
                print(f"Warning: {filepath} was read, but no classes were loaded. Using generic labels.")
                classes = [f"Sign_{i}" for i in range(2000)]  # Fallback
        except FileNotFoundError:
            print(f"Error: {filepath} not found. Using generic labels.")
            classes = [f"Sign_{i}" for i in range(2000)]  # Fallback
        except Exception as e:
            print(f"An error occurred while loading {filepath}: {e}")
            classes = [f"Sign_{i}" for i in range(2000)]  # Fallback
        
        return classes
    
    def preprocess(self, landmarks):
        """
        Preprocess landmarks for the transformer model.
        
        Args:
            landmarks (list or numpy array): Raw landmark sequences
            
        Returns:
            PyTorch tensor ready for model input
        """
        # Convert to numpy array if it's a list
        if isinstance(landmarks, list):
            landmarks = np.array(landmarks)
        
        # Trim neutral poses
        from .utils import detect_neutral_pose_and_trim
        landmarks = detect_neutral_pose_and_trim(landmarks)
        
        # Extract hand landmarks (model was trained on hand landmarks only)
        hand_landmarks = extract_hand_landmarks(landmarks)
        
        # Pad or truncate to fixed length
        processed_sequence = pad_or_truncate_sequence(
            hand_landmarks, 
            max_len=MAX_SEQ_LENGTH, 
            features_per_frame=self.input_dim
        )
        
        # Convert to tensor and add batch dimension if needed
        return to_tensor(processed_sequence)
    
    def predict(self, preprocessed_data):
        """
        Run prediction with the transformer model.
        
        Args:
            preprocessed_data: Preprocessed landmark tensor
            
        Returns:
            Raw logits from the model
        """
        with torch.no_grad():
            return self.model(preprocessed_data)
    
    def postprocess(self, raw_predictions, top_n=5):
        """
        Convert model outputs to human-readable predictions.
        
        Args:
            raw_predictions: Tensor of logits from the model
            top_n (int): Number of top predictions to return
            
        Returns:
            List of (label, confidence) tuples for the top predictions
        """
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(raw_predictions, dim=1)
        
        # Get top N predictions
        topk_probs, topk_indices = torch.topk(probabilities, min(top_n, len(self.classes)), dim=1)
        
        # Convert to numpy
        topk_probs = topk_probs.cpu().numpy().flatten()
        topk_indices = topk_indices.cpu().numpy().flatten()
        
        # Format as (label, confidence) pairs
        results = []
        for i, idx in enumerate(topk_indices):
            if idx < len(self.classes):
                results.append((self.classes[idx], float(topk_probs[i])))
        
        return results 