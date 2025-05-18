"""
Mock model implementation for ASL sign recognition.

This module provides a mock model that generates random predictions,
useful for testing and development when a real model is not available.
"""

import numpy as np
import random
from .base_model import ASLModel

class MockModel(ASLModel):
    """
    Mock implementation of the model interface for testing and development.
    """
    def __init__(self, model_path=None, class_list_path=None):
        """
        Initialize the mock model with random prediction behavior.
        
        Args:
            model_path (str): Path to model weights (ignored for mock model)
            class_list_path (str): Path to the file containing class labels
        """
        # Ignore model_path since this is a mock model
        self.classes = self._load_class_list(class_list_path)
        print("Initialized MockModel - will generate random predictions")
    
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
        Preprocess landmarks for the mock model (no actual processing needed).
        
        Args:
            landmarks (list or numpy array): Raw landmark sequences
            
        Returns:
            The input landmarks (unchanged)
        """
        # Convert to numpy if it's a list
        if isinstance(landmarks, list):
            landmarks = np.array(landmarks)
        return landmarks
    
    def predict(self, preprocessed_data):
        """
        Generate random predictions based on whether the sequence contains movement.
        
        Args:
            preprocessed_data: Preprocessed landmark data
            
        Returns:
            Flag indicating if sequence is empty and random class indices with confidences
        """
        # Check if sequence is mostly zeros (no significant motion)
        is_empty_sequence = not np.any(preprocessed_data)
        
        if is_empty_sequence:
            # For empty sequences, we'll handle this specially in postprocess
            return {"is_empty": True}
        
        # For normal sequences, generate random indices and confidences
        num_classes = len(self.classes)
        chosen_indices = random.sample(range(num_classes), min(5, num_classes))
        confidences = sorted([random.random() for _ in range(len(chosen_indices))], reverse=True)
        
        return {
            "is_empty": False,
            "indices": chosen_indices,
            "confidences": confidences
        }
    
    def postprocess(self, raw_predictions, top_n=5):
        """
        Convert random predictions to human-readable format.
        
        Args:
            raw_predictions: Output from predict method
            top_n: Number of top predictions to return
            
        Returns:
            List of (label, confidence) tuples for the top predictions
        """
        if raw_predictions.get("is_empty", False):
            # Handle empty sequence case
            results = [("No significant motion detected", 0.05)]  # 5% confidence
            
            # Add a few random low-confidence predictions
            num_extras = min(4, len(self.classes))
            if num_extras > 0:
                chosen_indices = random.sample(range(len(self.classes)), num_extras)
                for idx in chosen_indices:
                    confidence = round(random.uniform(0.001, 0.02), 3)  # 0.1% to 2%
                    results.append((self.classes[idx], confidence))
            
            return results
        
        # Regular case - use the randomly generated indices and confidences
        results = []
        for i, idx in enumerate(raw_predictions["indices"]):
            if i < top_n and idx < len(self.classes):
                # Scale to percentage (0-100%) but store as decimal (0-1)
                confidence = raw_predictions["confidences"][i]
                results.append((self.classes[idx], confidence))
        
        return results 