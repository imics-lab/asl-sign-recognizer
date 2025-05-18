from abc import ABC, abstractmethod
import numpy as np
import torch

class ASLModel(ABC):
    """
    Abstract base class for all ASL sign recognition models.
    
    This defines the interface that all models must implement to be used
    by the ASL sign recognition application.
    """
    
    @abstractmethod
    def preprocess(self, landmarks):
        """
        Preprocess the raw landmark data for model input.
        
        Args:
            landmarks (list or numpy array): Raw landmark sequences extracted from video
                
        Returns:
            Processed data in the format expected by the predict method
        """
        pass
        
    @abstractmethod
    def predict(self, preprocessed_data):
        """
        Run prediction on preprocessed landmarks.
        
        Args:
            preprocessed_data: Preprocessed landmark data from preprocess method
                
        Returns:
            Raw prediction outputs from the model
        """
        pass
        
    @abstractmethod
    def postprocess(self, raw_predictions, top_n=5):
        """
        Convert raw model predictions into human-readable format.
        
        Args:
            raw_predictions: Raw output from the predict method
            top_n (int): Number of top predictions to return
                
        Returns:
            List of (label, confidence) tuples for the top predictions
        """
        pass
    
    def __call__(self, landmarks, top_n=5):
        """
        Convenience method to run the full prediction pipeline.
        
        Args:
            landmarks (list or numpy array): Raw landmark sequences
            top_n (int): Number of top predictions to return
                
        Returns:
            List of (label, confidence) tuples for the top predictions
        """
        preprocessed = self.preprocess(landmarks)
        predictions = self.predict(preprocessed)
        return self.postprocess(predictions, top_n) 