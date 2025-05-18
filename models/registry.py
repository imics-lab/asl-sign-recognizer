"""
Model registry for ASL sign recognition models.

This module provides a registry for dynamically loading different model implementations
by name, allowing for easy swapping of different models.
"""

from typing import Dict, Type, Optional
from .base_model import ASLModel
from .transformer_model import TransformerModel
from .mock_model import MockModel

# Registry of available models
MODEL_REGISTRY: Dict[str, Type[ASLModel]] = {
    "transformer": TransformerModel,
    "mock": MockModel,
}

def get_model(model_name: str, **kwargs) -> Optional[ASLModel]:
    """
    Get an instance of the specified model.
    
    Args:
        model_name (str): Name of the model to load
        **kwargs: Additional arguments to pass to the model constructor
    
    Returns:
        ASLModel: An instance of the requested model, or None if not found
    """
    model_class = MODEL_REGISTRY.get(model_name.lower())
    if model_class is None:
        print(f"Error: Model '{model_name}' not found in registry.")
        print(f"Available models: {', '.join(MODEL_REGISTRY.keys())}")
        return None
    
    try:
        return model_class(**kwargs)
    except Exception as e:
        print(f"Error instantiating model '{model_name}': {e}")
        return None

def register_model(name: str, model_class: Type[ASLModel]) -> None:
    """
    Register a new model with the registry.
    
    Args:
        name (str): Name to register the model under
        model_class (Type[ASLModel]): Model class to register
    """
    MODEL_REGISTRY[name.lower()] = model_class
    print(f"Registered model '{name}'")

def list_available_models() -> list:
    """
    List all available models in the registry.
    
    Returns:
        list: List of available model names
    """
    return list(MODEL_REGISTRY.keys()) 