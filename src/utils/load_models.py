# src/utils/load_models.py
"""
Utility functions for dynamically loading models by name.
"""
from src.models import nns

def get_model_class_from_name(model_name):
    """
    Maps a model name (string) to its corresponding class.
    
    Args:
        model_name (str): The name of the model (e.g., "Net1", "DNet2")
        
    Returns:
        The model class (e.g., nns.Net1, nns.DNet2)
        
    Raises:
        ValueError: If model_name is not found in nns module
    """
    if hasattr(nns, model_name):
        return getattr(nns, model_name)
    else:
        # List available models to help with debugging
        available_models = [name for name in dir(nns) if not name.startswith('_') and name in getattr(nns, '__all__', [])]
        raise ValueError(f"Model '{model_name}' not found in nns module. Available models: {available_models}")

def list_available_model_names():
    """
    Returns a list of all available model names from the nns module.
    
    Returns:
        list: Names of all available models
    """
    # Use __all__ if defined, otherwise filter for non-private, non-module attributes
    if hasattr(nns, '__all__'):
        return nns.__all__
    else:
        return [name for name in dir(nns) 
                if not name.startswith('_') 
                and not name.startswith('nn') 
                and not name.startswith('torch')
                and name[0].isupper()]  # Classes typically start with uppercase