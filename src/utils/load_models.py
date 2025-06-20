"""
Model Loading and Discovery Utilities

This module provides utility functions for dynamically loading neural network models
by name and discovering available model classes. Features thread-safe model class
resolution with minimal computational overhead.

Threading Status: THREAD_SAFE (Read-only model class access)
Hardware Requirements: CPU_LIGHT, MINIMAL_MEMORY
Performance Notes:
    - Model class loading: Thread-safe, read-only operations
    - Class discovery: One-time overhead, cacheable results
    - Memory usage: Minimal (only class references)
    - Access time: <1ms per model class lookup

Threading Implementation Status:
    ✅ Thread-safe model class access
    ✅ Read-only module introspection
    ✅ No shared state modifications

Critical Parallelization Opportunities:
    1. Concurrent model class loading across experiments
    2. Independent model discovery operations
    3. Thread-safe model instantiation coordination
    4. Parallel experiment setup with different model types

Expected Performance Gains:
    - Current: Thread-safe, no bottlenecks
    - Caching potential: Model class discovery results
    - Overhead: Negligible lookup time
    - Scalability: Perfect for concurrent model loading
"""

# src/utils/load_models.py
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