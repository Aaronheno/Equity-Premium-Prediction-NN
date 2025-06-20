"""
Grid Search Helper Functions and Wrappers

This module provides wrapper functions and utilities for standardizing grid search
hyperparameter optimization calls across different experiments. Features minimal
computational overhead and thread-safe wrapper operations.

Threading Status: THREAD_SAFE (Simple wrapper functions)
Hardware Requirements: CPU_LIGHT, MINIMAL_MEMORY
Performance Notes:
    - Wrapper functions: Negligible computational overhead
    - Parameter standardization: Thread-safe operations
    - Memory usage: Minimal (parameter passing only)
    - Call overhead: <1ms per function call

Threading Implementation Status:
    ✅ Thread-safe wrapper functions
    ✅ Stateless parameter standardization
    ✅ No shared state modifications

Critical Parallelization Opportunities:
    1. Concurrent wrapper function calls across trials
    2. Independent parameter standardization
    3. Thread-safe integration with grid search execution
    4. Parallel experiment coordination through standardized interfaces

Expected Performance Gains:
    - Current: Thread-safe, no bottlenecks
    - Overhead: Negligible wrapper function cost
    - Scalability: Perfect linear scaling with concurrent usage
"""

def grid_hpo_runner_function(**kwargs):
    """Wrapper function to standardize how we call train_grid() from oos_common.py"""
    from src.utils.training_grid import train_grid
    
    # Extract parameters
    model_module = kwargs['model_module']
    regressor_class = kwargs['regressor_class']
    search_space_config = kwargs['search_space_config']
    X_train = kwargs['X_train']
    y_train = kwargs['y_train']
    X_val = kwargs['X_val']
    y_val = kwargs['y_val']
    n_features = kwargs['n_features']
    epochs = kwargs['epochs']
    device = kwargs['device']
    batch_size_default = kwargs.get('batch_size_default', 128)
    
    # Call train_grid with the correct parameters
    return train_grid(
        model_module=model_module,
        regressor_class=regressor_class,
        search_space_config=search_space_config,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_features=n_features,
        epochs=epochs,
        device=device,
        batch_size_default=batch_size_default
    ) 