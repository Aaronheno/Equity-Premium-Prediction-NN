"""
Grid Search In-Sample Optimization with MAE Scoring

This experiment conducts in-sample hyperparameter optimization using grid search
with Mean Absolute Error (MAE) as the validation metric instead of MSE. Features
perfect parallelization potential with alternative error metric.

Threading Status: PERFECTLY_PARALLEL (Independent parameter combinations)
Hardware Requirements: CPU_SUFFICIENT, CUDA_BENEFICIAL, MODERATE_MEMORY
Performance Notes:
    - MAE-based grid search: Perfect linear scaling across parameter combinations
    - Independent evaluations: No coordination overhead
    - Memory usage: Moderate (parameter grid storage)
    - Alternative metric: MAE provides different optimization landscape than MSE

Threading Implementation Status:
    ❌ Sequential parameter combination evaluation (main bottleneck)
    ❌ Sequential model processing
    ❌ Single-threaded MAE computation

Critical Parallelization Opportunities:
    1. Independent parameter combination evaluation with MAE scoring
    2. Concurrent model grid search (8x speedup)
    3. Parallel MAE computation across parameter sets
    4. Independent grid execution for different models

Expected Performance Gains:
    - Current: Sequential parameter evaluation
    - With parameter parallelism: 8-32x speedup (perfect scaling)
    - With model parallelism: Additional 4-8x speedup
    - Combined: 32-256x speedup potential
"""
import sys
from pathlib import Path
import os
import numpy as np
import pandas as pd
import torch
import datetime
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import from project modules
from src.models import nns
from src.utils.training_mae import run_grid_search_mae
from src.configs.search_spaces import GRID_IS

def run(models=["Net1"], trials=None, epochs=30, threads=4, batch=256, device='cpu', gamma_cer=5, dropout=0.2):
    """
    Run in-sample hyperparameter optimization using grid search with MAE scoring.
    
    Parameters:
    -----------
    models : list
        List of model names to optimize
    trials : int
        Not used for grid search, kept for API consistency
    epochs : int
        Number of epochs to train each model
    threads : int
        Number of threads to use
    batch : int
        Batch size for training
    device : str
        Device to use ('cpu' or 'cuda')
    gamma_cer : float
        Risk aversion parameter for CER calculation
    dropout : float
        Dropout rate for models that support it
        
    Returns:
    --------
    None, results are saved to disk
    """
    print(f"Starting in-sample grid search with MAE scoring")
    print(f"Models: {models}")
    print(f"Epochs: {epochs}")
    print(f"Device: {device}")
    print(f"Batch size: {batch}")
    
    # Set number of threads for torch
    torch.set_num_threads(threads)
    
    # Create timestamp for folder name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"./runs/5_MAE_IS_grid_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data_path = Path('./data')
    predictor_file = data_path / 'ml_equity_premium_data.xlsx'
    
    if not predictor_file.exists():
        print(f"Error: Data file not found at {predictor_file}")
        return
    
    print(f"Loading data from {predictor_file}")
    predictor_raw = pd.read_excel(predictor_file, sheet_name='result_predictor')
    predictor_raw.set_index('month', inplace=True)
    
    # Prepare data
    X = predictor_raw.iloc[:, 2:].values  # Skip log_equity_premium and equity_premium
    y = predictor_raw['log_equity_premium'].values
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    
    # Scale data
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    
    # Save configuration
    with open(output_dir / "config.txt", "w") as f:
        f.write(f"In-Sample Grid Search with MAE Scoring\n")
        f.write(f"=======================\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Models: {models}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Batch size: {batch}\n")
        f.write(f"Gamma CER: {gamma_cer}\n")
        f.write(f"Threads: {threads}\n")
    
    # Save scalers
    with open(output_dir / "scalers.pkl", "wb") as f:
        pickle.dump({"X": scaler_X}, f)
    
    # Process each model
    for model_name in models:
        print(f"\nProcessing model: {model_name}")
        
        model_dir = output_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        # Get model class
        model_class = getattr(nns, model_name)
        
        # Get parameter grid for this model
        param_grid = GRID_IS.get(model_name, {})
        
        # Add fixed parameters
        if "dropout" in [p for p in model_class.__init__.__code__.co_varnames if p != 'self']:
            param_grid["dropout"] = [dropout]
        
        # Run grid search with MAE scoring
        start_time = time.time()
        best_params = run_grid_search_mae(
            X_train_scaled, y_train, 
            X_val_scaled, y_val,
            model_class, param_grid,
            epochs=epochs, batch_size=batch,
            device=device
        )
        elapsed_time = time.time() - start_time
        
        # Save results
        results = {
            "model": model_name,
            "best_params": best_params,
            "elapsed_time": elapsed_time,
            "epochs": epochs,
            "device": device,
            "batch_size": batch,
            "timestamp": timestamp
        }
        
        # Save as pickle for later use
        with open(model_dir / "best_params.pkl", "wb") as f:
            pickle.dump(results, f)
        
        # Also save as text for human-readable format
        with open(model_dir / "results.txt", "w") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Best parameters: {best_params}\n")
            f.write(f"Time elapsed: {elapsed_time:.2f} seconds\n")
    
    print(f"\nIn-sample grid search with MAE scoring complete. Results saved to {output_dir}")
    return output_dir
