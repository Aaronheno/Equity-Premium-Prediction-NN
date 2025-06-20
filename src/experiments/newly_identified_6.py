"""
Newly Identified Variables Data Processing and Evaluation

This experiment processes and evaluates newly identified predictor variables from recent
financial literature research. Features moderate parallelization potential with data
loading operations and independent variable processing capabilities.

Threading Status: PARALLEL_READY (Independent data processing and evaluation modes)
Hardware Requirements: CPU_MODERATE, MODERATE_MEMORY, FAST_STORAGE_BENEFICIAL
Performance Notes:
    - Data loading: I/O bound, benefits from SSD storage
    - Variable processing: Independent operations across new predictors
    - Mode evaluation: Standalone vs integrated modes can run concurrently
    - Memory usage: Moderate scaling with variable count

Threading Implementation Status:
    ❌ Sequential data loading operations (I/O bottleneck)
    ❌ Sequential variable processing across new predictors
    ❌ Sequential mode evaluation (standalone vs integrated)

Critical Parallelization Opportunities:
    1. Concurrent data loading from multiple Excel sheets
    2. Parallel variable processing and validation
    3. Independent evaluation of standalone vs integrated modes
    4. Concurrent model training with different variable sets

Expected Performance Gains:
    - Current: Sequential data processing and evaluation
    - With I/O parallelism: 2-3x speedup for data loading
    - With mode parallelism: 2x speedup (standalone + integrated concurrent)
    - Combined: 4-6x speedup potential
"""

import os
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import joblib
import warnings
from sklearn.preprocessing import StandardScaler

from src.utils.io import RF_ALL
from src.utils.metrics_unified import scale_data, compute_in_r_square, compute_success_ratio, compute_CER
from src.utils.training_grid import train_grid
from src.utils.training_random import run_random_search
from src.utils.training_optuna import run_bayesian_optimization

from src.experiments.grid_is_0 import run as run_grid
from src.experiments.random_is_0 import run as run_random
from src.experiments.bayes_is_0 import run as run_bayes
from src.configs.search_spaces import GRID, RANDOM, BAYES
from src.models import nns

def load_data(integration_mode="standalone"):
    """
    Load data from the excel file, specifically targeting the newly identified variables.
    
    Args:
        integration_mode (str): Either "standalone" (only new variables) or 
                               "integrated" (combine with existing predictors)
    
    Returns:
        tuple: (X_data, y_data, rf_data) containing features, target and risk-free rate
    """
    print("Loading data...")
    
    # Load the main data file
    data_path = Path("ml_equity_premium_data.xlsx")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Cannot find data file at {data_path}")
    
    # Load the newly identified variables
    new_vars = pd.read_excel(data_path, sheet_name='NewlyIdentifiedVariables')
    print(f"Loaded {len(new_vars)} records of newly identified variables")
    
    # Ensure we're working with proper date range (199001 to 202112)
    new_vars = new_vars[(new_vars['Month'] >= 199001) & (new_vars['Month'] <= 202112)]
    
    if integration_mode == "integrated":
        # Load original predictor data
        orig_data = pd.read_excel(data_path, sheet_name='result_predictor')
        
        # Filter to same date range
        orig_data = orig_data[(orig_data['month'] >= 199001) & (orig_data['month'] <= 202112)]
        
        # Check alignment
        if len(orig_data) != len(new_vars):
            raise ValueError(f"Data alignment issue: Original data has {len(orig_data)} records, but newly identified variables have {len(new_vars)} records within the target date range")
        
        # Combine data sets
        # Extract features from original data (exclude month and target variables)
        orig_features = orig_data.drop(['month', 'log_equity_premium', 'equity_premium'], axis=1)
        
        # Extract features from new variables (exclude Month)
        new_features = new_vars.drop(['Month'], axis=1)
        
        # Create combined feature set
        X_ALL = pd.concat([new_features, orig_features], axis=1)
        
        # Extract target variable
        Y_ALL = orig_data['log_equity_premium'].values.reshape(-1, 1)
        
        # Get risk-free rate (assuming it's in RF_ALL from imports)
        rf_data = RF_ALL[(RF_ALL.index >= 199001) & (RF_ALL.index <= 202112)]
        
        # For safety, slice to matching length
        min_len = min(len(X_ALL), len(Y_ALL), len(rf_data))
        X_ALL = X_ALL.iloc[:min_len]
        Y_ALL = Y_ALL[:min_len]
        rf_data = rf_data[:min_len]
        
    else:  # standalone mode
        # Extract features from new variables (exclude Month)
        X_ALL = new_vars.drop(['Month'], axis=1)
        
        # Get target from the original data
        orig_data = pd.read_excel(data_path, sheet_name='result_predictor')
        orig_data = orig_data[(orig_data['month'] >= 199001) & (orig_data['month'] <= 202112)]
        
        # Ensure same length
        if len(orig_data) != len(X_ALL):
            raise ValueError(f"Data alignment issue: Original data has {len(orig_data)} records, but newly identified variables have {len(X_ALL)} records within the target date range")
            
        # Extract target variable
        Y_ALL = orig_data['log_equity_premium'].values.reshape(-1, 1)
        
        # Get risk-free rate
        rf_data = RF_ALL[(RF_ALL.index >= 199001) & (RF_ALL.index <= 202112)]
        
        # For safety, slice to matching length
        min_len = min(len(X_ALL), len(Y_ALL), len(rf_data))
        X_ALL = X_ALL.iloc[:min_len]
        Y_ALL = Y_ALL[:min_len]
        rf_data = rf_data[:min_len]
    
    print(f"Data loaded and prepared: X_ALL={X_ALL.shape}, Y_ALL={np.shape(Y_ALL)}")
    
    return X_ALL, Y_ALL, rf_data


def run(models=['Net1', 'Net2', 'Net3', 'Net4', 'Net5'],
        method='grid',
        integration_mode='standalone',
        trials=10,
        epochs=100,
        batch=256,
        threads=4,
        device='cpu',
        gamma_cer=3.0):
    """
    Run the experiments with newly identified variables.
    
    Args:
        models (list): List of models to evaluate
        method (str): Hyperparameter optimization method ('grid', 'random', or 'bayes')
        integration_mode (str): Either "standalone" (only new variables) or 
                               "integrated" (combine with existing predictors)
        trials (int): Number of trials for random/bayesian optimization
        epochs (int): Number of training epochs
        batch (int): Batch size
        threads (int): Number of threads for computation
        device (str): Device to use ('cpu' or 'cuda')
        gamma_cer (float): Risk aversion parameter for CER calculation
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    integration_tag = "integrated" if integration_mode == "integrated" else "standalone"
    
    # Create output directory
    out_base = Path(f"./runs/6_Newly_Identified_Variables/{integration_tag}")
    out_base.mkdir(parents=True, exist_ok=True)
    
    # Load data based on integration mode
    X_ALL, Y_ALL, rf_data = load_data(integration_mode)
    
    # Choose the appropriate method
    if method == 'grid':
        run_grid(
            models=models,
            trials=trials,
            epochs=epochs,
            threads=threads,
            batch=batch,
            device=device,
            gamma_cer=gamma_cer,
            custom_data=(X_ALL, Y_ALL, rf_data),
            out_dir_override=str(out_base / f"{ts}_{method}_{integration_tag}")
        )
    elif method == 'random':
        run_random(
            models=models,
            trials=trials,
            epochs=epochs,
            threads=threads,
            batch=batch,
            device=device,
            gamma_cer=gamma_cer,
            custom_data=(X_ALL, Y_ALL, rf_data),
            out_dir_override=str(out_base / f"{ts}_{method}_{integration_tag}")
        )
    elif method == 'bayes':
        run_bayes(
            models=models,
            trials=trials,
            epochs=epochs,
            threads=threads,
            batch=batch,
            device=device,
            gamma_cer=gamma_cer,
            custom_data=(X_ALL, Y_ALL, rf_data),
            out_dir_override=str(out_base / f"{ts}_{method}_{integration_tag}")
        )
    else:
        raise ValueError(f"Unknown method: {method}. Please choose from 'grid', 'random', or 'bayes'.")

if __name__ == "__main__":
    # Example usage
    run(models=['Net1'], method='grid', integration_mode='standalone', trials=1, epochs=10, batch=256)
