"""
fred_variables_7.py

This script implements the processing and evaluation of FRED variables
from the FRED_MD sheet in the data input file.

It follows the established workflow pattern with suffix "_7" for script names
and "7_" prefix for results directories.
"""

import os
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import joblib
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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

def load_fred_data():
    """
    Load FRED variables from the excel file, specifically targeting the FRED_MD sheet.
    
    Returns:
        tuple: (X_data, y_data, rf_data) containing features, target and risk-free rate
    """
    print("Loading FRED data...")
    
    # Load the main data file
    data_path = Path("ml_equity_premium_data.xlsx")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Cannot find data file at {data_path}")
    
    # Load the FRED variables
    fred_data = pd.read_excel(data_path, sheet_name='FRED_MD')
    print(f"Loaded {len(fred_data)} records of FRED variables")
    
    # Convert date format to numeric (YYYYMM)
    # FRED_MD sheet has dates in format MM/DD/YYYY or similar
    try:
        # Try to parse as datetime with flexible format detection
        fred_data['month'] = pd.to_datetime(fred_data['sasdate']).dt.strftime('%Y%m').astype(int)
    except Exception as e:
        print(f"Error converting dates: {e}")
        print("Attempting alternative date conversion...")
        # If the above fails, try a manual conversion approach
        date_pattern = r'(\d+)/(\d+)/(\d+)'
        try:
            # Extract month, day, year from MM/DD/YYYY format
            fred_data['month'] = fred_data['sasdate'].str.extract(date_pattern)
            # Convert to YYYYMM format
            fred_data['month'] = fred_data[2].astype(str) + fred_data[0].str.zfill(2)
            fred_data['month'] = fred_data['month'].astype(int)
        except Exception as e2:
            raise ValueError(f"Could not parse date format in FRED_MD sheet: {e2}\nExample date: {fred_data['sasdate'].iloc[0]}")
    
    print(f"Date range in FRED data: {fred_data['month'].min()} to {fred_data['month'].max()}")
    
    
    # Ensure we're working with proper date range (199001 to 202312)
    fred_data = fred_data[(fred_data['month'] >= 199001) & (fred_data['month'] <= 202312)]
    
    # Get target variable from the original data
    orig_data = pd.read_excel(data_path, sheet_name='result_predictor')
    orig_data = orig_data[(orig_data['month'] >= 199001) & (orig_data['month'] <= 202312)]
    
    # Align data by month
    merged_data = pd.merge(fred_data, orig_data[['month', 'log_equity_premium']], on='month', how='inner')
    
    # Extract features (exclude sasdate and month)
    X_ALL = merged_data.drop(['sasdate', 'month', 'log_equity_premium'], axis=1)
    
    # Apply min-max scaling to the FRED variables as in the example code
    X_ALL = pd.DataFrame(
        MinMaxScaler().fit_transform(X_ALL),
        columns=X_ALL.columns,
        index=X_ALL.index
    )
    
    # Extract target variable
    Y_ALL = merged_data['log_equity_premium'].values.reshape(-1, 1)
    
    # Get risk-free rate for the same period
    rf_data = RF_ALL[(RF_ALL.index >= 199001) & (RF_ALL.index <= 202312)]
    
    # For safety, slice to matching length (in case there are any misalignments)
    min_len = min(len(X_ALL), len(Y_ALL), len(rf_data))
    X_ALL = X_ALL.iloc[:min_len]
    Y_ALL = Y_ALL[:min_len]
    rf_data = rf_data[:min_len]
    
    print(f"Data loaded and prepared: X_ALL={X_ALL.shape}, Y_ALL={np.shape(Y_ALL)}")
    
    return X_ALL, Y_ALL, rf_data


def run(models=['Net1', 'Net2', 'Net3', 'Net4', 'Net5'],
        method='grid',
        trials=10,
        epochs=100,
        batch=256,
        threads=4,
        device='cpu',
        gamma_cer=3.0):
    """
    Run the experiments with FRED variables.
    
    Args:
        models (list): List of models to evaluate
        method (str): Hyperparameter optimization method ('grid', 'random', or 'bayes')
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
    
    # Create output directory
    out_base = Path(f"./runs/7_FRED_Variables")
    out_base.mkdir(parents=True, exist_ok=True)
    
    # Load FRED data
    X_ALL, Y_ALL, rf_data = load_fred_data()
    
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
            out_dir_override=str(out_base / f"{ts}_grid")
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
            out_dir_override=str(out_base / f"{ts}_random")
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
            out_dir_override=str(out_base / f"{ts}_bayes")
        )
    else:
        raise ValueError(f"Unknown method: {method}. Please choose from 'grid', 'random', or 'bayes'.")

if __name__ == "__main__":
    # Example usage
    run(models=['Net1'], method='grid', trials=1, epochs=10, batch=256)
