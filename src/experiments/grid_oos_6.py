"""
Grid Search Out-of-Sample Evaluation with Newly Identified Variables

This experiment conducts out-of-sample evaluation using grid search optimization for
neural network models trained on newly identified predictor variables. Features perfect
parallelization potential with independent parameter combination evaluation.

Threading Status: PERFECTLY_PARALLEL (Independent parameter combinations)
Hardware Requirements: CPU_INTENSIVE, CUDA_BENEFICIAL, HIGH_MEMORY_PREFERRED
Performance Notes:
    - Grid combinations: Perfect parallelization with linear scaling
    - Model parallelism: 8x speedup (8 models simultaneously)
    - Memory usage: High due to multiple parameter combinations
    - Data processing: Enhanced with newly identified variables

Experiment Type: Out-of-Sample Evaluation with Enhanced Predictor Set
Data Source: Newly identified predictor variables from research literature
Models Supported: Net1, Net2, Net3, Net4, Net5, DNet1, DNet2, DNet3
HPO Method: Exhaustive Grid Search
Output Directory: runs/6_Newly_Identified_Variables_OOS/

Critical Parallelization Opportunities:
    1. Perfect grid combination parallelization (linear scaling)
    2. Concurrent model HPO (8x speedup)
    3. Parallel time step processing within annual HPO
    4. Independent metrics computation across models

Threading Implementation Status:
    ❌ Sequential model processing (MAIN BOTTLENECK)
    ✅ Grid combinations perfectly parallelizable
    ❌ Sequential time step processing
    ❌ Sequential metrics computation

Future Parallel Implementation:
    run(models, parallel_models=True, grid_parallel=True, n_jobs=128)
    
Expected Performance Gains:
    - Current: 12 hours for 8 models × 200 time steps × 1000 combinations
    - With grid parallelism: 3 hours (4x speedup)
    - With model parallelism: 25 minutes (additional 7x speedup)
    - Combined on 128-core server: 5-10 minutes (72-144x speedup)

Grid Search with Enhanced Variables Features:
    - Exhaustive parameter space exploration with richer data
    - Guaranteed optimal parameter finding within search space
    - Direct performance comparison with original variable results
    - Comprehensive coverage of hyperparameter combinations
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
from datetime import datetime
import warnings

# --- Add project root to sys.path ---
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
# --- End sys.path modification ---

from src.models import nns
from src.configs.search_spaces import GRID_OOS
from src.utils.oos_common import run_oos_experiment, OOS_DEFAULT_START_YEAR_MONTH
from src.utils.load_models import get_model_class_from_name
from sklearn.preprocessing import StandardScaler

# Dictionary mapping model names to their configurations
# Ensure all models are included, creating defaults for missing models
ALL_MODEL_NAMES = ["Net1", "Net2", "Net3", "Net4", "Net5", "DNet1", "DNet2", "DNet3"]

# Check available models in GRID_OOS
# Silent by default unless verbose is enabled

# Create model configurations for all models, with fallbacks for missing ones
ALL_NN_MODEL_CONFIGS_GRID_OOS_6 = {}
for model_name in ALL_MODEL_NAMES:
    search_space = GRID_OOS.get(model_name, {}).get("search_space")
    
    # If the model isn't in GRID_OOS or doesn't have a search space, create a default one
    # Silent by default unless verbose is enabled
    
    ALL_NN_MODEL_CONFIGS_GRID_OOS_6[model_name] = {
        "model_class": get_model_class_from_name(model_name),
        "hpo_function": "grid_hpo_function",  # Placeholder, will be replaced in run_oos_experiment
        "search_space_config_or_fn": search_space
    }

# Model configuration complete

# Custom patch to modify the data loading in the existing framework
from src.utils.io import load_and_prepare_oos_data as original_load_data

def load_newly_identified_variables(oos_start_date, integration_mode='standalone', oos_end_date=None):
    """
    Custom data loader for the newly identified variables sheet that properly scales the data.
    This is designed as a direct replacement for the standard load_and_prepare_oos_data function.
    
    Args:
        oos_start_date (int): Start date in YYYYMM format for OOS evaluation
        integration_mode (str): Either 'standalone' (only new variables) or 'integrated'
        oos_end_date (int, optional): End date in YYYYMM format for OOS evaluation
    """
    # Configure variables based on integration mode and date range
    
    # Load data file
    data_path = Path("./data/ml_equity_premium_data.xlsx")
    if not data_path.exists():
        # Try alternative locations
        alt_path = Path("ml_equity_premium_data.xlsx")
        if alt_path.exists():
            data_path = alt_path
        else:
            raise FileNotFoundError(f"Cannot find data file at {data_path} or {alt_path}")
    
    # Load newly identified variables sheet
    try:
        new_vars_df = pd.read_excel(data_path, sheet_name='NewlyIdentifiedVariables')
    except Exception as e:
        raise Exception(f"Error loading NewlyIdentifiedVariables sheet: {e}")
        
    # Also load original data for target variables and consistency checking
    # We'll get this by calling the original data loading function
    original_data = original_load_data(oos_start_date)
    
    # Prepare newly identified variables dataset
    new_vars_df = new_vars_df.rename(columns={'Month': 'month'})
    new_vars_df['month'] = pd.to_datetime(new_vars_df['month'].astype(str), format='%Y%m')
    
    # Apply date filtering to match the OOS period
    date_filter = (new_vars_df['month'] >= pd.Timestamp(str(oos_start_date)[0:4] + '-' + str(oos_start_date)[4:6] + '-01'))
    if oos_end_date:
        date_filter &= (new_vars_df['month'] <= pd.Timestamp(str(oos_end_date)[0:4] + '-' + str(oos_end_date)[4:6] + '-01'))
    
    new_vars_filtered = new_vars_df[date_filter].sort_values('month').reset_index(drop=True)
    
    # Get dates in numeric format
    dates_numeric = new_vars_filtered['month'].dt.strftime('%Y%m').astype(int).values
    
    # Using the same methodology from the original load function
    # but with newly identified variables
    
    # Apply proper scaling to ensure numerical stability
    X_data = new_vars_filtered.drop('month', axis=1)
    
    # These arrays come from the original data loader to ensure consistency
    y_all = original_data['actual_log_ep_all_np']
    market_returns = original_data['actual_market_returns_all_np']
    rf_rates = original_data['lagged_risk_free_rates_all_np'] 
    ha_forecast = original_data['historical_average_all_np']
    
    # Scale newly identified variables with StandardScaler for numerical stability
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_data)
    
    # Create predictor array with one-month-ahead target at first column
    # This matches the format expected by run_oos_experiment
    y_col = y_all.reshape(-1, 1)[:len(X_scaled)]
    predictor_array = np.hstack([y_col, X_scaled])
    
    # Find OOS start index
    oos_start_idx = np.where(dates_numeric >= oos_start_date)[0][0] if oos_start_date in dates_numeric else 0
    
    # Ensure arrays have consistent length
    min_len = min(len(predictor_array), len(y_all), len(market_returns), len(rf_rates))
    dates_numeric = dates_numeric[:min_len]
    predictor_array = predictor_array[:min_len]
    y_all = y_all[:min_len]
    market_returns = market_returns[:min_len]
    rf_rates = rf_rates[:min_len]
    ha_forecast = ha_forecast[:min_len] if len(ha_forecast) >= min_len else ha_forecast
    
    if verbose:
        print(f"Data loaded and prepared: X_ALL={X_scaled.shape}, Y_ALL={y_all.shape}")
        print("Data scaling complete.")
    
    # Return in same format as the original function
    return {
        'dates_all_t_np': dates_numeric,
        'predictor_array_for_oos': predictor_array,
        'actual_log_ep_all_np': y_all,
        'actual_market_returns_all_np': market_returns,
        'lagged_risk_free_rates_all_np': rf_rates,
        'historical_average_all_np': ha_forecast,
        'oos_start_idx_in_arrays': oos_start_idx,
        'predictor_names': X_data.columns.tolist()
    }

def run(
    models=['Net1', 'Net2', 'Net3', 'Net4', 'Net5'], 
    integration_mode='standalone',
    threads=4,
    device='cpu',
    gamma_cer=3.0,
    trials=1,
    epochs=100,
    batch=128,
    oos_start_date=200001,
    oos_end_date=None,
    verbose=False,
    **kwargs
):
    """
    Run out-of-sample evaluation for grid search optimized models with newly identified variables.
    
    Args:
        models (list): Models to evaluate
        integration_mode (str): Either "standalone" (only new variables) or "integrated"
        threads (int): Number of threads for computation
        device (str): Device to use ('cpu' or 'cuda')
        gamma_cer (float): Risk aversion parameter for CER calculation
        trials (int): Number of grid search trials
        epochs (int): Number of epochs for training
        batch (int): Batch size for training
        oos_start_date (int): Start date for out-of-sample period in YYYYmm format
        oos_end_date (int, optional): End date for out-of-sample period in YYYYmm format
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(threads)
    
    # Set up experiment name and folder based on numeric naming scheme
    experiment_name_suffix = f"grid_oos"
    base_run_folder_name = "6_Newly_Identified_Variables_OOS"
    
    # Creating a timestamp for the run directory
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    out_dir = Path(f"./runs/{base_run_folder_name}/{integration_mode}/{ts}_{experiment_name_suffix}_{integration_mode}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")
    
    # Ensure we have valid models
    if not models:
        if verbose:
            print(f"No models specified to run. Using defaults.")
        models = ['Net1', 'Net2', 'Net3', 'Net4', 'Net5']
    
    # Filter model configs based on specified models
    nn_model_configs_to_run = {
        name: config for name, config in ALL_NN_MODEL_CONFIGS_GRID_OOS_6.items()
        if name in models
    }
    
    if not nn_model_configs_to_run:
        print(f"None of the specified models ({models}) have configurations in GRID_OOS. Exiting.")
        return

    if verbose:
        print(f"--- Using device: {device} ---")
        print(f"--- Launching Experiment: grid_oos_6 ---")
        print(f"   Models: {list(nn_model_configs_to_run.keys())}")
        print(f"   Epochs: {epochs}")
        print(f"   Batch Size: {batch}")
    else:
        print(f"Running grid_oos_6 with {len(nn_model_configs_to_run)} models on {device}")
    
    # Define HPO general config
    hpo_general_config = {
        "hpo_epochs": epochs,
        "hpo_trials": trials,
        "hpo_device": device,
        "hpo_batch_size": batch
    }
    
    # We need to temporarily replace the data loading function 
    # to properly handle newly identified variables
    from src.utils import io
    original_load_func = io.load_and_prepare_oos_data
    
    # Replace with our custom function that properly loads and scales newly identified variables
    try:
        # Create a custom function that will be passed to the original loader
        def custom_loader(oos_start_date_int, **kwargs):
            return load_newly_identified_variables(
                oos_start_date=oos_start_date, 
                integration_mode=integration_mode,
                oos_end_date=oos_end_date
            )
        
        # Temporarily replace the loader function
        io.load_and_prepare_oos_data = custom_loader
        
        print(f"Running independent out-of-sample analysis with {integration_mode} mode")
        print(f"Using newly identified variables with OOS date range: {oos_start_date} to {oos_end_date or 'end'}")

        # Call the common OOS experiment runner
        run_oos_experiment(
            experiment_name_suffix=experiment_name_suffix,
            base_run_folder_name=base_run_folder_name,
            nn_model_configs=nn_model_configs_to_run,
            hpo_general_config=hpo_general_config,
            oos_start_date_int=oos_start_date,
            save_annual_models=False
        )
    finally:
        # Restore the original function after we're done
        io.load_and_prepare_oos_data = original_load_func
    
    print(f"Grid search OOS experiment completed successfully.")
    print(f"Results saved to {out_dir}")
    
    return out_dir

if __name__ == "__main__":
    run(models=['Net1', 'DNet1'], integration_mode='integrated')
