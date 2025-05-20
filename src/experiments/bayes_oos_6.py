# src/experiments/bayes_oos_6.py
"""
Out-of-sample evaluation for models trained on newly identified variables
using Bayesian optimization.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd

# --- Add project root to sys.path ---
# This ensures Python can find the 'src' package when running the script.
# It assumes this script is located at src/experiments/bayes_oos_6.py
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
# --- End sys.path modification ---

from src.models import nns
from src.configs.search_spaces import BAYES_OOS
from src.utils.training_optuna import run_study as optuna_hpo_runner_function, OptunaSkorchNet
from src.utils.oos_common import run_oos_experiment, OOS_DEFAULT_START_YEAR_MONTH, load_and_prepare_oos_data
from src.utils.load_models import get_model_class_from_name

# Define a mapping from model names (strings) to their classes and HPO details
ALL_NN_MODEL_CONFIGS_BAYES_OOS = {
    model_name: {
        "model_class": get_model_class_from_name(model_name),
        "hpo_function": optuna_hpo_runner_function,
        "regressor_class": OptunaSkorchNet,
        "search_space_config_or_fn": BAYES_OOS.get(model_name, {}).get("hpo_config_fn")
    }
    for model_name in ["Net1", "Net2", "Net3", "Net4", "Net5", "DNet1", "DNet2", "DNet3"]
    if BAYES_OOS.get(model_name, {}).get("hpo_config_fn") is not None
}

# This function is not used anymore as we're monkey patching oos_common.load_and_prepare_oos_data instead
# Keeping the name for future reference
def _unused_run_config_oos_6(data_params):
    """
    Custom data preprocessing for experiment 6 with newly identified variables.
    This function loads the data, filters it to the date range 199001-202112,
    and handles NaN values properly.
    
    Args:
        data_params: Dictionary containing data parameters
        
    Returns:
        Modified data_params with newly identified variables
    """
    # Get the integration mode from the params if available, default to 'standalone'
    integration_mode = data_params.get('integration_mode', 'standalone')
    print(f"Running preprocessing for experiment 6 in {integration_mode} mode")
    
    # Load the newly identified variables data
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
        
        # Check for NaN values only if they exist
        nan_cols = new_vars_df.columns[new_vars_df.isna().any()].tolist()
        if nan_cols:
            print(f"Warning: Found NaN values in columns: {nan_cols}")
    except Exception as e:
        raise Exception(f"Error loading NewlyIdentifiedVariables sheet: {e}")
    
    # Load original data for target variables and consistency
    original_data = load_and_prepare_oos_data(OOS_DEFAULT_START_YEAR_MONTH)
    
    # Prepare newly identified variables dataset
    new_vars_df = new_vars_df.rename(columns={'Month': 'month'})
    new_vars_df['month'] = pd.to_datetime(new_vars_df['month'].astype(str), format='%Y%m')
    
    # Filter to the specified date range (199001-202112) - STRICT REQUIREMENT
    start_date_ts = pd.Timestamp('1990-01-01')
    end_date_ts = pd.Timestamp('2021-12-31')
    date_filter = (new_vars_df['month'] >= start_date_ts) & (new_vars_df['month'] <= end_date_ts)
    print(f"Limiting data to date range: 1990-01-01 to 2021-12-31")
    
    new_vars_filtered = new_vars_df[date_filter].sort_values('month').reset_index(drop=True)
    
    # Get dates in numeric format
    dates_numeric = new_vars_filtered['month'].dt.strftime('%Y%m').astype(int).values
    
    # Extract features and handle NaN values
    X_data = new_vars_filtered.drop('month', axis=1)
    
    # Handle NaN values in features - critical for stability
    # First, check how many NaNs we have
    total_cells = X_data.size
    nan_cells = X_data.isna().sum().sum()
    if nan_cells > 0:
        print(f"Found {nan_cells} NaN values out of {total_cells} cells ({nan_cells/total_cells*100:.2f}%)")
    
    # Method 1: Forward fill then backward fill (for time series data)
    X_data_filled = X_data.fillna(method='ffill').fillna(method='bfill')
    
    # Method 2: For any columns that still have NaNs, fill with column median
    still_has_nans = X_data_filled.columns[X_data_filled.isna().any()].tolist()
    if still_has_nans:
        print(f"Some columns still have NaNs after ffill/bfill: {still_has_nans}")
        col_medians = X_data_filled.median()
        X_data_filled = X_data_filled.fillna(col_medians)
    
    # Final check - use column means for any remaining NaNs
    if X_data_filled.isna().any().any():
        print("Warning: Still found NaNs after median filling. Using column means as final fallback.")
        X_data_filled = X_data_filled.fillna(X_data_filled.mean())
        
        # Last resort: replace any remaining NaNs with zeros
        X_data_filled = X_data_filled.fillna(0)
    
    # Replace the original data with the filled version
    X_data = X_data_filled
    print(f"NaN check after filling: {X_data.isna().any().any()}")
    
    if integration_mode == 'integrated':
        # For integrated mode, we need to merge with original features
        original_features = original_data['predictor_array_for_oos'][:, 1:]  # Skip target column
        original_feature_dates = original_data['dates_all_t_np']
        
        # Match dates to combine datasets
        matched_indices = []
        for i, date in enumerate(dates_numeric):
            if date in original_feature_dates:
                idx = np.where(original_feature_dates == date)[0][0]
                matched_indices.append((i, idx))
        
        if not matched_indices:
            raise ValueError("No matching dates found between original and new variables datasets")
        
        # Create combined dataset with both original and new features
        combined_X = []
        combined_dates = []
        
        for new_idx, orig_idx in matched_indices:
            # Get row from X_data and ensure it has no NaNs
            new_row = X_data.iloc[new_idx].values
            # Safety check - should be redundant now but keeping as final guard
            if np.isnan(new_row).any():
                print(f"Warning: Found NaNs in row {new_idx} after previous filling steps")
                new_row = np.nan_to_num(new_row, nan=0.0)
                
            # Combine with original features
            combined_row = np.concatenate([original_features[orig_idx], new_row])
            combined_X.append(combined_row)
            combined_dates.append(dates_numeric[new_idx])
        
        X_combined = np.array(combined_X, dtype=np.float32)  # Force float32 for consistency
        dates_numeric = np.array(combined_dates)
        
        print(f"Integrated mode: Combined features shape: {X_combined.shape}")
        X_to_use = X_combined
    else:
        # For standalone mode, just use the new variables
        X_to_use = X_data.values.astype(np.float32)  # Force float32 for consistency
        # Final guard against NaNs
        if np.isnan(X_to_use).any():
            print("Warning: NaNs found in final standalone data. Replacing with zeros.")
            X_to_use = np.nan_to_num(X_to_use, nan=0.0)
            
        print(f"Standalone mode: Using only new variables, shape: {X_to_use.shape}")
    
    # These arrays come from the original data loader to ensure consistency
    y_all = original_data['actual_log_ep_all_np']
    market_returns = original_data['actual_market_returns_all_np']
    rf_rates = original_data['lagged_risk_free_rates_all_np']
    ha_forecast = original_data['historical_average_all_np']
    
    # Find dates that match between original and new data
    common_dates = []
    common_indices_orig = []
    common_indices_new = []
    
    for i, date in enumerate(original_data['dates_all_t_np']):
        if date in dates_numeric:
            common_dates.append(date)
            common_indices_orig.append(i)
            common_indices_new.append(np.where(dates_numeric == date)[0][0])
    
    # Create target array with one-month-ahead target at first column
    y_col = y_all[common_indices_orig].reshape(-1, 1)
    predictor_array = np.hstack([y_col, X_to_use[common_indices_new]])
    
    # Find OOS start index
    oos_start_idx = np.where(np.array(common_dates) >= OOS_DEFAULT_START_YEAR_MONTH)[0][0] if OOS_DEFAULT_START_YEAR_MONTH in common_dates else 0
    
    # Return in same format as the original function
    return {
        'dates_all_t_np': np.array(common_dates),
        'predictor_array_for_oos': predictor_array,
        'actual_log_ep_all_np': y_all[common_indices_orig],
        'actual_market_returns_all_np': market_returns[common_indices_orig],
        'lagged_risk_free_rates_all_np': rf_rates[common_indices_orig],
        'historical_average_all_np': ha_forecast[common_indices_orig],
        'oos_start_idx_in_arrays': oos_start_idx,
        'predictor_names': X_data.columns.tolist() if integration_mode == 'standalone' else None
    }

# Override the load_and_prepare_oos_data function for our specific use case
# Create a module-level variable to store the integration mode
_INTEGRATION_MODE = 'standalone'

def _custom_load_oos_data_for_exp6(oos_start_date_int):
    """
    Custom data loader for experiment 6 with newly identified variables.
    This function loads the data, filters it to the date range 199001-202112.
    
    Args:
        oos_start_date_int: Start date for out-of-sample evaluation
        
    Returns:
        Data dictionary with newly identified variables
    """
    global _INTEGRATION_MODE
    integration_mode = _INTEGRATION_MODE
    print(f"Loading data for experiment 6 in {integration_mode} mode")
    
    # Load the newly identified variables data
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
        print(f"Loaded {len(new_vars_df)} records of newly identified variables")
    except Exception as e:
        raise Exception(f"Error loading NewlyIdentifiedVariables sheet: {e}")
    
    # Also load original data for target variables and consistency
    # We'll use the original function to load baseline data
    original_data = load_and_prepare_oos_data(oos_start_date_int)
    
    # Prepare newly identified variables dataset
    new_vars_df = new_vars_df.rename(columns={'Month': 'month'})
    new_vars_df['month'] = pd.to_datetime(new_vars_df['month'].astype(str), format='%Y%m')
    
    # Filter to the specified date range (199001-202112) - STRICT REQUIREMENT
    start_date_ts = pd.Timestamp('1990-01-01')
    end_date_ts = pd.Timestamp('2021-12-31')
    date_filter = (new_vars_df['month'] >= start_date_ts) & (new_vars_df['month'] <= end_date_ts)
    print(f"Limiting data to date range: 1990-01-01 to 2021-12-31")
    
    new_vars_filtered = new_vars_df[date_filter].sort_values('month').reset_index(drop=True)
    
    # Get dates in numeric format
    dates_numeric = new_vars_filtered['month'].dt.strftime('%Y%m').astype(int).values
    
    # Extract features
    X_data = new_vars_filtered.drop('month', axis=1)
    
    if integration_mode == 'integrated':
        # For integrated mode, we need to merge with original features
        original_features = original_data['predictor_array_for_oos'][:, 1:]  # Skip target column
        original_feature_dates = original_data['dates_all_t_np']
        
        # Match dates to combine datasets
        matched_indices = []
        for i, date in enumerate(dates_numeric):
            if date in original_feature_dates:
                idx = np.where(original_feature_dates == date)[0][0]
                matched_indices.append((i, idx))
        
        if not matched_indices:
            raise ValueError("No matching dates found between original and new variables datasets")
        
        # Create combined dataset with both original and new features
        combined_X = []
        combined_dates = []
        
        for new_idx, orig_idx in matched_indices:
            # Combine with original features
            combined_row = np.concatenate([original_features[orig_idx], X_data.iloc[new_idx].values])
            combined_X.append(combined_row)
            combined_dates.append(dates_numeric[new_idx])
        
        X_combined = np.array(combined_X, dtype=np.float32)  # Force float32 for consistency
        dates_numeric = np.array(combined_dates)
        
        print(f"Integrated mode: Combined features shape: {X_combined.shape}")
        X_to_use = X_combined
    else:
        # For standalone mode, just use the new variables
        X_to_use = X_data.values.astype(np.float32)  # Force float32 for consistency
        print(f"Standalone mode: Using only new variables, shape: {X_to_use.shape}")
    
    # These arrays come from the original data loader to ensure consistency
    y_all = original_data['actual_log_ep_all_np']
    market_returns = original_data['actual_market_returns_all_np']
    rf_rates = original_data['lagged_risk_free_rates_all_np']
    ha_forecast = original_data['historical_average_all_np']
    
    # Find dates that match between original and new data
    common_dates = []
    common_indices_orig = []
    common_indices_new = []
    
    for i, date in enumerate(original_data['dates_all_t_np']):
        if date in dates_numeric:
            common_dates.append(date)
            common_indices_orig.append(i)
            common_indices_new.append(np.where(dates_numeric == date)[0][0])
    
    # Create target array with one-month-ahead target at first column
    y_col = y_all[common_indices_orig].reshape(-1, 1)
    predictor_array = np.hstack([y_col, X_to_use[common_indices_new]])
    
    # Find OOS start index
    oos_start_idx = np.where(np.array(common_dates) >= oos_start_date_int)[0][0] if oos_start_date_int in common_dates else 0
    
    # Return in same format as the original function
    return {
        'dates_all_t_np': np.array(common_dates),
        'predictor_array_for_oos': predictor_array,
        'actual_log_ep_all_np': y_all[common_indices_orig],
        'actual_market_returns_all_np': market_returns[common_indices_orig],
        'lagged_risk_free_rates_all_np': rf_rates[common_indices_orig],
        'historical_average_all_np': ha_forecast[common_indices_orig],
        'oos_start_idx_in_arrays': oos_start_idx,
        'predictor_names': X_data.columns.tolist() if integration_mode == 'standalone' else None
    }

# Replace the standard loader with our custom one
# Note: We're patching oos_common.load_and_prepare_oos_data which comes from src.utils.io
# This is consistent with what grid_oos_6.py and random_oos_6.py are doing
from src.utils import oos_common
original_load_and_prepare_oos_data = oos_common.load_and_prepare_oos_data

def run(
    model_names,
    oos_start_date_int=OOS_DEFAULT_START_YEAR_MONTH,
    hpo_general_config=None,
    save_annual_models=False,
    integration_mode='standalone'
):
    """
    Runs the Out-of-Sample (OOS) experiment using Bayesian Optimization (Optuna) for HPO,
    using newly identified variables for experiment 6.
    
    Parameters:
    -----------
    model_names : list
        List of model names to run experiments for
    oos_start_date_int : int
        Start date for out-of-sample evaluation (format: YYYYMM)
    hpo_general_config : dict
        Configuration for hyperparameter optimization
    save_annual_models : bool
        Whether to save models trained on annual data
    integration_mode : str
        Mode of integration - 'standalone' or 'integrated'
    """
    experiment_name_suffix = "bayes_opt_oos_6"  # Updated suffix for experiment 6
    base_run_folder_name = "6_Newly_Identified_Variables_OOS"  # Updated folder name for experiment 6
    
    # Use default HPO config if none is provided
    if hpo_general_config is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        hpo_general_config = {
            "hpo_epochs": 50,
            "hpo_trials": 50,
            "hpo_device": device,
            "hpo_batch_size": 128
        }
    
    # Filter model configs based on provided model names
    if not model_names:
        print("No models specified to run. Exiting.")
        return

    nn_model_configs_to_run = {
        name: config for name, config in ALL_NN_MODEL_CONFIGS_BAYES_OOS.items()
        if name in model_names
    }
    
    if not nn_model_configs_to_run:
        print(f"None of the specified models ({model_names}) have configurations in BAYES_OOS. Exiting.")
        return

    print(f"--- Running Bayesian OOS 6 for models: {list(nn_model_configs_to_run.keys())} ---")
    print(f"Integration mode: {integration_mode}")
    
    # Set the global integration mode to be used by our custom loader
    global _INTEGRATION_MODE
    _INTEGRATION_MODE = integration_mode
    
    # Patch the oos_common module with our custom data loader
    oos_common.load_and_prepare_oos_data = _custom_load_oos_data_for_exp6
    
    try:
        # Run the OOS experiment with standard parameters - our custom loader will be used automatically
        run_oos_experiment(
            experiment_name_suffix=experiment_name_suffix,
            base_run_folder_name=base_run_folder_name,
            nn_model_configs=nn_model_configs_to_run,
            hpo_general_config=hpo_general_config,
            oos_start_date_int=oos_start_date_int,
            save_annual_models=save_annual_models
        )
    finally:
        # Restore the original data loader no matter what happens
        oos_common.load_and_prepare_oos_data = original_load_and_prepare_oos_data

if __name__ == '__main__':
    # This section allows direct execution of this script for testing
    print("--- Running bayes_oos_6.py directly for testing purposes ---")
    
    # Define the models to test
    test_models = ["Net1", "DNet1"]
    
    # Determine device for testing
    test_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Test device: {test_device}")
    
    # Define hyperparameter optimization configuration
    test_hpo_general_config = {
        "hpo_epochs": 50,
        "hpo_trials": 10,  # Reduced for testing
        "hpo_device": test_device,
        "hpo_batch_size": 128
    }
    print(f"HPO Config: {test_hpo_general_config}")
    
    # Check if all models have configurations
    ready_to_test = True
    for model_name_test in test_models:
        if model_name_test not in BAYES_OOS or not BAYES_OOS[model_name_test].get("hpo_config_fn"):
            print(f"Error: BAYES_OOS['{model_name_test}']['hpo_config_fn'] is not defined. Cannot run test.", file=sys.stderr)
            ready_to_test = False
            break
            
    if ready_to_test:
        # Run the experiment with default parameters
        run(
            model_names=test_models,
            oos_start_date_int=OOS_DEFAULT_START_YEAR_MONTH,
            hpo_general_config=test_hpo_general_config,
            save_annual_models=False,
            integration_mode='standalone'  # Default to standalone mode for testing
        )
    else:
        print("Please define hpo_config_fn for test models in BAYES_OOS for testing.", file=sys.stderr)
        
    print("--- Test run of bayes_oos_6.py finished ---")
