"""
Bayesian Out-of-Sample Evaluation with Newly Identified Variables

This experiment conducts out-of-sample evaluation using Bayesian optimization for neural
network models trained on newly identified predictor variables. Features the same
parallelization potential as standard Bayesian OOS with independent trial execution.

Threading Status: PARALLEL_READY (Independent trials and model processing)
Hardware Requirements: CPU_REQUIRED, CUDA_BENEFICIAL, HIGH_MEMORY_PREFERRED
Performance Notes:
    - Bayesian trials: Near-linear scaling with coordination overhead
    - Model parallelism: 8x speedup (8 models simultaneously)
    - Memory usage: High due to Optuna study storage and model instances
    - Data loading: Enhanced with newly identified variables

Experiment Type: Out-of-Sample Evaluation with Enhanced Predictor Set
Data Source: Newly identified predictor variables from research literature
Models Supported: Net1, Net2, Net3, Net4, Net5, DNet1, DNet2, DNet3
HPO Method: Bayesian Optimization (Optuna TPE)
Output Directory: runs/6_Newly_Identified_Variables_OOS/

Critical Parallelization Opportunities:
    1. Independent Bayesian trial evaluation (near-linear scaling)
    2. Concurrent model HPO (8x speedup)
    3. Parallel time step processing within annual HPO
    4. Independent metrics computation across models

Threading Implementation Status:
    ❌ Sequential model processing (MAIN BOTTLENECK)
    ✅ Optuna trials parallelizable (coordination overhead)
    ❌ Sequential time step processing
    ❌ Sequential metrics computation

Future Parallel Implementation:
    run(models, parallel_models=True, trial_parallel=True, n_jobs=64)
    
Expected Performance Gains:
    - Current: 8 hours for 8 models × 200 time steps × 100 trials
    - With trial parallelism: 3 hours (2.7x speedup with coordination)
    - With model parallelism: 25 minutes (additional 7x speedup)
    - Combined on 128-core server: 5-8 minutes (60-96x speedup)

Newly Identified Variables Features:
    - Enhanced predictor set from recent financial literature
    - Same robust OOS evaluation framework as standard experiments
    - Direct comparison capability with original variable results
    - Potential improved prediction performance with richer data
"""

# src/experiments/bayes_oos_6.py

import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import time

# --- Add project root to sys.path ---
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
# --- End sys.path modification ---

from src.models import nns
from src.configs.search_spaces import BAYES_OOS
from src.utils.training_optuna import run_study as optuna_hpo_runner_function, OptunaSkorchNet
from src.utils.oos_common import run_oos_experiment, OOS_DEFAULT_START_YEAR_MONTH
from src.utils.load_models import get_model_class_from_name
from src.utils.io import load_and_prepare_oos_data, _PREDICTOR_COLUMNS

# Constants for experiment 6
EXP6_START_DATE = 199001  # Data availability starts
EXP6_END_DATE = 202112    # Data availability ends
EXP6_OOS_START_DATE = 200001  # Default OOS start (can be overridden)

def load_newly_identified_variables(data_path=None):
    """
    Load the newly identified variables from the Excel file.
    
    Returns:
        pd.DataFrame: DataFrame with 'month' column and predictor columns
    """
    if data_path is None:
        data_path = Path("./data/ml_equity_premium_data.xlsx")
        if not data_path.exists():
            alt_path = Path("ml_equity_premium_data.xlsx")
            if alt_path.exists():
                data_path = alt_path
            else:
                raise FileNotFoundError(f"Cannot find data file at {data_path} or {alt_path}")
    
    # Load newly identified variables sheet
    new_vars_df = pd.read_excel(data_path, sheet_name='NewlyIdentifiedVariables')
    
    # Process month column
    new_vars_df = new_vars_df.rename(columns={'Month': 'month'})
    new_vars_df['month'] = pd.to_datetime(new_vars_df['month'].astype(str), format='%Y%m')
    
    # Sort by month
    new_vars_df = new_vars_df.sort_values('month').reset_index(drop=True)
    
    return new_vars_df

def calculate_historical_average_restricted(log_ep_series, dates_array, start_date=EXP6_START_DATE):
    """
    Calculate Historical Average predictions using only data from start_date onwards.
    
    For each time t, HA_{t+1} = mean(r_start, r_{start+1}, ..., r_t)
    where start is the index corresponding to start_date.
    
    Args:
        log_ep_series: Array of log equity premium values
        dates_array: Array of dates in YYYYMM format
        start_date: Date from which to start calculating HA (YYYYMM format)
        
    Returns:
        Array of HA predictions aligned with log_ep_series
    """
    n = len(log_ep_series)
    ha_predictions = np.full(n, np.nan)
    
    # Find the index where our restricted period starts
    start_idx = np.where(dates_array >= start_date)[0][0] if start_date in dates_array else 0
    
    print(f"Calculating restricted HA from index {start_idx} (date: {dates_array[start_idx]})")
    
    # For each position, calculate the expanding average from start_idx
    for t in range(n):
        if t < start_idx:
            # Before the start date, we could either use NaN or the first available value
            # Using NaN to clearly indicate no prediction available
            ha_predictions[t] = np.nan
        else:
            # HA for position t predicts t+1
            # Uses mean of returns from start_idx to t (inclusive)
            if t == start_idx:
                # First prediction uses just the first value
                ha_predictions[t] = log_ep_series[start_idx]
            else:
                # Use expanding window from start_idx to current position
                ha_predictions[t] = np.mean(log_ep_series[start_idx:t+1])
    
    return ha_predictions

def prepare_data_for_exp6(integration_mode='standalone', oos_start_date_int=EXP6_OOS_START_DATE):
    """
    Prepare data for experiment 6 with newly identified variables.
    
    Args:
        integration_mode: 'standalone' or 'integrated'
        oos_start_date_int: Start date for OOS evaluation (YYYYMM format)
        
    Returns:
        dict: Data dictionary in the same format as load_and_prepare_oos_data
    """
    # Load newly identified variables
    new_vars_df = load_newly_identified_variables()
    
    # Filter to the valid date range
    start_date_ts = pd.Timestamp(f'{EXP6_START_DATE//100}-{EXP6_START_DATE%100:02d}-01')
    end_date_ts = pd.Timestamp(f'{EXP6_END_DATE//100}-{EXP6_END_DATE%100:02d}-01')
    
    date_mask = (new_vars_df['month'] >= start_date_ts) & (new_vars_df['month'] <= end_date_ts)
    new_vars_filtered = new_vars_df[date_mask].copy()
    
    print(f"Filtered newly identified variables: {len(new_vars_filtered)} records from {EXP6_START_DATE} to {EXP6_END_DATE}")
    
    # Load original data for targets and benchmarks
    original_data = load_and_prepare_oos_data(oos_start_date_int)
    
    # Convert dates to numeric format for matching
    new_vars_dates = new_vars_filtered['month'].dt.strftime('%Y%m').astype(int).values
    original_dates = original_data['dates_all_t_np']
    
    # Find common date range
    common_dates = np.intersect1d(new_vars_dates, original_dates)
    common_dates = common_dates[(common_dates >= EXP6_START_DATE) & (common_dates <= EXP6_END_DATE)]
    
    if len(common_dates) == 0:
        raise ValueError("No overlapping dates between original and newly identified variables")
    
    print(f"Common date range: {common_dates[0]} to {common_dates[-1]} ({len(common_dates)} periods)")
    
    # Get indices for common dates
    new_idx_map = {date: idx for idx, date in enumerate(new_vars_dates)}
    orig_idx_map = {date: idx for idx, date in enumerate(original_dates)}
    
    # Extract features based on integration mode
    if integration_mode == 'standalone':
        # Use only newly identified variables
        feature_cols = [col for col in new_vars_filtered.columns if col != 'month']
        X_features = []
        
        for date in common_dates:
            new_idx = new_idx_map[date]
            row_features = new_vars_filtered.iloc[new_idx][feature_cols].values
            X_features.append(row_features)
        
        X_features = np.array(X_features, dtype=np.float32)
        predictor_names = feature_cols
        print(f"Standalone mode: Using {len(feature_cols)} newly identified variables")
        
    else:  # integrated mode
        # Combine original and new variables
        original_features = original_data['predictor_array_for_oos'][:, 1:]  # Skip target
        new_feature_cols = [col for col in new_vars_filtered.columns if col != 'month']
        
        X_features = []
        
        for date in common_dates:
            orig_idx = orig_idx_map[date]
            new_idx = new_idx_map[date]
            
            # Combine features
            orig_feats = original_features[orig_idx]
            new_feats = new_vars_filtered.iloc[new_idx][new_feature_cols].values
            combined_feats = np.concatenate([orig_feats, new_feats])
            X_features.append(combined_feats)
        
        X_features = np.array(X_features, dtype=np.float32)
        predictor_names = list(_PREDICTOR_COLUMNS) + new_feature_cols
        print(f"Integrated mode: Using {len(_PREDICTOR_COLUMNS)} original + {len(new_feature_cols)} new = {len(predictor_names)} total variables")
    
    # Handle NaN values
    if np.isnan(X_features).any():
        print(f"Warning: Found {np.isnan(X_features).sum()} NaN values in features. Applying forward/backward fill...")
        # Forward fill then backward fill along time axis
        for col_idx in range(X_features.shape[1]):
            col = X_features[:, col_idx]
            # Forward fill
            mask = np.isnan(col)
            idx = np.where(~mask, np.arange(len(mask)), 0)
            np.maximum.accumulate(idx, out=idx)
            col[mask] = col[idx[mask]]
            # Backward fill for any remaining NaNs
            mask = np.isnan(col)
            idx = np.where(~mask, np.arange(len(mask)), len(mask)-1)
            idx = np.minimum.accumulate(idx[::-1])[::-1]
            col[mask] = col[idx[mask]]
            # Final check - fill with column mean if still NaN
            if np.isnan(col).any():
                col[np.isnan(col)] = np.nanmean(col)
            X_features[:, col_idx] = col
    
    # Get target variables and other arrays for common dates
    common_indices_orig = [orig_idx_map[date] for date in common_dates]
    
    y_values = original_data['actual_log_ep_all_np'][common_indices_orig]
    market_returns = original_data['actual_market_returns_all_np'][common_indices_orig]
    rf_rates = original_data['lagged_risk_free_rates_all_np'][common_indices_orig]
    
    # IMPORTANT: Recalculate Historical Average using only data from EXP6_START_DATE
    print(f"\nRecalculating Historical Average using only data from {EXP6_START_DATE}...")
    ha_forecasts = calculate_historical_average_restricted(y_values, common_dates, EXP6_START_DATE)
    
    # Verify HA calculation
    first_valid_idx = np.where(~np.isnan(ha_forecasts))[0][0]
    print(f"First valid HA prediction at index {first_valid_idx} (date: {common_dates[first_valid_idx]})")
    print(f"Sample HA values: {ha_forecasts[first_valid_idx:first_valid_idx+5]}")
    
    # Create predictor array [y_{t+1}, X_t]
    predictor_array = np.hstack([y_values.reshape(-1, 1), X_features])
    
    # Find OOS start index
    oos_start_idx = np.where(common_dates >= oos_start_date_int)[0][0] if oos_start_date_int in common_dates else 0
    
    # Ensure OOS start is after we have valid HA predictions
    if oos_start_idx < first_valid_idx:
        print(f"Warning: OOS start ({common_dates[oos_start_idx]}) is before first valid HA. Adjusting to {common_dates[first_valid_idx]}")
        oos_start_idx = first_valid_idx
    
    print(f"OOS period starts at index {oos_start_idx} (date: {common_dates[oos_start_idx]})")
    
    return {
        'dates_all_t_np': common_dates,
        'predictor_array_for_oos': predictor_array,
        'actual_log_ep_all_np': y_values,
        'actual_market_returns_all_np': market_returns,
        'lagged_risk_free_rates_all_np': rf_rates,
        'historical_average_all_np': ha_forecasts,  # Now using recalculated HA
        'oos_start_idx_in_arrays': oos_start_idx,
        'predictor_names': predictor_names
    }

def run_oos_experiment_exp6(
    experiment_name_suffix,
    base_run_folder_name,
    nn_model_configs,
    hpo_general_config,
    oos_start_date_int,
    integration_mode='standalone',
    save_annual_models=False
):
    """
    Custom OOS experiment runner for experiment 6 that uses prepare_data_for_exp6
    instead of the standard data loader.
    """
    from src.utils.oos_common import (
        get_oos_paths, StandardScaler, LinearRegression, 
        mean_squared_error, joblib, tqdm, EarlyStopping,
        compute_CER, compute_success_ratio, compute_in_r_square,
        CW_test, PT_test, HPO_ANNUAL_TRIGGER_MONTH, 
        VAL_RATIO_FOR_HPO, CER_GAMMA
    )
    
    print(f"--- Starting OOS Experiment 6: {experiment_name_suffix} ---")
    print(f"Integration mode: {integration_mode}")
    print(f"Date range: {EXP6_START_DATE} to {EXP6_END_DATE}")
    print(f"Historical Average will be calculated using only data from {EXP6_START_DATE}")
    
    paths = get_oos_paths(base_run_folder_name, experiment_name_suffix)
    
    # Load data using our custom function
    data_dict = prepare_data_for_exp6(integration_mode, oos_start_date_int)
    
    # The rest of the logic is the same as run_oos_experiment
    # I'll include the key parts here for completeness
    
    dates_t_all = data_dict['dates_all_t_np']
    predictor_array = data_dict['predictor_array_for_oos']
    actual_log_ep_all = data_dict['actual_log_ep_all_np']
    actual_market_returns_all = data_dict['actual_market_returns_all_np']
    lagged_rf_all = data_dict['lagged_risk_free_rates_all_np']
    ha_forecasts_all = data_dict['historical_average_all_np']
    oos_start_idx = data_dict['oos_start_idx_in_arrays']
    predictor_names = data_dict['predictor_names']
    
    num_total_periods = predictor_array.shape[0]
    
    # Initialize storage
    oos_predictions_nn = {model_name: [] for model_name in nn_model_configs.keys()}
    oos_predictions_cf = []
    annual_best_hps_nn = {model_name: None for model_name in nn_model_configs.keys()}
    
    print(f"OOS period: {dates_t_all[oos_start_idx]} to {dates_t_all[-1]}")
    print(f"Total OOS steps: {num_total_periods - oos_start_idx}")
    
    # Main OOS loop
    for t_idx in tqdm(range(oos_start_idx, num_total_periods), desc="OOS Evaluation"):
        current_date_t = dates_t_all[t_idx]
        current_month_int = int(str(current_date_t)[-2:])
        
        train_data_slice = predictor_array[:t_idx, :]
        
        if train_data_slice.shape[0] < 20:
            print(f"Skipping {current_date_t}: Insufficient training data")
            for model_name in nn_model_configs.keys():
                oos_predictions_nn[model_name].append(np.nan)
            oos_predictions_cf.append(np.nan)
            continue
        
        X_train_full_unscaled_curr = train_data_slice[:, 1:]
        y_train_full_unscaled_curr = train_data_slice[:, 0].reshape(-1, 1)
        n_features = X_train_full_unscaled_curr.shape[1]
        
        # Fit scalers
        scaler_x_current = StandardScaler().fit(X_train_full_unscaled_curr)
        scaler_y_current = StandardScaler().fit(y_train_full_unscaled_curr)
        
        # Annual HPO trigger
        trigger_hpo_this_step = (current_month_int == HPO_ANNUAL_TRIGGER_MONTH)
        
        # Process each neural network model
        for model_name, config in nn_model_configs.items():
            model_class = config['model_class']
            hpo_function = config['hpo_function']
            regressor_class = config['regressor_class']
            search_space_config_or_fn = config['search_space_config_or_fn']
            
            # Get HPO parameters
            hpo_epochs = hpo_general_config.get("hpo_epochs", 25)
            hpo_trials = hpo_general_config.get("hpo_trials", 20)
            hpo_device = hpo_general_config.get("hpo_device", "cpu")
            hpo_batch_size = hpo_general_config.get("hpo_batch_size", 128)
            
            # Run HPO if triggered
            if trigger_hpo_this_step:
                print(f"\nRunning HPO for {model_name} at {current_date_t}...")
                
                # Prepare HPO data
                n_samples_total_hpo = X_train_full_unscaled_curr.shape[0]
                n_val_hpo = max(1, int(n_samples_total_hpo * VAL_RATIO_FOR_HPO))
                n_train_hpo = n_samples_total_hpo - n_val_hpo
                
                if n_train_hpo < 1:
                    print(f"Insufficient data for HPO. Using previous parameters.")
                    continue
                
                # Split and scale data
                X_train_hpo_unscaled = X_train_full_unscaled_curr[:n_train_hpo, :]
                y_train_hpo_unscaled = y_train_full_unscaled_curr[:n_train_hpo, :]
                X_val_hpo_unscaled = X_train_full_unscaled_curr[n_train_hpo:, :]
                y_val_hpo_unscaled = y_train_full_unscaled_curr[n_train_hpo:, :]
                
                X_train_hpo_scaled = scaler_x_current.transform(X_train_hpo_unscaled)
                y_train_hpo_scaled = scaler_y_current.transform(y_train_hpo_unscaled)
                X_val_hpo_scaled = scaler_x_current.transform(X_val_hpo_unscaled)
                y_val_hpo_scaled = scaler_y_current.transform(y_val_hpo_unscaled)
                
                X_train_hpo_tensor = torch.from_numpy(X_train_hpo_scaled.astype(np.float32)).to(hpo_device)
                y_train_hpo_tensor = torch.from_numpy(y_train_hpo_scaled.astype(np.float32)).to(hpo_device)
                X_val_hpo_tensor = torch.from_numpy(X_val_hpo_scaled.astype(np.float32)).to(hpo_device)
                y_val_hpo_tensor = torch.from_numpy(y_val_hpo_scaled.astype(np.float32)).to(hpo_device)
                
                try:
                    # Run HPO
                    best_params_from_hpo, hpo_study_or_results = hpo_function(
                        model_module=model_class,
                        skorch_net_class=regressor_class,
                        hpo_config_fn=search_space_config_or_fn,
                        X_hpo_train=X_train_hpo_tensor,
                        y_hpo_train=y_train_hpo_tensor,
                        X_hpo_val=X_val_hpo_tensor,
                        y_hpo_val=y_val_hpo_tensor,
                        trials=hpo_trials,
                        epochs=hpo_epochs,
                        device=hpo_device,
                        batch_size_default=hpo_batch_size,
                        study_name_prefix=f"{experiment_name_suffix}_{model_name}_{current_date_t}"
                    )
                    
                    annual_best_hps_nn[model_name] = best_params_from_hpo
                    
                    # Save best params
                    joblib.dump(best_params_from_hpo, 
                              paths['annual_best_params'] / f"{model_name}_{current_date_t}_best_params.joblib")
                    
                except Exception as e:
                    print(f"HPO failed for {model_name}: {e}")
                    best_params_from_hpo = annual_best_hps_nn.get(model_name, {})
            
            # Get current best parameters
            current_best_params_dict = annual_best_hps_nn.get(model_name, {})
            
            if not current_best_params_dict:
                print(f"No parameters available for {model_name}. Skipping.")
                oos_predictions_nn[model_name].append(np.nan)
                continue
            
            # Prepare model parameters
            model_init_kwargs = {"n_feature": n_features, "n_output": 1}
            
            # Extract module parameters
            for k, v in current_best_params_dict.items():
                if k.startswith("module__"):
                    param_name = k.replace("module__", "")
                    # Remove any model-specific suffix
                    for suffix in [f"_{model_name}", "_Net1", "_Net2", "_Net3", "_Net4", "_Net5", "_DNet1", "_DNet2", "_DNet3"]:
                        if param_name.endswith(suffix):
                            param_name = param_name[:-len(suffix)]
                            break
                    model_init_kwargs[param_name] = v
            
            # Create model instance
            try:
                model_instance = model_class(**model_init_kwargs)
            except Exception as e:
                print(f"Failed to create model {model_name}: {e}")
                oos_predictions_nn[model_name].append(np.nan)
                continue
            
            # Get optimizer
            optimizer_name = current_best_params_dict.get("optimizer", "Adam")
            if isinstance(optimizer_name, str):
                optimizer_class = getattr(torch.optim, optimizer_name)
            else:
                optimizer_class = optimizer_name
            
            # Train final model
            X_train_fit_scaled = scaler_x_current.transform(X_train_full_unscaled_curr)
            y_train_fit_scaled = scaler_y_current.transform(y_train_full_unscaled_curr)
            
            X_train_fit_tensor = torch.from_numpy(X_train_fit_scaled.astype(np.float32)).to(hpo_device)
            y_train_fit_tensor = torch.from_numpy(y_train_fit_scaled.astype(np.float32)).to(hpo_device)
            
            final_nn_model = regressor_class(
                module=model_instance,
                max_epochs=hpo_epochs,
                optimizer=optimizer_class,
                lr=current_best_params_dict.get("lr", 0.001),
                optimizer__weight_decay=current_best_params_dict.get("optimizer__weight_decay", 0.0),
                batch_size=hpo_batch_size,
                l1_lambda=current_best_params_dict.get("l1_lambda", 0.0),
                iterator_train__shuffle=True,
                callbacks=[EarlyStopping(patience=10)],
                device=hpo_device,
            )
            
            final_nn_model.fit(X_train_fit_tensor, y_train_fit_tensor)
            
            # Make prediction
            X_oos_current_unscaled = predictor_array[t_idx, 1:].reshape(1, -1)
            X_oos_current_scaled = scaler_x_current.transform(X_oos_current_unscaled)
            X_oos_current_tensor = torch.from_numpy(X_oos_current_scaled.astype(np.float32)).to(hpo_device)
            
            y_pred_oos_scaled = final_nn_model.predict(X_oos_current_tensor)
            y_pred_oos_unscaled = scaler_y_current.inverse_transform(y_pred_oos_scaled.reshape(-1, 1)).item()
            oos_predictions_nn[model_name].append(y_pred_oos_unscaled)
        
        # Combination Forecast (CF) - simple average of OLS predictions
        cf_preds = []
        for p_idx, p_name in enumerate(predictor_names[:min(20, len(predictor_names))]):  # Limit to 20 predictors
            X_train_cf_single = X_train_full_unscaled_curr[:, [p_idx]]
            y_train_cf = y_train_full_unscaled_curr.ravel()
            
            if X_train_cf_single.shape[0] > 0:
                ols_model = LinearRegression()
                ols_model.fit(X_train_cf_single, y_train_cf)
                X_oos_cf_single = predictor_array[t_idx, [1 + p_idx]]
                cf_preds.append(ols_model.predict(X_oos_cf_single.reshape(1, -1))[0])
        
        oos_predictions_cf.append(np.mean(cf_preds) if cf_preds else np.nan)
    
    # Save predictions and calculate metrics
    print("\nSaving predictions and calculating metrics...")
    
    # Create DataFrames
    df_oos_predictions_nn = pd.DataFrame(oos_predictions_nn, index=dates_t_all[oos_start_idx:])
    df_oos_predictions_cf = pd.DataFrame({'CF': oos_predictions_cf}, index=dates_t_all[oos_start_idx:])
    df_oos_predictions_ha = pd.DataFrame({'HA': ha_forecasts_all[oos_start_idx:]}, index=dates_t_all[oos_start_idx:])
    df_oos_actual_log_ep = pd.DataFrame({'ActualLogEP': actual_log_ep_all[oos_start_idx:]}, index=dates_t_all[oos_start_idx:])
    df_oos_actual_mkt_ret = pd.DataFrame({'ActualMktRet': actual_market_returns_all[oos_start_idx:]}, index=dates_t_all[oos_start_idx:])
    df_oos_lagged_rf = pd.DataFrame({'LaggedRF': lagged_rf_all[oos_start_idx:]}, index=dates_t_all[oos_start_idx:])
    
    # Combine all predictions
    df_all_oos_outputs = pd.concat([
        df_oos_actual_log_ep, df_oos_actual_mkt_ret, df_oos_lagged_rf,
        df_oos_predictions_ha, df_oos_predictions_cf, df_oos_predictions_nn
    ], axis=1)
    
    df_all_oos_outputs.to_csv(paths['full_predictions_file'])
    print(f"Predictions saved to {paths['full_predictions_file']}")
    
    # Calculate metrics
    oos_metrics_results = []
    
    actual_log_ep_oos_final = actual_log_ep_all[oos_start_idx:]
    actual_mkt_ret_oos_final = actual_market_returns_all[oos_start_idx:]
    lagged_rf_oos_final = lagged_rf_all[oos_start_idx:]
    ha_oos_final = ha_forecasts_all[oos_start_idx:]
    
    # Filter out NaN values from HA for metrics calculation
    ha_valid_mask = ~np.isnan(ha_oos_final)
    
    # HA metrics
    if np.sum(ha_valid_mask) > 0:
        sr_ha = compute_success_ratio(actual_log_ep_oos_final[ha_valid_mask], 
                                      ha_oos_final[ha_valid_mask]) * 100
        cer_ha = compute_CER(actual_mkt_ret_oos_final[ha_valid_mask], 
                            ha_oos_final[ha_valid_mask], 
                            lagged_rf_oos_final[ha_valid_mask], CER_GAMMA) * 100
        _, pt_stat_ha, pt_pval_ha = PT_test(actual_log_ep_oos_final[ha_valid_mask], 
                                            ha_oos_final[ha_valid_mask])
    else:
        sr_ha = cer_ha = pt_stat_ha = pt_pval_ha = np.nan
    
    oos_metrics_results.append({
        'Model': 'HA',
        'OOS_R2_vs_HA (%)': 0.0,
        'Success_Ratio (%)': sr_ha,
        'CER_annual (%)': cer_ha,
        'CW_stat': np.nan,
        'CW_pvalue': np.nan,
        'PT_stat': pt_stat_ha,
        'PT_pvalue': pt_pval_ha
    })
    
    # CF metrics
    cf_preds_oos_final = np.array(oos_predictions_cf)
    valid_cf_mask = ~np.isnan(cf_preds_oos_final) & ha_valid_mask  # Both CF and HA must be valid
    if np.sum(valid_cf_mask) > 0:
        oos_r2_cf = compute_in_r_square(actual_log_ep_oos_final[valid_cf_mask], 
                                       ha_oos_final[valid_cf_mask], 
                                       cf_preds_oos_final[valid_cf_mask]) * 100
        sr_cf = compute_success_ratio(actual_log_ep_oos_final[valid_cf_mask], 
                                    cf_preds_oos_final[valid_cf_mask]) * 100
        cer_cf = compute_CER(actual_mkt_ret_oos_final[valid_cf_mask], 
                           cf_preds_oos_final[valid_cf_mask], 
                           lagged_rf_oos_final[valid_cf_mask], CER_GAMMA) * 100
        cw_stat_cf, cw_pval_cf = CW_test(actual_log_ep_oos_final[valid_cf_mask], 
                                        ha_oos_final[valid_cf_mask], 
                                        cf_preds_oos_final[valid_cf_mask])
        _, pt_stat_cf, pt_pval_cf = PT_test(actual_log_ep_oos_final[valid_cf_mask], 
                                           cf_preds_oos_final[valid_cf_mask])
    else:
        oos_r2_cf = sr_cf = cer_cf = cw_stat_cf = cw_pval_cf = pt_stat_cf = pt_pval_cf = np.nan
    
    oos_metrics_results.append({
        'Model': 'CF',
        'OOS_R2_vs_HA (%)': oos_r2_cf,
        'Success_Ratio (%)': sr_cf,
        'CER_annual (%)': cer_cf,
        'CW_stat': cw_stat_cf,
        'CW_pvalue': cw_pval_cf,
        'PT_stat': pt_stat_cf,
        'PT_pvalue': pt_pval_cf
    })
    
    # NN model metrics
    for model_name in nn_model_configs.keys():
        preds_nn_final = df_oos_predictions_nn[model_name].values
        valid_nn_mask = ~np.isnan(preds_nn_final) & ha_valid_mask  # Both NN and HA must be valid
        
        if np.sum(valid_nn_mask) > 0:
            oos_r2_nn = compute_in_r_square(actual_log_ep_oos_final[valid_nn_mask],
                                          ha_oos_final[valid_nn_mask],
                                          preds_nn_final[valid_nn_mask]) * 100
            sr_nn = compute_success_ratio(actual_log_ep_oos_final[valid_nn_mask],
                                        preds_nn_final[valid_nn_mask]) * 100
            cer_nn = compute_CER(actual_mkt_ret_oos_final[valid_nn_mask],
                               preds_nn_final[valid_nn_mask],
                               lagged_rf_oos_final[valid_nn_mask], CER_GAMMA) * 100
            cw_stat_nn, cw_pval_nn = CW_test(actual_log_ep_oos_final[valid_nn_mask],
                                            ha_oos_final[valid_nn_mask],
                                            preds_nn_final[valid_nn_mask])
            _, pt_stat_nn, pt_pval_nn = PT_test(actual_log_ep_oos_final[valid_nn_mask],
                                               preds_nn_final[valid_nn_mask])
        else:
            oos_r2_nn = sr_nn = cer_nn = cw_stat_nn = cw_pval_nn = pt_stat_nn = pt_pval_nn = np.nan
        
        oos_metrics_results.append({
            'Model': model_name,
            'OOS_R2_vs_HA (%)': oos_r2_nn,
            'Success_Ratio (%)': sr_nn,
            'CER_annual (%)': cer_nn,
            'CW_stat': cw_stat_nn,
            'CW_pvalue': cw_pval_nn,
            'PT_stat': pt_stat_nn,
            'PT_pvalue': pt_pval_nn
        })
    
    # Add CER gain vs HA
    ha_cer = oos_metrics_results[0]['CER_annual (%)']
    for i in range(len(oos_metrics_results)):
        oos_metrics_results[i]['CER_gain_vs_HA (%)'] = oos_metrics_results[i]['CER_annual (%)'] - ha_cer
    
    # Create and save metrics DataFrame
    df_oos_metrics = pd.DataFrame(oos_metrics_results)
    
    # Reorder columns
    cols = list(df_oos_metrics.columns)
    cer_idx = cols.index('CER_annual (%)')
    cols.insert(cer_idx + 1, cols.pop(cols.index('CER_gain_vs_HA (%)')))
    df_oos_metrics = df_oos_metrics[cols]
    
    print("\n--- OOS Metrics ---")
    print(df_oos_metrics)
    df_oos_metrics.to_csv(paths['metrics_file'], index=False)
    
    return df_oos_metrics, df_all_oos_outputs

# Define model configurations
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

def run(
    model_names,
    oos_start_date_int=EXP6_OOS_START_DATE,
    hpo_general_config=None,
    save_annual_models=False,
    integration_mode='standalone'
):
    """
    Runs the Out-of-Sample (OOS) experiment using Bayesian Optimization (Optuna) for HPO,
    using newly identified variables for experiment 6.
    """
    # Validate date range
    if oos_start_date_int < EXP6_START_DATE:
        print(f"Warning: OOS start date {oos_start_date_int} is before data availability. Setting to {EXP6_START_DATE}")
        oos_start_date_int = EXP6_START_DATE
    
    if oos_start_date_int > EXP6_END_DATE:
        raise ValueError(f"OOS start date {oos_start_date_int} is after data end date {EXP6_END_DATE}")
    
    experiment_name_suffix = f"bayes_opt_oos_6_{integration_mode}"
    base_run_folder_name = "6_Newly_Identified_Variables_OOS"
    
    # Use default HPO config if none provided
    if hpo_general_config is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        hpo_general_config = {
            "hpo_epochs": 50,
            "hpo_trials": 50,
            "hpo_device": device,
            "hpo_batch_size": 128
        }
    
    # Filter model configs
    if not model_names:
        print("No models specified. Exiting.")
        return
    
    nn_model_configs_to_run = {
        name: config for name, config in ALL_NN_MODEL_CONFIGS_BAYES_OOS.items()
        if name in model_names
    }
    
    if not nn_model_configs_to_run:
        print(f"None of the specified models ({model_names}) have configurations. Exiting.")
        return
    
    print(f"--- Running Bayesian OOS 6 for models: {list(nn_model_configs_to_run.keys())} ---")
    print(f"Integration mode: {integration_mode}")
    print(f"OOS start date: {oos_start_date_int}")
    
    # Run the custom experiment
    return run_oos_experiment_exp6(
        experiment_name_suffix=experiment_name_suffix,
        base_run_folder_name=base_run_folder_name,
        nn_model_configs=nn_model_configs_to_run,
        hpo_general_config=hpo_general_config,
        oos_start_date_int=oos_start_date_int,
        integration_mode=integration_mode,
        save_annual_models=save_annual_models
    )

if __name__ == '__main__':
    # Test the implementation
    print("--- Running bayes_oos_6.py directly for testing ---")
    
    test_models = ["Net1", "Net2", "Net3", "Net4", "Net5", "DNet1", "DNet2", "DNet3"]
    test_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    test_hpo_config = {
        "hpo_epochs": 30,
        "hpo_trials": 10,
        "hpo_device": test_device,
        "hpo_batch_size": 128
    }
    
    # Test standalone mode
    print("\n=== Testing STANDALONE mode ===")
    run(
        model_names=test_models,
        oos_start_date_int=200001,
        hpo_general_config=test_hpo_config,
        save_annual_models=False,
        integration_mode='standalone'
    )
    
    # Test integrated mode
    print("\n=== Testing INTEGRATED mode ===")
    run(
        model_names=test_models,
        oos_start_date_int=200001,
        hpo_general_config=test_hpo_config,
        save_annual_models=False,
        integration_mode='integrated'
    )