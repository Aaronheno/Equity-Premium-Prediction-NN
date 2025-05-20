# src/utils/oos_common.py   
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib # For saving/loading models and params
from tqdm import tqdm
import time
from skorch.callbacks import EarlyStopping # Import EarlyStopping
import json
import inspect # <<< ADDED THIS IMPORT

# Project structure allows these imports
from src.utils.io import load_and_prepare_oos_data, _PREDICTOR_COLUMNS # _PREDICTOR_COLUMNS for CF
from src.utils.metrics import compute_CER, compute_success_ratio, compute_in_r_square
from src.utils.statistical_tests import CW_test, PT_test

# --- Import HPO runner functions and related utilities ---
from src.utils.training_optuna import run_study as optuna_hpo_runner_function
from src.utils.training_grid import train_grid as grid_hpo_function
from src.utils.training_random import train_random as random_hpo_runner_function
# from src.utils.grid_helpers import grid_hpo_runner_function # This seemed unused in experiment scripts, replaced by train_grid

# --- Configuration ---
OOS_DEFAULT_START_YEAR_MONTH = 195701 # Default, can be overridden by CLI
HPO_ANNUAL_TRIGGER_MONTH = 1 # Re-run HPO in January (month 1) of each year
VAL_RATIO_FOR_HPO = 0.15 # 15% of current training data for validation during HPO
CER_GAMMA = 3.0 # Risk aversion for CER, consistent with your metrics.py

def get_oos_paths(base_run_folder_name, experiment_name_suffix):
    """Generates standardized paths for OOS experiment outputs."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{ts}_{experiment_name_suffix}"
    
    base_output_dir = Path("runs") / base_run_folder_name / run_name
    
    paths = {
        "base": base_output_dir,
        "predictions_individual_models": base_output_dir / "predictions_nn_models",
        "predictions_cf": base_output_dir / "predictions_cf",
        "annual_best_models": base_output_dir / "annual_best_models",
        "annual_best_params": base_output_dir / "annual_best_params",
        "annual_optuna_studies": base_output_dir / "annual_optuna_studies", # Specific to Bayesian
        "metrics_file": base_output_dir / "oos_final_metrics.csv",
        "full_predictions_file": base_output_dir / "oos_all_predictions_raw_with_actuals.csv"
    }
    for path_key, path_val in paths.items():
        if path_key != "metrics_file" and path_key != "full_predictions_file": # Files are created, not dirs
            path_val.mkdir(parents=True, exist_ok=True)
    return paths

def run_oos_experiment(
    experiment_name_suffix,
    base_run_folder_name,
    nn_model_configs, # Dict: {'Net1': {'model_class': Net1, 'hpo_function': hpo_fn, 'regressor_class': SkorchNet, 'search_space_config_or_fn': cfg}, ...}
    hpo_general_config, # Dict: {'hpo_epochs': E, 'hpo_trials': T, 'hpo_device': D, 'hpo_batch_size': B}
    oos_start_date_int=OOS_DEFAULT_START_YEAR_MONTH,
    hpo_trigger_month=HPO_ANNUAL_TRIGGER_MONTH,
    val_ratio_hpo=VAL_RATIO_FOR_HPO,
    predictor_cols_for_cf=None, # List of predictor names for CF, defaults to _PREDICTOR_COLUMNS
    save_annual_models=False    # <<< ADDED: Parameter to control saving of annual models
):
    """
    Main Out-of-Sample (OOS) evaluation loop.
    """
    print(f"--- Starting OOS Experiment: {experiment_name_suffix} ---")
    paths = get_oos_paths(base_run_folder_name, experiment_name_suffix)
    
    if predictor_cols_for_cf is None:
        predictor_cols_for_cf = _PREDICTOR_COLUMNS # From io.py

    # 1. Load and Prepare Full Dataset
    print(f"Loading data for OOS starting {oos_start_date_int}...")
    data_dict = load_and_prepare_oos_data(oos_start_date_int)
    
    dates_t_all = data_dict['dates_all_t_np']
    predictor_array = data_dict['predictor_array_for_oos'] # [y_{t+1}, X_t]
    actual_log_ep_all = data_dict['actual_log_ep_all_np'] # y_{t+1}
    actual_market_returns_all = data_dict['actual_market_returns_all_np'] # R_{m,t+1}
    lagged_rf_all = data_dict['lagged_risk_free_rates_all_np'] # R_{f,t}
    ha_forecasts_all = data_dict['historical_average_all_np'] # HA for y_{t+1}
    oos_start_idx = data_dict['oos_start_idx_in_arrays']
    # predictor_names_from_data_load = data_dict['predictor_names'] # For NNs, features are all cols after target

    num_total_periods = predictor_array.shape[0] # This is N-1 from original data
    
    # --- OOS Predictions Storage ---
    oos_predictions_nn = {model_name: [] for model_name in nn_model_configs.keys()}
    oos_predictions_cf = []
    
    annual_best_hps_nn = {model_name: None for model_name in nn_model_configs.keys()}
    # Store the fitted scalers annually if HPO is annual, or more frequently if needed
    # For simplicity, we refit them on each training slice before HPO/retraining
    
    print(f"OOS period: {dates_t_all[oos_start_idx]} to {dates_t_all[-1]}")
    print(f"Total OOS steps: {num_total_periods - oos_start_idx}")

    # 2. Out-of-Sample Loop
    for t_idx in tqdm(range(oos_start_idx, num_total_periods), desc="OOS Evaluation"):
        current_date_t = dates_t_all[t_idx] # Date at time t (forecast for t+1)
        current_month_int = int(str(current_date_t)[-2:]) # Extract month (1-12)

        # Data for current step:
        # Training data is predictor_array[0...t_idx-1, :]
        # We predict for period t_idx (target is actual_log_ep_all[t_idx])
        # using predictors X from predictor_array[t_idx, 1:]
        
        train_data_slice = predictor_array[:t_idx, :] 
        
        if train_data_slice.shape[0] < 20: # Min data to start (e.g., ~2 years for monthly)
            print(f"Skipping {current_date_t}: Insufficient training data ({train_data_slice.shape[0]} periods)")
            for model_name in nn_model_configs.keys():
                oos_predictions_nn[model_name].append(np.nan)
            oos_predictions_cf.append(np.nan)
            continue

        X_train_full_unscaled_curr = train_data_slice[:, 1:]
        y_train_full_unscaled_curr = train_data_slice[:, 0].reshape(-1, 1) # Ensure 2D for scaler

        n_features = X_train_full_unscaled_curr.shape[1]

        # --- Scalers for X and y (fit on current full training data) ---
        # These scalers are used for HPO validation data and final model training/prediction
        scaler_x_current = StandardScaler().fit(X_train_full_unscaled_curr)
        scaler_y_current = StandardScaler().fit(y_train_full_unscaled_curr)

        # --- Annual HPO and Model Retraining ---
        trigger_hpo_this_step = (current_month_int == hpo_trigger_month)

        # --- Neural Network Models ---
        for model_name, config in nn_model_configs.items():
            model_class = config['model_class']
            hpo_function = config['hpo_function']
            regressor_class = config['regressor_class']
            search_space_config_or_fn = config['search_space_config_or_fn']

            # Extract HPO parameters from hpo_general_config with defaults
            hpo_epochs = hpo_general_config.get("hpo_epochs", 25)
            hpo_trials = hpo_general_config.get("hpo_trials", 20) # Used by Random and Bayes
            hpo_device = hpo_general_config.get("hpo_device", "cpu")
            hpo_batch_size = hpo_general_config.get("hpo_batch_size", 128)
            # Common HPO settings (can be overridden or extended in hpo_general_config if needed)
            hpo_scoring = 'neg_mean_squared_error'
            hpo_verbose = 1 # Minimal verbosity for HPO stages
            use_early_stopping_hpo = True # Default to True for HPO
            early_stopping_patience_hpo = 5 # Default patience for HPO
            early_stopping_delta_hpo = 0.001 # Default delta for HPO

            best_params_from_hpo = None
            best_score_from_hpo = -np.inf
            hpo_study_or_results = None

            hpo_start_time = time.time()

            print(f"Running HPO for {model_name} (year {current_date_t // 100}, OOS step date {current_date_t})...")
            print(f"  HPO Config: Epochs={hpo_epochs}, Trials={hpo_trials if hpo_function != grid_hpo_function else 'N/A (Grid)'}, Device={hpo_device}, BatchSize={hpo_batch_size}")

            # --- Prepare Data for HPO (Split and Scale) ---
            # HPO uses a validation set taken from the end of X_train_full_unscaled_curr
            n_samples_total_hpo = X_train_full_unscaled_curr.shape[0]
            n_val_hpo = int(n_samples_total_hpo * val_ratio_hpo)
            
            if n_val_hpo < 1 and n_samples_total_hpo >= 1:
                n_val_hpo = 1 # Ensure at least one validation sample if possible
            
            n_train_hpo = n_samples_total_hpo - n_val_hpo

            if n_train_hpo < 1:
                print(f"Warning: Not enough data for HPO training split for {model_name} at {current_date_t} (Total: {n_samples_total_hpo}, Train: {n_train_hpo}, Val: {n_val_hpo}). Skipping HPO.", file=sys.stderr)
                # HPO will likely fail or use previous year's params; ensure best_params_from_hpo remains None or {}
                annual_best_hps_nn[model_name] = annual_best_hps_nn.get(model_name, {}) # Keep old or empty
                # The existing logic to handle empty current_best_params_dict will take care of skipping retraining
            else:
                X_train_hpo_unscaled = X_train_full_unscaled_curr[:n_train_hpo, :]
                y_train_hpo_unscaled = y_train_full_unscaled_curr[:n_train_hpo, :]
                X_val_hpo_unscaled = X_train_full_unscaled_curr[n_train_hpo:, :]
                y_val_hpo_unscaled = y_train_full_unscaled_curr[n_train_hpo:, :]

                # Scale HPO data using the scalers fit on the *entire current* training data slice
                X_train_hpo_scaled = scaler_x_current.transform(X_train_hpo_unscaled)
                y_train_hpo_scaled = scaler_y_current.transform(y_train_hpo_unscaled)
                X_val_hpo_scaled = scaler_x_current.transform(X_val_hpo_unscaled)
                y_val_hpo_scaled = scaler_y_current.transform(y_val_hpo_unscaled)

                X_train_hpo_tensor = torch.from_numpy(X_train_hpo_scaled.astype(np.float32)).to(hpo_device)
                y_train_hpo_tensor = torch.from_numpy(y_train_hpo_scaled.astype(np.float32)).to(hpo_device)
                X_val_hpo_tensor = torch.from_numpy(X_val_hpo_scaled.astype(np.float32)).to(hpo_device)
                y_val_hpo_tensor = torch.from_numpy(y_val_hpo_scaled.astype(np.float32)).to(hpo_device)

                try:
                    if hpo_function == grid_hpo_function:
                        # The train_grid function returns (best_params, best_net)
                        # We need to match the interface correctly
                        best_params_from_hpo, best_net = hpo_function(
                            model_module=model_class,
                            regressor_class=regressor_class,
                            search_space_config=search_space_config_or_fn,
                            X_train=X_train_hpo_tensor,
                            y_train=y_train_hpo_tensor,
                            X_val=X_val_hpo_tensor,
                            y_val=y_val_hpo_tensor,
                            n_features=X_train_hpo_tensor.shape[1],
                            epochs=hpo_epochs,
                            device=hpo_device,
                            batch_size_default=hpo_batch_size
                        )
                        # For consistency with other HPO methods
                        hpo_study_or_results = best_net
                        best_score_from_hpo = float('inf')  # Not returned by grid search
                    elif hpo_function == random_hpo_runner_function:
                        # The train_random function returns (best_hyperparams, best_net)
                        # We need to match the interface correctly
                        best_params_from_hpo, best_net = hpo_function(
                            model_module=model_class,
                            regressor_class=regressor_class,
                            search_space_config=search_space_config_or_fn,
                            X_train=X_train_hpo_tensor, 
                            y_train=y_train_hpo_tensor, 
                            X_val=X_val_hpo_tensor,     
                            y_val=y_val_hpo_tensor,     
                            n_features=X_train_hpo_tensor.shape[1],
                            epochs=hpo_epochs,
                            device=hpo_device,
                            trials=hpo_trials,
                            batch_size_default=hpo_batch_size
                        )
                        # For consistency with other HPO methods, treat best_net as hpo_study_or_results
                        hpo_study_or_results = best_net
                        best_score_from_hpo = float('inf')  # Not returned by random search
                    elif hpo_function == optuna_hpo_runner_function:
                        # The optuna_hpo_runner_function is the run_study function from training_optuna.py
                        # Match the interface expected by run_study
                        best_params_from_hpo, hpo_study_or_results = hpo_function(
                            model_module=model_class,               # PyTorch model class
                            skorch_net_class=regressor_class,       # Skorch wrapper
                            hpo_config_fn=search_space_config_or_fn, # Function from search_spaces.py
                            X_hpo_train=X_train_hpo_tensor,          # Training data
                            y_hpo_train=y_train_hpo_tensor,          # Training targets
                            X_hpo_val=X_val_hpo_tensor,              # Validation data
                            y_hpo_val=y_val_hpo_tensor,              # Validation targets
                            trials=hpo_trials,                      # Number of trials
                            epochs=hpo_epochs,                      # Max epochs per trial
                            device=hpo_device,                      # Device (cuda/cpu)
                            batch_size_default=hpo_batch_size,      # Batch size
                            study_name_prefix=f"{experiment_name_suffix}_{model_name}_{current_date_t}" # For naming the study
                        )
                        if hpo_study_or_results and hpo_study_or_results.best_trial:
                            best_params_from_hpo = hpo_study_or_results.best_trial.params
                            best_score_from_hpo = hpo_study_or_results.best_trial.value
                        else:
                            print(f"Warning: Optuna study for {model_name} did not yield a best trial.", file=sys.stderr)
                            # Fallback or error handling needed if no best trial
                            best_params_from_hpo = {} # Empty dict to avoid downstream errors
                            best_score_from_hpo = -np.inf

                    else:
                        raise ValueError(f"Unsupported HPO function type for {model_name}")

                    hpo_duration = time.time() - hpo_start_time
                    print(f"HPO for {model_name} (year {current_date_t // 100}) took {hpo_duration:.2f}s. Best params: {best_params_from_hpo}")
                    annual_best_hps_nn[model_name] = best_params_from_hpo
                    # Save Optuna study object if it's Optuna and save_annual_models is True (or a new flag)
                    if hpo_function == optuna_hpo_runner_function and hpo_study_or_results and save_annual_models: # Check save_annual_models or a more specific flag
                        study_path = paths['annual_optuna_studies'] / f"{model_name}_{current_date_t}_optuna_study.joblib"
                        joblib.dump(hpo_study_or_results, study_path)
                        print(f"Saved Optuna study for {model_name} to {study_path}")
                    
                    # Save best params (already done in original code, keeping it)
                    joblib.dump(best_params_from_hpo, paths['annual_best_params'] / f"{model_name}_{current_date_t}_best_params.joblib")

                except Exception as e:
                    print(f"ERROR: Exception during HPO for {model_name} at {current_date_t}: {e}", file=sys.stderr)
                    # Ensure best_params_from_hpo is an empty dict to allow fallback to previous year's HPs or skip
                    best_params_from_hpo = {} 
                    # Keep existing annual_best_hps_nn[model_name] if HPO fails, or set to empty if first time
                    annual_best_hps_nn[model_name] = annual_best_hps_nn.get(model_name, {})

            # Retrain model on full current training data (X_train_full_unscaled_curr, y_train_full_unscaled_curr)
            # Update annual_best_hps_nn only if HPO ran successfully and produced valid parameters
            if trigger_hpo_this_step and best_params_from_hpo and any(k.startswith("module__") for k in best_params_from_hpo):
                annual_best_hps_nn[model_name] = best_params_from_hpo
                print(f"Updated best parameters for {model_name} at {current_date_t}: {best_params_from_hpo}")
                
            # Use current year's best params if available, otherwise fall back to previous year's params
            current_best_params_dict = annual_best_hps_nn.get(model_name, {})

            # Note: annual_best_hps_nn was already updated above if HPO was successful

            # Defensive check for HPO results before retraining
            if not current_best_params_dict or not any(k.startswith("module__") and k != "module__n_features" for k in current_best_params_dict):
                print(f"Warning: HPO for {model_name} at {current_date_t} did not return sufficient module parameters. Best params found: {current_best_params_dict}. Skipping retraining and prediction for this step.", file=sys.stderr)
                oos_predictions_nn[model_name].append(np.nan)
                if save_annual_models and paths: # Ensure paths is initialized
                     try:
                        with open(paths['annual_best_models'] / f"{model_name}_{current_date_t}_MODEL_SKIPPED_NO_VALID_HPO_PARAMS.txt", "w") as f:
                            f.write(f"HPO failed or returned insufficient module parameters: {current_best_params_dict}")
                     except Exception as e_path:
                        print(f"Error saving skipped model placeholder: {e_path}", file=sys.stderr)
                continue # Skip to next OOS step for this model

            X_train_fit_scaled = scaler_x_current.transform(X_train_full_unscaled_curr) # Use already fitted scaler
            y_train_fit_scaled = scaler_y_current.transform(y_train_full_unscaled_curr)
            
            X_train_fit_tensor = torch.from_numpy(X_train_fit_scaled.astype(np.float32)).to(hpo_device)
            y_train_fit_tensor = torch.from_numpy(y_train_fit_scaled.astype(np.float32)).to(hpo_device)

            # Create and fit final model with the best params for this year
            # We need special handling for model parameters that should be passed to the model constructor
            model_init_kwargs = {"n_feature": n_features, "n_output": 1}
            
            # Get expected args for the specific model_class constructor
            # This makes the instantiation robust to extra params from HPO
            sig = inspect.signature(model_class.__init__)
            valid_model_args = {p.name for p in sig.parameters.values() if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}
            # valid_model_args will include 'self', 'n_feature', 'n_output', and model-specific HPs like 'n_hidden1', 'dropout', etc.

            for k_orig, v_orig in current_best_params_dict.items():
                if k_orig.startswith("module__"):
                    # First remove the module__ prefix
                    k_clean = k_orig.replace("module__", "")
                    
                    # Then remove any model-specific suffix (e.g., _Net2)
                    for suffix in [f"_{model_name}", "_Net1", "_Net2", "_Net3", "_Net4", "_Net5", "_DNet1", "_DNet2", "_DNet3"]:
                        if k_clean.endswith(suffix):
                            k_clean = k_clean[:-len(suffix)]
                            break
                    
                    # Check if the cleaned parameter name is a valid model argument
                    if k_clean in valid_model_args:
                        model_init_kwargs[k_clean] = v_orig
                        print(f"Using parameter {k_orig} -> {k_clean} = {v_orig} for model {model_name}", file=sys.stderr)
                    # Optionally, print a warning for skipped params:
                    # else:
                    #     if model_name.startswith("Net") and k_clean.startswith("n_hidden") and int(k_clean[-1]) > 1 : # Net1-5 only have n_hidden2 etc.
                    #         pass # Expected for Net1-5 to not take n_hidden2 etc.
                    #     elif model_name.startswith("DNet") and k_clean == "activation_hidden":
                    #         pass # Expected for DNet to not take activation_hidden
                    #     elif model_name.startswith("Net") and k_clean == "activation_fn":
                    #         pass # Expected for Net to not take activation_fn
                    #     else:
                    #         print(f"DEBUG: For {model_name}, skipping HPO param {k_orig} (cleaned: {k_clean}) as it's not in {model_class.__name__}.__init__ args: {valid_model_args}", file=sys.stderr)

            try:
                model_instance = model_class(**model_init_kwargs)
            except TypeError as e_type:
                # Construct a more informative error message
                error_message_detail = (
                    f"ERROR: TypeError during model instantiation for {model_name} (class: {model_class.__name__}).\n"
                    f"       Attempted to initialize with kwargs: {model_init_kwargs}\n"
                    f"       Constructor signature expected args (excluding self): {[p for p in valid_model_args if p != 'self']}\n"
                    f"       Effective HPO params used to build these kwargs for this model ({model_name}): {current_best_params_dict}\n"
                    f"       Search space used for HPO for this model ({model_name}): {search_space_config_or_fn}\n"
                    f"       Original TypeError: {e_type}"
                )
                print(error_message_detail, file=sys.stderr)
                # Re-raise the original error to maintain stack trace if needed, or a new one with more info
                raise TypeError(error_message_detail) from e_type # Re-raise with more context
            except Exception as e: # Catch other potential errors during instantiation
                print(f"ERROR: Exception during model instantiation for {model_name}: {e}", file=sys.stderr)
                raise e # Re-raise the error after printing debug info

            # --------------------------------------------------
            #  Build optimiser class + kwargs for final training
            # --------------------------------------------------
            def _resolve_optimizer(opt_spec):
                """
                Accepts either:
                  • a string  -> 'Adam'           → torch.optim.Adam
                  • a class   -> torch.optim.Adam → torch.optim.Adam
                """
                if isinstance(opt_spec, str):
                    return getattr(torch.optim, opt_spec)
                elif isinstance(opt_spec, type):
                    return opt_spec
                else:
                    raise TypeError(
                        f"Unsupported optimiser spec '{opt_spec}' "
                        f"({type(opt_spec)}). Provide str name or torch.optim.* class."
                    )

            optimizer_name_or_cls = current_best_params_dict.pop("optimizer", torch.optim.Adam)
            opt_class = _resolve_optimizer(optimizer_name_or_cls)

            # Create NeuralNetRegressor with the best hyperparameters
            final_nn_model = regressor_class(
                module=model_instance,
                max_epochs=hpo_epochs,
                optimizer=opt_class,
                lr=current_best_params_dict.get("lr", 0.001), # lr is a direct Skorch param
                optimizer__weight_decay=current_best_params_dict.get("optimizer__weight_decay", 0.0), # Corrected key
                batch_size=hpo_batch_size, # batch_size is a direct Skorch param
                l1_lambda=current_best_params_dict.get("l1_lambda", 0.0), # Pass l1_lambda if your regressor (e.g., GridNet) uses it
                # Other parameters...
                iterator_train__shuffle=True,
                callbacks=[EarlyStopping(patience=10)], # This should be fine as Skorch creates a val split by default
                device=hpo_device,
            )
            final_nn_model.fit(X_train_fit_tensor, y_train_fit_tensor)
            
            if trigger_hpo_this_step: # Save the annually retrained model
                 joblib.dump(final_nn_model, paths['annual_best_models'] / f"{model_name}_{current_date_t}_model.joblib")

            # Predict for X_t (which is predictor_array[t_idx, 1:])
            X_oos_current_unscaled = predictor_array[t_idx, 1:].reshape(1, -1)
            X_oos_current_scaled = scaler_x_current.transform(X_oos_current_unscaled)
            X_oos_current_tensor = torch.from_numpy(X_oos_current_scaled.astype(np.float32)).to(hpo_device)
            
            y_pred_oos_scaled = final_nn_model.predict(X_oos_current_tensor)
            y_pred_oos_unscaled = scaler_y_current.inverse_transform(y_pred_oos_scaled.reshape(-1,1)).item()
            oos_predictions_nn[model_name].append(y_pred_oos_unscaled)

        # --- Combination Forecast (CF) ---
        # Uses individual OLS models for each predictor in `predictor_cols_for_cf`
        cf_individual_preds_curr_step = []
        for p_idx, p_name in enumerate(predictor_cols_for_cf):
            # Find the column index of this predictor in X_train_full_unscaled_curr
            # This assumes predictor_cols_for_cf are a subset of or same as features used for NNs
            try:
                original_predictor_idx = data_dict['predictor_names'].index(p_name)
            except ValueError:
                print(f"Warning: Predictor {p_name} for CF not found in main predictor list. Skipping for this CF component.", file=sys.stderr)
                cf_individual_preds_curr_step.append(np.nan)
                continue

            X_train_cf_single = X_train_full_unscaled_curr[:, [original_predictor_idx]]
            y_train_cf = y_train_full_unscaled_curr.ravel() # OLS expects 1D y
            
            if X_train_cf_single.shape[0] > 0:
                ols_model = LinearRegression()
                ols_model.fit(X_train_cf_single, y_train_cf)
                
                # Predictor value for current OOS step for this specific predictor
                X_oos_cf_single = predictor_array[t_idx, [1 + original_predictor_idx]] 
                cf_individual_preds_curr_step.append(ols_model.predict(X_oos_cf_single.reshape(1,-1))[0])
            else:
                cf_individual_preds_curr_step.append(np.nan)
        
        oos_predictions_cf.append(np.nanmean(cf_individual_preds_curr_step))

    # 3. Consolidate and Save All Raw OOS Predictions
    df_oos_predictions_nn = pd.DataFrame(oos_predictions_nn, index=dates_t_all[oos_start_idx:])
    df_oos_predictions_cf = pd.DataFrame({'CF': oos_predictions_cf}, index=dates_t_all[oos_start_idx:])
    df_oos_predictions_ha = pd.DataFrame({'HA': ha_forecasts_all[oos_start_idx:]}, index=dates_t_all[oos_start_idx:])
    df_oos_actual_log_ep = pd.DataFrame({'ActualLogEP': actual_log_ep_all[oos_start_idx:]}, index=dates_t_all[oos_start_idx:])
    df_oos_actual_mkt_ret = pd.DataFrame({'ActualMktRet': actual_market_returns_all[oos_start_idx:]}, index=dates_t_all[oos_start_idx:])
    df_oos_lagged_rf = pd.DataFrame({'LaggedRF': lagged_rf_all[oos_start_idx:]}, index=dates_t_all[oos_start_idx:])

    df_all_oos_outputs = pd.concat([
        df_oos_actual_log_ep, df_oos_actual_mkt_ret, df_oos_lagged_rf, 
        df_oos_predictions_ha, df_oos_predictions_cf, df_oos_predictions_nn
    ], axis=1)
    df_all_oos_outputs.to_csv(paths['full_predictions_file'])
    print(f"All OOS raw outputs saved to {paths['full_predictions_file']}")

    # 4. Calculate and Save Final OOS Metrics
    oos_metrics_results = []
    
    actual_log_ep_oos_final = actual_log_ep_all[oos_start_idx:]
    actual_mkt_ret_oos_final = actual_market_returns_all[oos_start_idx:]
    lagged_rf_oos_final = lagged_rf_all[oos_start_idx:]
    ha_oos_final = ha_forecasts_all[oos_start_idx:]

    # HA Metrics
    sr_ha = compute_success_ratio(actual_log_ep_oos_final, ha_oos_final) * 100
    # Pass CER_GAMMA as positional argument to avoid parameter conflict
    cer_ha = compute_CER(actual_mkt_ret_oos_final, ha_oos_final, lagged_rf_oos_final, CER_GAMMA) * 100 
    _, pt_stat_ha, pt_pval_ha = PT_test(actual_log_ep_oos_final, ha_oos_final)
    oos_metrics_results.append({
        'Model': 'HA', 'OOS_R2_vs_HA (%)': 0.0, 
        'Success_Ratio (%)': sr_ha, 'CER_annual (%)': cer_ha,
        'CW_stat': np.nan, 'CW_pvalue': np.nan, # HA is the benchmark
        'PT_stat': pt_stat_ha, 'PT_pvalue': pt_pval_ha
    })

    # CF Metrics
    cf_preds_oos_final = np.array(oos_predictions_cf)
    valid_cf_mask = ~np.isnan(cf_preds_oos_final)
    if np.sum(valid_cf_mask) > 0:
        oos_r2_cf = compute_in_r_square(actual_log_ep_oos_final[valid_cf_mask], ha_oos_final[valid_cf_mask], cf_preds_oos_final[valid_cf_mask]) * 100
        sr_cf = compute_success_ratio(actual_log_ep_oos_final[valid_cf_mask], cf_preds_oos_final[valid_cf_mask]) * 100
        # Pass CER_GAMMA as positional argument to avoid parameter conflict
        cer_cf = compute_CER(actual_mkt_ret_oos_final[valid_cf_mask], cf_preds_oos_final[valid_cf_mask], lagged_rf_oos_final[valid_cf_mask], CER_GAMMA) * 100
        cw_stat_cf, cw_pval_cf = CW_test(actual_log_ep_oos_final[valid_cf_mask], ha_oos_final[valid_cf_mask], cf_preds_oos_final[valid_cf_mask])
        _, pt_stat_cf, pt_pval_cf = PT_test(actual_log_ep_oos_final[valid_cf_mask], cf_preds_oos_final[valid_cf_mask])
    else:
        oos_r2_cf, sr_cf, cer_cf, cw_stat_cf, cw_pval_cf, pt_stat_cf, pt_pval_cf = [np.nan] * 7
    oos_metrics_results.append({
        'Model': 'CF', 'OOS_R2_vs_HA (%)': oos_r2_cf, 
        'Success_Ratio (%)': sr_cf, 'CER_annual (%)': cer_cf,
        'CW_stat': cw_stat_cf, 'CW_pvalue': cw_pval_cf,
        'PT_stat': pt_stat_cf, 'PT_pvalue': pt_pval_cf
    })

    # NN Model Metrics
    for model_name in nn_model_configs.keys():
        preds_nn_final = df_oos_predictions_nn[model_name].values # From DataFrame for easier alignment
        valid_nn_mask = ~np.isnan(preds_nn_final)
        if np.sum(valid_nn_mask) > 0:
            oos_r2_nn = compute_in_r_square(actual_log_ep_oos_final[valid_nn_mask], ha_oos_final[valid_nn_mask], preds_nn_final[valid_nn_mask]) * 100
            sr_nn = compute_success_ratio(actual_log_ep_oos_final[valid_nn_mask], preds_nn_final[valid_nn_mask]) * 100
            # Pass CER_GAMMA as positional argument to avoid parameter conflict
            cer_nn = compute_CER(actual_mkt_ret_oos_final[valid_nn_mask], preds_nn_final[valid_nn_mask], lagged_rf_oos_final[valid_nn_mask], CER_GAMMA) * 100
            cw_stat_nn, cw_pval_nn = CW_test(actual_log_ep_oos_final[valid_nn_mask], ha_oos_final[valid_nn_mask], preds_nn_final[valid_nn_mask])
            _, pt_stat_nn, pt_pval_nn = PT_test(actual_log_ep_oos_final[valid_nn_mask], preds_nn_final[valid_nn_mask])
        else:
            oos_r2_nn, sr_nn, cer_nn, cw_stat_nn, cw_pval_nn, pt_stat_nn, pt_pval_nn = [np.nan] * 7
        
        oos_metrics_results.append({
            'Model': model_name, 'OOS_R2_vs_HA (%)': oos_r2_nn, 
            'Success_Ratio (%)': sr_nn, 'CER_annual (%)': cer_nn,
            'CW_stat': cw_stat_nn, 'CW_pvalue': cw_pval_nn,
            'PT_stat': pt_stat_nn, 'PT_pvalue': pt_pval_nn
        })

    # Calculate CER gain relative to HA
    ha_cer = oos_metrics_results[0]['CER_annual (%)']  # Get HA CER value
    oos_metrics_results[0]['CER_gain_vs_HA (%)'] = 0.0  # HA has zero gain vs itself
    oos_metrics_results[1]['CER_gain_vs_HA (%)'] = oos_metrics_results[1]['CER_annual (%)'] - ha_cer
    for i in range(2, len(oos_metrics_results)):
        oos_metrics_results[i]['CER_gain_vs_HA (%)'] = oos_metrics_results[i]['CER_annual (%)'] - ha_cer

    # Create and save final metrics dataframe
    df_oos_metrics = pd.DataFrame(oos_metrics_results)
    
    # Reorder columns to put CER_gain after CER_annual
    cols = list(df_oos_metrics.columns)
    cer_idx = cols.index('CER_annual (%)')
    cols.insert(cer_idx + 1, cols.pop(cols.index('CER_gain_vs_HA (%)')))
    df_oos_metrics = df_oos_metrics[cols]
    
    print("\n--- OOS Evaluation Complete for", experiment_name_suffix, "---")
    print(df_oos_metrics)
    df_oos_metrics.to_csv(paths['metrics_file'], index=False)

    # --------------------------------
    # Print debug information to help diagnose the issue
    # --------------------------------
    print("\nHA forecast sample:", ha_oos_final[:5])
    print("CF forecast sample:", np.array(oos_predictions_cf)[:5])
    
    # Check if they're identical
    are_identical = np.allclose(ha_oos_final, np.array(oos_predictions_cf), rtol=1e-5, atol=1e-8)
    print(f"Are HA and CF identical? {are_identical}")
    
    # Compute success ratios with verbose output
    print("\nSuccess Ratio Details:")
    print("HA sign matches:", np.sum(np.sign(actual_log_ep_oos_final) == np.sign(ha_oos_final)))
    print("CF sign matches:", np.sum(np.sign(actual_log_ep_oos_final) == np.sign(np.array(oos_predictions_cf))))
    print("Total observations:", len(actual_log_ep_oos_final))
    
    # HA Model Metrics
    sr_ha = compute_success_ratio(actual_log_ep_oos_final, ha_oos_final) * 100
    # Fix: Pass CER_GAMMA as positional argument instead of keyword to avoid parameter conflict
    cer_ha = compute_CER(actual_mkt_ret_oos_final, ha_oos_final, lagged_rf_oos_final, CER_GAMMA) * 100
    
    # ... rest of your code ...

    return df_oos_metrics, df_all_oos_outputs