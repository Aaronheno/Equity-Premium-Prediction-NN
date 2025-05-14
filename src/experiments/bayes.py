import sys
from pathlib import Path
import os # Import os for path manipulation if needed, though Path is often sufficient

# --- Add project root to sys.path ---
# This ensures Python and linters can find the 'src' package
# Assumes this script is in src/experiments/bayes.py
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
# --- End sys.path modification ---

# --- Now perform imports ---
import time
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import optuna
from datetime import datetime

# Imports can now be absolute from src or relative
# Using absolute is often clearer if sys.path is set correctly
from src.models import nns
from src.configs.search_spaces import BAYES as SPACE
from src.utils.metrics import compute_in_r_square, compute_success_ratio, compute_CER
from src.utils.processing import scale_data
from src.utils.training_optuna import run_study
from src.utils.io import X_ALL, Y_ALL, RF_ALL, train_val_split
from skorch import NeuralNetRegressor

def run(
    models=None, 
    trials=5, 
    epochs=10,  
    batch=None, 
    threads=1, 
    device="cpu",
    gamma=3.0
):
    """
    Run Bayesian hyperparameter optimization for neural network models.
    
    Parameters:
    -----------
    models : list or None
        List of model names to optimize. If None, optimize all models.
    trials : int, default=50
        Number of trials for Bayesian optimization per model.
    epochs : int, default=50 
        Number of epochs for training each model.
    batch : int or None
        Batch size for training. If None, use values from search space.
    threads : int, default=1
        Number of parallel threads for optimization.
    device : str, default="cpu"
        Device to use for training ('cpu' or 'cuda').
    gamma : float, default=3.0
        Risk aversion parameter for CER calculation.
    """
    start_time = time.time()

    # --- Device Check ---
    if device == "cuda" and not torch.cuda.is_available():
        print("--- WARNING: CUDA requested but not available. Falling back to CPU. ---")
        device = "cpu" # Force CPU if CUDA isn't available
    print(f"--- Using device: {device} ---")
    # --- End Device Check ---

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    # --- Output Path ---
    # Use Path relative to the project root (which is now likely the CWD when run via `python -m`)
    out_base = Path("./runs/0_Bayesian_Optimisation_In_Sample")
    run_name = f"{timestamp}_{device}_all_models_bayes_insample"
    out = out_base / run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions").mkdir(exist_ok=True)
    (out / "studies").mkdir(exist_ok=True)
    print(f"Output directory: {out}")

    # --- Load Data ---
    y_ALL_df = Y_ALL # DataFrame from io.py
    X_ALL_df = X_ALL # DataFrame from io.py
    RF_ALL_df = RF_ALL # DataFrame from io.py
    print(f"Data loaded successfully. Shapes: X_ALL={X_ALL_df.shape}, y_ALL={y_ALL_df.shape}, RF_ALL={RF_ALL_df.shape}")

    # --- Scale Data ---
    X_ALL_scaled, y_ALL_scaled, scaler_x, scaler_y = scale_data(X_ALL_df, y_ALL_df)
    
    # Check if the scaled data is already numpy arrays or still DataFrames
    if hasattr(X_ALL_scaled, 'values'):
        # It's a DataFrame, convert to NumPy
        X_ALL_scaled_np = X_ALL_scaled.values
        y_ALL_scaled_np = y_ALL_scaled.values
    else:
        # Already a NumPy array
        X_ALL_scaled_np = X_ALL_scaled
        y_ALL_scaled_np = y_ALL_scaled
    
    # Save the index before scaling for later use
    index = X_ALL_df.index
    
    # --- Define actuals and risk-free BEFORE loop ---
    actual_ALL_unscaled = y_ALL_df.values # Numpy array for metrics
    rf_full = RF_ALL_df.values # Numpy array for metrics
    print(f"\nShape of actual_ALL_unscaled: {actual_ALL_unscaled.shape}")

    # --- Benchmark: Historical Average (Expanding Window) ---
    # Calculate HA on the original, unscaled target variable DataFrame
    y_pred_HA_expanding_series = y_ALL_df.iloc[:, 0].expanding(min_periods=1).mean().shift(1)
    # Fill the first NaN. Use the first value of the *original* unscaled series
    first_actual_val = y_ALL_df.iloc[0, 0]
    y_pred_HA_expanding_series = y_pred_HA_expanding_series.fillna(first_actual_val)
    y_pred_HA_expanding = y_pred_HA_expanding_series.values.reshape(-1, 1) # Numpy array for metrics
    print("Expanding Historical Average benchmark calculated.")
    print(f"Shape of y_pred_HA_expanding: {y_pred_HA_expanding.shape}")

    # --- Train/validation split for hyperparameter tuning ---
    # Create temporary DataFrames if needed for train_val_split
    if not hasattr(X_ALL_scaled, 'index'):
        # Convert arrays back to DataFrames temporarily
        X_ALL_scaled_df = pd.DataFrame(X_ALL_scaled, index=index)
        y_ALL_scaled_df = pd.DataFrame(y_ALL_scaled, index=index)
        
        X_tr_df, X_val_df, y_tr_df, y_val_df = train_val_split(
            X_ALL_scaled_df, y_ALL_scaled_df, val_ratio=0.15, split_by_index=True
        )
    else:
        # Already DataFrames
        X_tr_df, X_val_df, y_tr_df, y_val_df = train_val_split(
            X_ALL_scaled, y_ALL_scaled, val_ratio=0.15, split_by_index=True
        )
    
    # Convert to numpy arrays after split if not already
    X_tr = X_tr_df.values if hasattr(X_tr_df, 'values') else X_tr_df
    X_val = X_val_df.values if hasattr(X_val_df, 'values') else X_val_df
    y_tr = y_tr_df.values if hasattr(y_tr_df, 'values') else y_tr_df
    y_val = y_val_df.values if hasattr(y_val_df, 'values') else y_val_df

    # --- Store results ---
    all_trials_results = {}
    best_params_all_models = {}
    validation_mses = {}

    # --- Run Optuna Study for each model ---
    study_objects = {} # To store optuna study objects
    final_predictions = {} # To store final model predictions on the full dataset
    final_predictions_metrics = {} # Initialize the dictionary HERE

    # --- Bayesian Optimization Loop ---
    start_time_bayes = time.time()
    # Loop through model names in the list
    for model_name in models:
        print(f"\n--- Starting Bayesian Optimization for {model_name} ---")
        
        # Define objective function inline (similar to your original code)
        def objective(trial):
            # Get model class
            model_class = getattr(nns, model_name)
            
            # Get search space for this model
            search_space = SPACE.get(model_name, {})
            
            # Extract hyperparameters from trial
            params = {}
            fixed_params = search_space.get('fixed', {})
            
            # Add fixed parameters
            if fixed_params:
                for k, v in fixed_params.items():
                    params[k] = v
            
            # Add sampled parameters
            for param_name, param_config in search_space.items():
                if param_name == 'fixed':
                    continue
                
                # Handle different parameter types based on their values
                if isinstance(param_config, list):
                    # For categorical parameters (lists of options)
                    params[param_name] = trial.suggest_categorical(param_name, param_config)
                
                elif isinstance(param_config, tuple) and len(param_config) >= 2:
                    # For numeric range parameters (tuples of min, max)
                    if isinstance(param_config[0], float) or isinstance(param_config[1], float):
                        # Float parameter
                        low, high = param_config[0], param_config[1]
                        # Check if the range spans orders of magnitude, suggesting log scale
                        use_log = (high / max(abs(low), 1e-10) > 100) if low != 0 else False
                        params[param_name] = trial.suggest_float(param_name, low, high, log=use_log)
                    else:
                        # Int parameter
                        params[param_name] = trial.suggest_int(param_name, param_config[0], param_config[1])
                
                elif isinstance(param_config, (int, float)):
                    # For fixed parameters
                    params[param_name] = param_config
            
            # Handle module__ prefix for skorch if needed
            skorch_params = {}
            for k, v in params.items():
                # Special handling for n_hidden from ranges to actual params
                if k.endswith('_range'):
                    # Extract base name (e.g., n_hidden1 from n_hidden1_range)
                    base_name = k[:-6]  # Remove _range
                    if base_name not in params:  # Only add if not already set
                        skorch_params[f'module__{base_name}'] = v
                elif k.startswith('n_hidden') or k == 'dropout' or k == 'l1_lambda':
                    skorch_params[f'module__{k}'] = v
                elif k != 'batch_size':  # Skip batch_size to avoid duplicate
                    skorch_params[k] = v
            
            # Add required parameters for neural network initialization
            skorch_params['module__n_feature'] = X_tr.shape[1]  # Number of input features
            skorch_params['module__n_output'] = 1  # Regression task with one output

            # Remove input_dim if present (to avoid conflicts)
            if 'module__input_dim' in skorch_params:
                del skorch_params['module__input_dim']
            
            # Create model with parameters
            model = NeuralNetRegressor(
                module=model_class,
                max_epochs=epochs,
                batch_size=batch,  # Use the batch from function args, not search space
                optimizer=torch.optim.Adam,
                device=device,
                **skorch_params
            )
            
            # Convert to tensors
            X_tr_tensor = torch.tensor(X_tr, dtype=torch.float32)
            y_tr_tensor = torch.tensor(y_tr, dtype=torch.float32)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            
            # Train model
            try:
                model.fit(X_tr_tensor, y_tr_tensor)
                
                # Predict on validation set
                y_val_pred = model.predict(X_val_tensor)
                
                # Calculate MSE
                val_mse = ((y_val_pred - y_val.flatten()) ** 2).mean()
                return val_mse
            except Exception as e:
                print(f"Error in trial for {model_name}: {e}")
                return float('inf')
        
        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=trials, n_jobs=threads)
        
        # Get best parameters and store
        best_params = study.best_params
        best_value = study.best_value
        best_params_all_models[model_name] = best_params
        validation_mses[model_name] = best_value
        
        print(f"Best parameters for {model_name}: {best_params}")
        print(f"Best validation MSE: {best_value:.6f}")
        
        # Save study
        joblib.dump(study, out / "studies" / f"{model_name}_study.pkl")

    # --- Train Final Models with Best Params ---
    print("\n--- Training Final Models with Best Parameters ---")
    final_predictions = {}
    
    # Convert full dataset to tensors for final training
    X_ALL_tensor = torch.tensor(X_ALL_scaled_np, dtype=torch.float32).to(device)
    y_ALL_tensor = torch.tensor(y_ALL_scaled_np, dtype=torch.float32).to(device)
    
    for model_name, best_params in best_params_all_models.items():
        print(f"Training final {model_name} model...")
        
        # Get model class
        model_class = getattr(nns, model_name)
        
        # Get fixed parameters
        search_space = SPACE.get(model_name, {})
        fixed_params = search_space.get('fixed', {})
        
        # Combine fixed and best parameters
        final_params = {**fixed_params, **best_params}
        
        # Format parameters for skorch
        skorch_final_params = {}
        for k, v in final_params.items():
            if k.endswith('_range'):
                # Extract base name (e.g., n_hidden1 from n_hidden1_range)
                base_name = k[:-6]  # Remove _range
                if base_name not in final_params:  # Only add if not already set
                    skorch_final_params[f'module__{base_name}'] = v
            elif k.startswith('n_hidden') or k == 'dropout' or k == 'l1_lambda':
                skorch_final_params[f'module__{k}'] = v
            elif k != 'batch_size':  # Skip batch_size to avoid duplicate
                skorch_final_params[k] = v
        
        # Add required parameters for neural network initialization
        skorch_final_params['module__n_feature'] = X_ALL_scaled_np.shape[1]  # Number of input features
        skorch_final_params['module__n_output'] = 1  # Regression task with one output

        # Remove input_dim if present (to avoid conflicts)
        if 'module__input_dim' in skorch_final_params:
            del skorch_final_params['module__input_dim']
        
        # Create final model
        final_model = NeuralNetRegressor(
            module=model_class,
            max_epochs=epochs,
            batch_size=batch,  # Use the batch from function args, not search space
            optimizer=torch.optim.Adam,
            device=device,
            **skorch_final_params
        )
        
        # Train on full dataset
        try:
            final_model.fit(X_ALL_tensor, y_ALL_tensor)
            
            # Generate predictions
            preds_scaled = final_model.predict(X_ALL_tensor)
            
            # Inverse transform predictions
            preds_unscaled = scaler_y.inverse_transform(
                preds_scaled.reshape(-1, 1)
            )
            
            # Store predictions
            final_predictions[model_name] = preds_unscaled
            
            # Calculate metrics
            r2 = compute_in_r_square(
                actual_ALL_unscaled, 
                y_pred_HA_expanding, 
                preds_unscaled
            ) * 100
            
            sr = compute_success_ratio(
                actual_ALL_unscaled, 
                preds_unscaled
            ) * 100

            # Ensure inputs to CER are 1D
            actual_for_cer = actual_ALL_unscaled.ravel()
            preds_model_for_cer = preds_unscaled.ravel()
            rf_for_cer = rf_full.ravel() # Assuming rf_full is already aligned and 1D

            # Align lengths just in case, though they should be from data prep
            min_len_cer = min(len(actual_for_cer), len(preds_model_for_cer), len(rf_for_cer))
            actual_for_cer = actual_for_cer[:min_len_cer]
            preds_model_for_cer = preds_model_for_cer[:min_len_cer]
            rf_for_cer = rf_for_cer[:min_len_cer]
            
            # CER for the model
            cer_model_val = compute_CER(
                actual_for_cer, 
                preds_model_for_cer, 
                rf_for_cer, 
                gamma=gamma # Use the passed gamma
            ) * 100 # Multiply by 100 for percentage points
            
            # CER for the HA benchmark
            # Ensure y_pred_HA_expanding is also aligned and 1D
            preds_ha_for_cer = y_pred_HA_expanding.ravel()[:min_len_cer]

            cer_ha_val = compute_CER(
                actual_for_cer,
                preds_ha_for_cer, # HA's predictions
                rf_for_cer,
                gamma=gamma
            ) * 100

            cer_gain_val = cer_model_val - cer_ha_val # Already calculated with values * 100
            
            # Store metrics for this model (to be aggregated later)
            # This part seems to be inside a loop that processes one model at a time
            # The aggregation into the 'metrics' list happens outside this try/except block
            # So, we need to store these calculated values to be used in that aggregation.
            # Let's assume validation_mses, r2_values, sr_values, cer_model_values, cer_ha_values, cer_gain_values are dicts
            
            # Storing them in dictionaries keyed by model_name
            # These dictionaries should be initialized before the model loop
            # For example: r2_values = {}, sr_values = {}, etc.
            
            # This structure is a bit different from random.py and grid.py where metrics are appended directly.
            # The existing code appends to 'metrics' list *after* the loop over models.
            # Let's adapt to put the detailed CER values into the final_predictions_metrics dictionary
            
            if model_name not in final_predictions_metrics:
                final_predictions_metrics[model_name] = {}
            
            final_predictions_metrics[model_name]['r2'] = r2
            final_predictions_metrics[model_name]['sr'] = sr
            final_predictions_metrics[model_name]['cer_model'] = cer_model_val
            final_predictions_metrics[model_name]['cer_ha'] = cer_ha_val
            final_predictions_metrics[model_name]['cer_gain_vs_ha'] = cer_gain_val
            
            # Store predictions
            pd.DataFrame({
                'Actual': actual_ALL_unscaled.flatten(),
                'Predicted': preds_unscaled.flatten(),
                'HA': y_pred_HA_expanding.flatten()
            }).to_csv(out / "predictions" / f"{model_name}_predictions.csv", index=False)
            
        except Exception as e:
            print(f"Error training final {model_name} model: {e}")
            final_predictions[model_name] = np.full_like(actual_ALL_unscaled, np.nan)

    # --- Aggregate and Save Metrics ---
    metrics = []
    for model_name_iter in models: # Use models_to_run for iteration
        if model_name_iter in final_predictions_metrics and model_name_iter in study_objects: # Check if model ran and has metrics
            model_metrics = final_predictions_metrics[model_name_iter]
            study = study_objects[model_name_iter]
            best_val_mse_for_model = np.nan
            if study and study.best_trial:
                best_val_mse_for_model = study.best_trial.value

            metrics.append({
                'Model': model_name_iter,
                'Validation MSE': best_val_mse_for_model,
                'In-sample R2 (%)': model_metrics.get('r2', np.nan),
                'Success Ratio (%)': model_metrics.get('sr', np.nan),
                'CER Model (%, ann.)': model_metrics.get('cer_model', np.nan),
                'CER HA (%, ann.)': model_metrics.get('cer_ha', np.nan),
                'CER Gain vs HA (%, ann.)': model_metrics.get('cer_gain_vs_ha', np.nan),
                'Best Hyperparameters': str(study.best_trial.params) if study and study.best_trial else "N/A"
            })
        elif model_name_iter in study_objects: # Model was attempted but might have failed before metrics
            study = study_objects[model_name_iter]
            best_val_mse_for_model = np.nan
            best_params_str = "N/A (Failed or no best trial)"
            if study:
                try:
                    if study.best_trial:
                        best_val_mse_for_model = study.best_trial.value
                        best_params_str = str(study.best_trial.params)
                except ValueError: # No trials completed
                    pass
            
            metrics.append({
                'Model': model_name_iter,
                'Validation MSE': best_val_mse_for_model,
                'In-sample R2 (%)': np.nan,
                'Success Ratio (%)': np.nan,
                'CER Model (%, ann.)': np.nan,
                'CER HA (%, ann.)': np.nan,
                'CER Gain vs HA (%, ann.)': np.nan,
                'Best Hyperparameters': best_params_str
            })
        # else: model was not in models_to_run or had other issues

    # Add HA benchmark row
    # Ensure actual_ALL_unscaled and y_pred_HA_expanding are 1D for SR and CER
    actual_for_ha_bench = actual_ALL_unscaled.ravel()[:min_len_cer] # Use min_len_cer from model loop
    preds_ha_for_ha_bench = y_pred_HA_expanding.ravel()[:min_len_cer]
    rf_for_ha_bench = rf_full.ravel()[:min_len_cer]


    ha_sr = compute_success_ratio(actual_for_ha_bench, preds_ha_for_ha_bench) * 100
    cer_ha_benchmark_val = compute_CER(
        actual_for_ha_bench,
        preds_ha_for_ha_bench,
        rf_for_ha_bench,
        gamma=gamma # Use the run's gamma
    ) * 100
    
    metrics.append({
        'Model': 'HA_Benchmark',
        'Validation MSE': np.nan,
        'In-sample R2 (%)': 0.0,
        'Success Ratio (%)': ha_sr,
        'CER Model (%, ann.)': cer_ha_benchmark_val, # CER of HA is its own "model" CER
        'CER HA (%, ann.)': cer_ha_benchmark_val,   # CER of HA is also the benchmark CER
        'CER Gain vs HA (%, ann.)': 0.0,           # HA vs HA gain is 0
        'Best Hyperparameters': "N/A"
    })
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics) # Use the aggregated metrics list
    metrics_df.to_csv(out / "final_metrics_bayes_search.csv", index=False) # Consistent filename
    print(f"Final metrics saved to {out}/final_metrics_bayes_search.csv")
    
    # Print total time
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")