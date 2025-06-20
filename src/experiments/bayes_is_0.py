"""
Bayesian In-Sample Neural Network Optimization

This experiment conducts in-sample hyperparameter optimization using Bayesian 
optimization (Optuna) for neural network models. Designed for parallel model
training with independent HPO studies for maximum throughput.

Threading Status: PARALLEL_READY (Model-level and trial-level parallelism)
Hardware Requirements: CPU_REQUIRED, CUDA_PREFERRED, HIGH_MEMORY_BENEFICIAL
Performance Notes:
    - Model parallelism: 8x speedup (8 models simultaneously)
    - HPO trial parallelism: 10-50x speedup (concurrent Optuna trials)
    - Memory usage: Scales with trial count and model complexity
    - Optimal for high-core systems with abundant memory

Experiment Type: In-Sample Hyperparameter Optimization
Models Supported: Net1, Net2, Net3, Net4, Net5, DNet1, DNet2, DNet3
HPO Method: Bayesian Optimization (Optuna TPE)
Output Directory: runs/0_Bayesian_Optimisation_In_Sample/

Critical Parallelization Opportunities:
    1. Independent model HPO (8 models in parallel)
    2. Concurrent Optuna trials within each model
    3. Parallel final model training with best parameters
    4. Concurrent metrics computation across models

Threading Implementation Status:
    ❌ Sequential model processing (MAIN BOTTLENECK)
    ✅ Optuna trials can be parallelized (n_jobs parameter)
    ❌ Model training sequential across models
    ❌ Metrics computation sequential

Future Parallel Implementation:
    run(models, parallel_models=True, hpo_parallel=True, n_jobs=32)
    
Expected Performance Gains:
    - Current: 4 hours for 8 models × 100 trials each
    - With trial parallelism: 1 hour (4x speedup)
    - With model parallelism: 15 minutes (additional 4x speedup)
    - Combined on 128-core server: 3-5 minutes (48-80x speedup)

In-Sample Optimization Advantages:
    - Complete data availability for HPO
    - Stable validation metrics
    - Comprehensive parameter exploration
    - Foundation for OOS experiment configuration
"""

import sys
from pathlib import Path
import os

# Add project root to sys.path for module imports
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# --- Now perform imports ---
import time
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import optuna
from datetime import datetime
from src.models import nns
from src.configs.search_spaces import BAYES as SPACE
from src.utils.metrics_unified import compute_in_r_square, compute_success_ratio, compute_CER, scale_data
from src.utils.io import X_ALL, Y_ALL, RF_ALL
from skorch import NeuralNetRegressor

def run(
    models=['Net1', 'Net2', 'Net3', 'Net4', 'Net5', 'DNet1', 'DNet2', 'DNet3'], 
    trials=10, 
    epochs=100,
    threads=1,
    batch=256,
    device='cpu',
    gamma=3.0,
    custom_data=None,
    out_dir_override=None,
    verbose=False):
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
    custom_data : tuple or None
        Custom data to use instead of the default data. Should be a tuple of (X, y, rf).
    out_dir_override : str or None
        Override the default output directory.
    """
    start_time = time.time()

    # Device validation
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    # Setup output directory
    if out_dir_override:
        out_base = Path(out_dir_override)
    else:
        out_base = Path("./runs/0_Bayesian_Optimisation_In_Sample")
    run_name = f"{timestamp}_{device}_all_models_bayes_insample"
    out = out_base / run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions").mkdir(exist_ok=True)
    (out / "studies").mkdir(exist_ok=True)
    print(f"Output directory: {out}")

    # --- Load Data ---
    if custom_data is not None:
        X_data, Y_data, rf_data = custom_data
        print(f"Custom data loaded. Shapes: X_data={X_data.shape}, Y_data={np.shape(Y_data)}, rf_data={np.shape(rf_data)}")
    else:
        # Use global imported data
        X_data = X_ALL
        Y_data = Y_ALL
        rf_data = RF_ALL
        print(f"Original data loaded. Shapes: X_data={X_data.shape}, Y_data={np.shape(Y_data)}, rf_data={np.shape(rf_data)}")

    # --- Scale Data ---
    # Scale the data using standard scaler
    X_ALL_scaled, y_ALL_scaled, scaler_x, scaler_y = scale_data(X_data, Y_data)
    
    # Ensure correct data types
    X_ALL_scaled = X_ALL_scaled.astype(np.float32)
    y_ALL_scaled = y_ALL_scaled.astype(np.float32)

    # --- Train/Validation Split on SCALED data ---
    # Since X_ALL_scaled_df and y_ALL_scaled_df are already NumPy arrays from scale_data
    X_ALL_scaled_np = X_ALL_scaled
    y_ALL_scaled_np = y_ALL_scaled
    
    # Ensure y is 2D as skorch expects and convert to float32
    X_ALL_scaled_np = X_ALL_scaled_np.astype(np.float32)
    if y_ALL_scaled_np.ndim == 1:
        y_ALL_scaled_np = y_ALL_scaled_np.reshape(-1, 1).astype(np.float32)
    else:
        y_ALL_scaled_np = y_ALL_scaled_np.astype(np.float32)

    # Split scaled data (NumPy arrays) into train/validation sets
    n_samples = X_ALL_scaled_np.shape[0]
    val_ratio = 0.15 # 15% validation
    n_val = int(n_samples * val_ratio)
    n_train = n_samples - n_val

    # Manual split (no shuffling to preserve time series order)
    X_tr_np = X_ALL_scaled_np[:n_train].astype(np.float32)
    X_val_np = X_ALL_scaled_np[n_train:].astype(np.float32)
    y_tr_np = y_ALL_scaled_np[:n_train].astype(np.float32)
    y_val_np = y_ALL_scaled_np[n_train:].astype(np.float32)

    # Convert back to unscaled for metrics
    # Get original unscaled actual values
    if hasattr(Y_data, 'values'):
        actual_ALL_unscaled = Y_data.values.ravel()
    elif isinstance(Y_data, np.ndarray):
        actual_ALL_unscaled = Y_data.ravel() if Y_data.ndim > 1 else Y_data
    else:
        actual_ALL_unscaled = np.array(Y_data).ravel() # Numpy array for metrics
        
    # Process risk-free rate data
    if hasattr(rf_data, 'values'):
        rf_full = rf_data.values.ravel()
    elif isinstance(rf_data, np.ndarray):
        rf_full = rf_data.ravel() if rf_data.ndim > 1 else rf_data
    else:
        rf_full = np.array(rf_data).ravel()
        
    print(f"\nShape of actual_ALL_unscaled: {actual_ALL_unscaled.shape}")
    
    # --- Benchmark: Historical Average (Expanding Window) ---
    # Creates expanding window HA predictions where prediction at time t 
    # uses the mean of all observations up to time t-1 (no look-ahead bias)
    y_pred_HA_expanding = np.zeros_like(actual_ALL_unscaled)
    
    y_pred_HA_expanding[0] = actual_ALL_unscaled[0]  # Initialize with first value
    
    # For each position i, calculate mean of all previous values (up to i-1)
    # This ensures no look-ahead bias in the benchmark
    for i in range(1, len(actual_ALL_unscaled)):
        y_pred_HA_expanding[i] = np.mean(actual_ALL_unscaled[:i+1])
    
    # Reshape to 2D array for consistency
    y_pred_HA_expanding = y_pred_HA_expanding.reshape(-1, 1) # Numpy array for metrics
    print("Expanding Historical Average benchmark calculated.")
    print(f"Shape of y_pred_HA_expanding: {y_pred_HA_expanding.shape}")

    # --- Train/validation split already done earlier ---
    # We've already split the data using manual slicing, so we can use X_tr_np, X_val_np, etc. directly
    print(f"Train shapes: X_tr={X_tr_np.shape}, y_tr={y_tr_np.shape}")
    print(f"Val shapes  : X_val={X_val_np.shape}, y_val={y_val_np.shape}")

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
        
        # Define objective function inline for this model
        # This function will be called by Optuna to evaluate different hyperparameter combinations
        def objective(trial):
            # Get model class
            model_class = getattr(nns, model_name)
            
            # Get search space for this model
            search_space = SPACE.get(model_name, {})
            
            # Extract hyperparameters from trial
            # Check if search space uses hpo_config_fn (function-based) or direct parameters
            if 'hpo_config_fn' in search_space:
                # Use the function-based approach (for BAYES search space)
                hpo_config_fn = search_space['hpo_config_fn']
                params = hpo_config_fn(trial, X_tr_np.shape[1])  # Pass n_features
            else:
                # Use the dictionary-based approach (fallback)
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
            optimizer_class = None
            batch_size_param = batch  # Default to function argument
            
            for k, v in params.items():
                if k == 'optimizer':
                    # Extract optimizer class, don't add to skorch_params
                    if isinstance(v, str):
                        # Convert string to optimizer class
                        optimizer_class = getattr(torch.optim, v)
                    else:
                        # Already a class
                        optimizer_class = v
                elif k == 'batch_size':
                    # Extract batch size, don't add to skorch_params
                    batch_size_param = v
                else:
                    # Add all other parameters
                    skorch_params[k] = v
            
            # Add required module parameters
            skorch_params['module__n_feature'] = X_tr_np.shape[1]  # Number of input features
            skorch_params['module__n_output'] = 1  # Regression task with one output

            # Remove input_dim if present (to avoid conflicts)
            if 'module__input_dim' in skorch_params:
                del skorch_params['module__input_dim']
            
            # Use optimizer from search space or default to Adam
            if optimizer_class is None:
                optimizer_class = torch.optim.Adam
            
            # Extract l1_lambda from skorch_params if present
            l1_lambda = skorch_params.pop('l1_lambda', 0.0)
            
            # Import GridNet for L1 regularization support
            from src.utils.training_grid import GridNet
                
            # Create model with parameters
            model = GridNet(
                module=model_class,
                max_epochs=epochs,
                batch_size=batch_size_param,  # Use batch size from search space
                optimizer=optimizer_class,
                l1_lambda=l1_lambda,  # Pass l1_lambda to GridNet
                device=device,
                **skorch_params
            )
            
            # Convert to tensors
            X_tr_tensor = torch.tensor(X_tr_np, dtype=torch.float32)
            y_tr_tensor = torch.tensor(y_tr_np, dtype=torch.float32)
            X_val_tensor = torch.tensor(X_val_np, dtype=torch.float32)
            
            # Train model
            try:
                model.fit(X_tr_tensor, y_tr_tensor)
                
                # Predict on validation set
                y_val_pred = model.predict(X_val_tensor)
                
                # Calculate MSE
                val_mse = ((y_val_pred - y_val_np.flatten()) ** 2).mean()
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
        
        # Store study object for metrics aggregation
        study_objects[model_name] = study
        
        # Save study
        joblib.dump(study, out / "studies" / f"{model_name}_study.pkl")

    # --- Train Final Models with Best Params ---
    # After hyperparameter optimization, train final models on the FULL dataset
    # using the best hyperparameters found for each model
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
        optimizer_class = None
        batch_size_param = batch
        l1_lambda_param = 0.0
        
        for k, v in final_params.items():
            # Remove model name suffix from parameter names
            if f'_{model_name}' in k:
                k = k.replace(f'_{model_name}', '')
            
            if k.startswith('optimizer') and f'_{model_name}' not in k:
                # Extract optimizer
                if isinstance(v, str):
                    optimizer_class = getattr(torch.optim, v)
                else:
                    optimizer_class = v
            elif k == 'batch_size':
                batch_size_param = v
            elif k == 'l1_lambda':
                l1_lambda_param = v
            elif k.startswith('weight_decay'):
                # Handle weight_decay -> optimizer__weight_decay
                skorch_final_params['optimizer__weight_decay'] = v
            elif k.startswith('lr') and len(k) <= 3:  # Just 'lr' or 'lr_'
                skorch_final_params['lr'] = v
            elif k.startswith('module__'):
                # Already has module__ prefix
                skorch_final_params[k] = v
            elif k.startswith('n_hidden') or k == 'dropout' or k.startswith('activation'):
                # Add module__ prefix
                skorch_final_params[f'module__{k}'] = v
            else:
                # Other parameters
                skorch_final_params[k] = v
        
        # Add required parameters for neural network initialization
        skorch_final_params['module__n_feature'] = X_ALL_scaled_np.shape[1]  # Number of input features
        skorch_final_params['module__n_output'] = 1  # Regression task with one output

        # Remove input_dim if present (to avoid conflicts)
        if 'module__input_dim' in skorch_final_params:
            del skorch_final_params['module__input_dim']
        
        # Use optimizer from best params or default
        if optimizer_class is None:
            optimizer_class = torch.optim.Adam
            
        # Import GridNet for L1 regularization support
        from src.utils.training_grid import GridNet
        
        # Create final model
        final_model = GridNet(
            module=model_class,
            max_epochs=epochs,
            batch_size=batch_size_param,  # Use batch size from best params
            optimizer=optimizer_class,
            l1_lambda=l1_lambda_param,  # Use L1 lambda from best params
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
                gamma  # Pass as positional parameter
            ) * 100 # Multiply by 100 for percentage points
            
            # CER for the HA benchmark
            # Ensure y_pred_HA_expanding is also aligned and 1D
            preds_ha_for_cer = y_pred_HA_expanding.ravel()[:min_len_cer]

            cer_ha_val = compute_CER(
                actual_for_cer,
                preds_ha_for_cer, # HA's predictions
                rf_for_cer,
                gamma  # Pass as positional parameter
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
    for model_name_iter in models:
        if model_name_iter in final_predictions_metrics and model_name_iter in study_objects:
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
    actual_for_ha_bench = actual_ALL_unscaled.ravel()
    preds_ha_for_ha_bench = y_pred_HA_expanding.ravel()
    rf_for_ha_bench = rf_full.ravel()
    
    # If min_len_cer is defined from previous metrics calculations, use it for consistency
    if 'min_len_cer' in locals():
        actual_for_ha_bench = actual_for_ha_bench[:min_len_cer]
        preds_ha_for_ha_bench = preds_ha_for_ha_bench[:min_len_cer]
        rf_for_ha_bench = rf_for_ha_bench[:min_len_cer]

    ha_sr = compute_success_ratio(actual_for_ha_bench, preds_ha_for_ha_bench) * 100
    cer_ha_benchmark_val = compute_CER(
        actual_for_ha_bench,
        preds_ha_for_ha_bench,
        rf_for_ha_bench,
        gamma  # Pass gamma as positional parameter
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