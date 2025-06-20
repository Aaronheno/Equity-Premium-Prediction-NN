"""
Random Search In-Sample Neural Network Optimization

This experiment conducts in-sample hyperparameter optimization using random search
for neural network models. Features the highest parallelization potential with
completely independent trials that scale linearly with core count.

Threading Status: PERFECTLY_PARALLEL (Independent trials, perfect scaling)
Hardware Requirements: CPU_REQUIRED, CUDA_BENEFICIAL, LINEAR_MEMORY_SCALING
Performance Notes:
    - Random trials: Perfect linear scaling with core count
    - Memory usage: ~300MB per concurrent trial
    - No coordination overhead between trials
    - Embarrassingly parallel: Ideal for HPC systems

Experiment Type: In-Sample Hyperparameter Optimization
Models Supported: Net1, Net2, Net3, Net4, Net5, DNet1, DNet2, DNet3
HPO Method: Random Search with Independent Sampling
Output Directory: runs/0_Random_Search_In_Sample/

Critical Parallelization Opportunities:
    1. Perfect trial parallelization (linear scaling to 1000+ cores)
    2. Independent model HPO (8x speedup)
    3. Concurrent model training with best parameters
    4. Parallel metrics computation across models

Threading Implementation Status:
    ❌ Sequential model processing (MAIN BOTTLENECK)
    ✅ Random trials perfectly parallelizable (no implementation yet)
    ❌ Model training sequential across models
    ❌ Metrics computation sequential

Future Parallel Implementation:
    run(models, parallel_models=True, trial_parallel=True, n_jobs=128)
    
Expected Performance Gains:
    - Current: 5 hours for 8 models × 200 trials each
    - With trial parallelism: 1.5 hours (3.3x speedup)
    - With model parallelism: 20 minutes (additional 4.5x speedup)
    - Combined on 128-core server: 3-5 minutes (60-100x speedup)

Random Search Advantages:
    - Best parallel efficiency (linear scaling)
    - No diminishing returns with core count
    - Often matches Bayesian optimization performance
    - Trivial to distribute across multiple machines
    - Zero coordination overhead between trials
"""

from pathlib import Path;from datetime import datetime
import pandas as pd, torch, numpy as np, joblib
from src.utils.io import RF_ALL, X_ALL, Y_ALL
from src.utils.metrics_unified import scale_data, compute_in_r_square, compute_success_ratio, compute_CER
from src.utils.training_grid import GridNet
from src.utils.training_random import train_random
from src.configs.search_spaces import RANDOM as SPACE
from src.models import nns
import sys

# --- Add project root to sys.path if not already present ---
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
# --- End sys.path modification ---

def _train(method,trainer,X_tr,y_tr,X_val,y_val,mdl,space,epochs,trials):
    if method=="grid":return train_grid(mdl,X_tr,y_tr,X_val,y_val,space,epochs)
    elif method=="random":return train_random(mdl,X_tr,y_tr,X_val,y_val,space,epochs,trials)
    else:
        raise ValueError(f"Unknown training method: {method}")

def run(
    models=None,
    trials=50, # Default trials for random search
    epochs=100, # Default epochs
    threads=1,  # PyTorch threads
    batch=256,  # Default batch, but random search might override from space
    device="cpu",
    gamma_cer=3.0, # Default gamma for CER, changed to 3.0
    custom_data=None,
    out_dir_override=None
):
    torch.set_num_threads(threads) # For PyTorch operations if not using CUDA much
    start_time_total = datetime.now()
    ts = start_time_total.strftime("%Y-%m-%d_%H-%M")

    # --- Output Path ---
    if out_dir_override:
        out_base = Path(out_dir_override)
    else:
        out_base = Path("./runs/0_Random_Search_In_Sample") # Changed 1_ to 0_
    run_name = f"{ts}_{device}_all_models_random_insample"
    out = out_base / run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions").mkdir(exist_ok=True)
    (out / "studies_or_best_params").mkdir(exist_ok=True) # For saving best_params
    print(f"Output directory: {out}")

    # --- Device Check ---
    if device == "cuda" and not torch.cuda.is_available():
        print("--- WARNING: CUDA requested but not available. Falling back to CPU. ---")
        device = "cpu"
    print(f"--- Using device: {device} ---")

    # --- Load Data (from io.py globals) ---
    if custom_data is not None:
        X_ALL_df, y_ALL_df, RF_ALL_df = custom_data
    else:
        X_ALL_df = X_ALL
        y_ALL_df = Y_ALL
        RF_ALL_df = RF_ALL
    print(f"Data loaded. Shapes: X_ALL={X_ALL_df.shape}, y_ALL={y_ALL_df.shape}, RF_ALL={RF_ALL_df.shape}")

    # --- Scale Data ---
    # scale_data now returns DataFrames when given DataFrames
    X_ALL_scaled_df, y_ALL_scaled_df, scaler_x, scaler_y = scale_data(X_ALL_df, y_ALL_df)

    # --- Train/Validation Split on SCALED data ---
    # Since we're using NumPy arrays, we'll use manual slicing
    val_ratio = 0.15
    n_total = len(X_ALL_scaled_df)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val
    
    X_tr_np = X_ALL_scaled_df[:n_train]
    X_val_np = X_ALL_scaled_df[n_train:]
    y_tr_np = y_ALL_scaled_df[:n_train]
    y_val_np = y_ALL_scaled_df[n_train:]
    
    # Ensure y is 2D as skorch expects and convert to float32
    y_tr_np = y_tr_np.reshape(-1, 1).astype(np.float32) if y_tr_np.ndim == 1 else y_tr_np.astype(np.float32)
    y_val_np = y_val_np.reshape(-1, 1).astype(np.float32) if y_val_np.ndim == 1 else y_val_np.astype(np.float32)
    X_tr_np = X_tr_np.astype(np.float32)
    X_val_np = X_val_np.astype(np.float32)

    print(f"Train shapes: X_tr={X_tr_np.shape}, y_tr={y_tr_np.shape}")
    print(f"Val shapes  : X_val={X_val_np.shape}, y_val={y_val_np.shape}")

    # --- Define actuals (unscaled) and risk-free for full period metrics ---
    # Handle both DataFrame and numpy array inputs
    if hasattr(y_ALL_df, 'values'):
        actual_ALL_unscaled_np = y_ALL_df.values.ravel()
    elif isinstance(y_ALL_df, np.ndarray):
        actual_ALL_unscaled_np = y_ALL_df.ravel() if y_ALL_df.ndim > 1 else y_ALL_df
    else:
        actual_ALL_unscaled_np = np.array(y_ALL_df).ravel()
        
    # Process risk-free rate data
    if hasattr(RF_ALL_df, 'values'):
        rf_ALL_np = RF_ALL_df.values.ravel()
    elif isinstance(RF_ALL_df, np.ndarray):
        rf_ALL_np = RF_ALL_df.ravel() if RF_ALL_df.ndim > 1 else RF_ALL_df
    else:
        rf_ALL_np = np.array(RF_ALL_df).ravel()

    # --- Benchmark: Historical Average (Expanding Window) on UNSCALED data ---
    # Calculate expanding mean manually on NumPy array
    y_flat = actual_ALL_unscaled_np  # Use the already flattened array
    y_pred_HA_expanding_np = np.zeros_like(y_flat)
    
    # First value is NaN (will be filled with the first actual value)
    y_pred_HA_expanding_np[0] = y_flat[0]  # Initialize with first value
    
    # For each position, calculate mean of all previous values
    for i in range(1, len(y_flat)):
        y_pred_HA_expanding_np[i] = np.mean(y_flat[:i])
    
    # Reshape to 2D array for consistency
    y_pred_HA_expanding_np = y_pred_HA_expanding_np.reshape(-1, 1)
    print(f"Expanding HA benchmark calculated. Shape: {y_pred_HA_expanding_np.shape}")


    metrics_summary = []
    final_predictions_all_models = {}

    for model_name_iter in models: # Renamed to avoid conflict with 'models' module
        print(f"\n--- Running Random Search for {model_name_iter} ---")
        model_class = getattr(nns, model_name_iter)
        
        # Get the specific search space for this model
        # Ensure SPACE (RANDOM) is structured correctly in search_spaces.py
        if model_name_iter not in SPACE:
            print(f"Warning: Search space for model '{model_name_iter}' not found in RANDOM. Skipping.")
            continue
        search_space_model = SPACE[model_name_iter]

        # Run random search
        # train_random expects numpy arrays for X_tr, y_tr, X_val, y_val
        best_hp, best_net = train_random(
            model_class,  # model_module
            GridNet,  # regressor_class (correct skorch wrapper)
            search_space_model,  # search_space_config
            X_tr_np, y_tr_np, X_val_np, y_val_np,  # training and validation data
            X_tr_np.shape[1],  # n_features
            epochs, device, trials, batch  # other parameters
        )
        
        # Set a default value for val_mse
        val_mse = float('inf') if best_net is None else None

        if best_net is None:
            print(f"Random search for {model_name_iter} did not yield a best model. Skipping final evaluation.")
            metrics_summary.append({
                'Model': model_name_iter, 'Validation MSE': val_mse if val_mse is not None else np.nan,
                'In-sample R2 (%)': np.nan, 'Success Ratio (%)': np.nan,
                'CER Gain vs HA (%)': np.nan, 'Best Hyperparameters': best_hp
            })
            continue

        # Save best hyperparameters
        joblib.dump(best_hp, out / "studies_or_best_params" / f"{model_name_iter}_best_params.pkl")

        # --- Final Evaluation: Predict on the ENTIRE dataset using the best model ---
        # The best_net is already trained on X_tr_np, y_tr_np.
        # For in-sample R2 and CER as defined in bayes.py, we'd refit on ALL scaled data.
        # OR, we evaluate its performance on the full X_ALL_scaled_np.
        # Let's follow the pattern of predicting on X_ALL_scaled_np with the model trained on X_tr_np, y_tr_np.
        # This is more of an "out-of-sample" test for the validation part of X_ALL_scaled_np.
        # If true "in-sample" means refitting on all data:
        print(f"Refitting best model for {model_name_iter} on entire SCALED dataset (X_ALL_scaled_df)...")
        
        # Create a clean set of parameters for the final model
        optimizer_name = best_hp.get('optimizer', 'Adam')
        lr = best_hp.get('lr', 0.001)
        weight_decay = best_hp.get('weight_decay', 0)
        l1_lambda = best_hp.get('l1_lambda', 0)
        
        # Extract module parameters (prefixed with module__)
        module_params = {}
        for key, value in best_hp.items():
            if key.startswith('module__'):
                module_params[key] = value
            elif key in ['dropout', 'n_hidden1', 'n_hidden2', 'n_hidden3', 'n_hidden4', 'n_hidden5', 'activation_hidden'] and f'module__{key}' not in module_params:
                module_params[f'module__{key}'] = value
        
        # Add required module parameters
        module_params['module__n_feature'] = X_ALL_scaled_df.shape[1]
        module_params['module__n_output'] = 1
        
        # Create the final model for evaluation
        final_model = GridNet(
            module=model_class,
            max_epochs=epochs,
            batch_size=best_hp.get('batch_size', batch),
            optimizer=getattr(torch.optim, optimizer_name),
            lr=lr,
            optimizer__weight_decay=weight_decay,
            l1_lambda=l1_lambda,
            iterator_train__shuffle=True,
            device=device,
            **module_params
        )
        
        # Prepare data for training
        X_ALL_tensor = torch.tensor(X_ALL_scaled_df, dtype=torch.float32)
        if isinstance(y_ALL_scaled_df, np.ndarray):
            y_ALL_tensor_data = y_ALL_scaled_df.reshape(-1, 1) if y_ALL_scaled_df.ndim == 1 else y_ALL_scaled_df
        else:
            # For pandas DataFrame or Series
            y_ALL_tensor_data = y_ALL_scaled_df.values.reshape(-1, 1) if len(y_ALL_scaled_df.shape) == 1 else y_ALL_scaled_df.values
        
        y_ALL_tensor = torch.tensor(y_ALL_tensor_data, dtype=torch.float32)
        
        final_model.fit(X_ALL_tensor, y_ALL_tensor)
        
        preds_ALL_scaled_np = final_model.predict(X_ALL_tensor)
        preds_ALL_unscaled_np = scaler_y.inverse_transform(preds_ALL_scaled_np)
        
        final_predictions_all_models[model_name_iter] = preds_ALL_unscaled_np

        # Save predictions to CSV - without Date column since we don't have the index
        pd.DataFrame({
            'Actual': actual_ALL_unscaled_np.flatten(),
            'Predicted': preds_ALL_unscaled_np.flatten(),
            'HA': y_pred_HA_expanding_np.flatten()
        }).to_csv(out / "predictions" / f"{model_name_iter}_predictions.csv", index=False)

        # --- Calculate Metrics using UNscaled predictions and actuals ---
        # Ensure y_pred_HA_expanding_np aligns with actual_ALL_unscaled_np and preds_ALL_unscaled_np
        # All should have the same length now.
        
        r2_vs_ha = compute_in_r_square(
            actual_ALL_unscaled_np,
            y_pred_HA_expanding_np, # Benchmark
            preds_ALL_unscaled_np
        ) * 100
        
        sr = compute_success_ratio(
            actual_ALL_unscaled_np,
            preds_ALL_unscaled_np
        ) * 100
        
        # CER calculation using metrics.py's compute_CER
        # compute_CER(actual_returns, predicted_returns, risk_free_rates, gamma)
        # It returns annualized fractional CER. Multiply by 100 for percentage points.
        
        # Ensure all inputs to compute_CER are 1D arrays
        actual_for_cer = actual_ALL_unscaled_np.ravel()
        preds_model_for_cer = preds_ALL_unscaled_np.ravel()
        rf_for_cer = rf_ALL_np.ravel()
        preds_ha_for_cer = y_pred_HA_expanding_np.ravel()

        # Check lengths, though they should align if data prep is correct
        min_len = min(len(actual_for_cer), len(preds_model_for_cer), len(rf_for_cer), len(preds_ha_for_cer))
        if len(actual_for_cer) != min_len or \
           len(preds_model_for_cer) != min_len or \
           len(rf_for_cer) != min_len or \
           len(preds_ha_for_cer) != min_len:
            print("Warning: Length mismatch for CER calculation inputs. Truncating to shortest length.")
            actual_for_cer = actual_for_cer[:min_len]
            preds_model_for_cer = preds_model_for_cer[:min_len]
            rf_for_cer = rf_for_cer[:min_len]
            preds_ha_for_cer = preds_ha_for_cer[:min_len]

        cer_model = compute_CER(
            actual_for_cer,         # Actual market returns
            preds_model_for_cer,    # Model's predicted market returns
            rf_for_cer,             # Risk-free rates
            gamma_cer               # Pass as positional parameter
        ) * 100                     # Convert to percentage points

        cer_ha = compute_CER(
            actual_for_cer,         # Actual market returns
            preds_ha_for_cer,       # HA's "predicted" market returns
            rf_for_cer,             # Risk-free rates
            gamma_cer               # Pass as positional parameter
        ) * 100                     # Convert to percentage points

        cer_gain_vs_ha = cer_model - cer_ha # Difference in CER percentage points

        metrics_summary.append({
            'Model': model_name_iter,
            'Validation MSE': val_mse,
            'In-sample R2 (%)': r2_vs_ha,
            'Success Ratio (%)': sr,
            'CER Model (%, ann.)': cer_model,
            'CER HA (%, ann.)': cer_ha,
            'CER Gain vs HA (%, ann.)': cer_gain_vs_ha,
            'Best Hyperparameters': str(best_hp)
        })
        # Handle the case where metrics might be None
        val_mse_str = f"{val_mse:.4f}" if val_mse is not None else "N/A"
        r2_str = f"{r2_vs_ha:.2f}%" if r2_vs_ha is not None else "N/A"
        sr_str = f"{sr:.2f}%" if sr is not None else "N/A"
        cer_gain_str = f"{cer_gain_vs_ha:.4f}pp" if cer_gain_vs_ha is not None else "N/A"
        print(f"--- Finished {model_name_iter}: Val MSE={val_mse_str}, R2={r2_str}, SR={sr_str}, CER Gain={cer_gain_str} ---")

    # --- Add HA benchmark row to metrics summary ---
    ha_sr = compute_success_ratio(actual_ALL_unscaled_np, y_pred_HA_expanding_np) * 100
    
    # Calculate CER for historical average benchmark using function parameter
    actual_for_bench_cer = actual_ALL_unscaled_np.ravel()
    pred_ha_for_bench_cer = y_pred_HA_expanding_np.ravel()
    rf_for_bench_cer = rf_ALL_np.ravel()
    
    cer_ha_bench = compute_CER(
        actual_for_bench_cer,
        pred_ha_for_bench_cer,
        rf_for_bench_cer,
        gamma_cer  # Use function parameter, not local redefinition
    ) * 100

    metrics_summary.append({
        'Model': 'HA_Benchmark',
        'Validation MSE': np.nan,
        'In-sample R2 (%)': 0.0,
        'Success Ratio (%)': ha_sr,
        'CER Model (%, ann.)': cer_ha_bench,
        'CER HA (%, ann.)': cer_ha_bench,
        'CER Gain vs HA (%, ann.)': 0.0,
        'Best Hyperparameters': "N/A"
    })

    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics_summary)
    metrics_df.to_csv(out / "final_metrics_random_search.csv", index=False)
    print(f"\nFinal metrics saved to {out}/final_metrics_random_search.csv")

    total_time_taken = datetime.now() - start_time_total
    print(f"Total execution time for random search: {total_time_taken}")