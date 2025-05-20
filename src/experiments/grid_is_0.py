# src/experiments/grid.py
from pathlib import Path
from datetime import datetime
import pandas as pd, torch, numpy as np, joblib
from sklearn.metrics import mean_squared_error
from src.utils.io import train_val_split, RF_ALL, X_ALL, Y_ALL
from src.utils.metrics_unified import scale_data, compute_in_r_square, compute_success_ratio, compute_CER
from src.utils.training_grid import train_grid, GridNet
from src.configs.search_spaces import GRID as SPACE
from src.models import nns
import inspect

def run(
    models, 
    trials, # trials might not be used by grid search, but keep for consistency if cli passes it
    epochs, 
    threads, 
    batch, # batch size for training
    device, # device for training (cpu/cuda)
    gamma_cer=3.0, # gamma parameter for CER calculation
    custom_data=None, # Optional tuple of (X_ALL, Y_ALL, RF_ALL) to use instead of default data
    out_dir_override=None # Optional override for output directory
    ):
    torch.set_num_threads(threads)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    # --- Output Path ---
    if out_dir_override:
        out_base = Path(out_dir_override)
    else:
        out_base = Path("./runs/0_Grid_Search_In_Sample")
    
    # Dynamic run_name based on models
    # A simple heuristic: if more than, say, 3-4 models are run, or if it matches a known 'all' count
    # For a more precise "all models" check, you might need to pass the default model list length
    # or the cli.DEFAULT_MODELS_GRID itself if easily accessible.
    
    # Get a list of all discoverable model class names in nns.py
    # This is a more robust way to check for "all models" if you don't use __all__
    all_model_names_in_nns = [name for name, obj in inspect.getmembers(nns) if inspect.isclass(obj) and issubclass(obj, torch.nn.Module) and obj is not torch.nn.Module and not name.startswith('_')]

    if len(models) == 1:
        model_name_part = models[0]
    elif sorted(models) == sorted(all_model_names_in_nns) or len(models) > 5 : # If models list matches all defined in nns.py or many models
        model_name_part = "all_models"
    else:
        model_name_part = "_".join(sorted(models))

    run_name = f"{ts}_{model_name_part}_grid_insample"
    out = out_base / run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions").mkdir(exist_ok=True)
    (out / "studies_or_best_params").mkdir(exist_ok=True) # For best_params
    print(f"Output directory: {out}")

    # --- Load Data ---
    # Use custom data if provided, otherwise use global imports
    if custom_data is not None and len(custom_data) == 3:
        X_data, Y_data, rf_data = custom_data
        # Convert to NumPy arrays if they're DataFrames
        X_data_np = X_data.values if hasattr(X_data, 'values') else X_data
        Y_data_np = Y_data if isinstance(Y_data, np.ndarray) else Y_data.values
        rf_data_np = rf_data.values if hasattr(rf_data, 'values') else rf_data
        print(f"Custom data loaded. Shapes: X_data={X_data.shape}, Y_data={np.shape(Y_data)}, rf_data={np.shape(rf_data)}")
    else:
        # Use global imported data
        X_data = X_ALL
        Y_data = Y_ALL
        rf_data = RF_ALL
        X_data_np = X_ALL.values if hasattr(X_ALL, 'values') else X_ALL
        Y_data_np = Y_ALL if isinstance(Y_ALL, np.ndarray) else Y_ALL.values
        rf_data_np = RF_ALL.values if hasattr(RF_ALL, 'values') else RF_ALL
        print(f"Original data loaded. Shapes: X_data={X_data.shape}, Y_data={np.shape(Y_data)}, rf_data={np.shape(rf_data)}")


    # --- Scale Full Data ---
    # Use scale_data with numpy arrays
    X_ALL_scaled_np, y_ALL_scaled_np, scaler_x, scaler_y = scale_data(X_data, Y_data)
    
    # Ensure all data is float32 for PyTorch compatibility
    X_ALL_scaled_np = X_ALL_scaled_np.astype(np.float32)
    y_ALL_scaled_np = y_ALL_scaled_np.astype(np.float32)
    print(f"Full data scaled. Shapes: X_ALL_scaled_np={X_ALL_scaled_np.shape}, y_ALL_scaled_np={y_ALL_scaled_np.shape}")

    # --- Train/Validation Split on SCALED data ---
    # Since we're using NumPy arrays, we'll use manual slicing
    val_ratio = 0.15
    n_total = len(X_ALL_scaled_np)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val
    
    X_tr_np = X_ALL_scaled_np[:n_train]
    X_val_np = X_ALL_scaled_np[n_train:]
    y_tr_np = y_ALL_scaled_np[:n_train]
    y_val_np = y_ALL_scaled_np[n_train:]
    
    print(f"Data split (val_ratio={val_ratio}): Train={X_tr_np.shape}, Validation={X_val_np.shape}")
    
    # Ensure y is 2D as skorch expects
    y_tr_np = y_tr_np.reshape(-1,1) if y_tr_np.ndim == 1 else y_tr_np
    y_val_np = y_val_np.reshape(-1,1) if y_val_np.ndim == 1 else y_val_np

    # For final metrics, we need unscaled actuals and corresponding risk-free rates
    actual_full_unscaled_np = Y_data_np.ravel() if hasattr(Y_data_np, 'ravel') else Y_data_np.flatten()
    
    # RF data should already be aligned with X and Y data
    rf_full_aligned_np = rf_data_np.ravel() if hasattr(rf_data_np, 'ravel') else rf_data_np.flatten()

    # Ensure consistent lengths after potential NaN removal during alignment or other processing
    min_len_data = min(len(actual_full_unscaled_np), len(rf_full_aligned_np), len(X_ALL_scaled_np))
    if len(actual_full_unscaled_np) > min_len_data:
        print(f"Warning: Truncating actual_full_unscaled_np from {len(actual_full_unscaled_np)} to {min_len_data} for consistency.")
        actual_full_unscaled_np = actual_full_unscaled_np[:min_len_data]
    if len(rf_full_aligned_np) > min_len_data:
        print(f"Warning: Truncating rf_full_aligned_np from {len(rf_full_aligned_np)} to {min_len_data} for consistency.")
        rf_full_aligned_np = rf_full_aligned_np[:min_len_data]
    if len(X_ALL_scaled_np) > min_len_data:
        print(f"Warning: Truncating X_ALL_scaled_np from {len(X_ALL_scaled_np)} to {min_len_data} for consistency.")
        X_ALL_scaled_np = X_ALL_scaled_np[:min_len_data]
        # also y_ALL_scaled_np if used directly later for refit target
        if len(y_ALL_scaled_np) > min_len_data:
             y_ALL_scaled_np = y_ALL_scaled_np[:min_len_data]


    # --- Benchmark: Historical Average (Expanding Window) on UNSCALED data ---
    # Calculate expanding mean manually on NumPy array
    y_flat = Y_ALL.reshape(-1)  # Flatten the array
    y_pred_HA_expanding_np = np.zeros_like(y_flat)
    
    # First value is NaN (will be filled with the first actual value)
    y_pred_HA_expanding_np[0] = y_flat[0]  # Initialize with first value
    
    # For each position, calculate mean of all previous values
    for i in range(1, len(y_flat)):
        y_pred_HA_expanding_np[i] = np.mean(y_flat[:i])
    
    # Ensure HA predictions align with the actual data
    y_pred_HA_expanding_np = y_pred_HA_expanding_np[:len(actual_full_unscaled_np)]


    metrics = []
    final_predictions_all_models = {} # To store predictions for each model

    for m in models:
        print(f"\n--- Running Grid Search for {m} ---")
        model_class = getattr(nns, m)
        
        if m not in SPACE:
            print(f"Warning: Search space for model '{m}' not found in GRID. Skipping.")
            continue
        search_space_model = SPACE[m]

        # train_grid expects (model_module, regressor_class, search_space_config, X_train, y_train, X_val, y_val, n_features, epochs, device)
        best_hp, best_net = train_grid(
            model_class,          # The PyTorch model class
            GridNet,             # The regressor class
            search_space_model,  # Grid parameters
            X_tr_np, y_tr_np, X_val_np, y_val_np,  # Training and validation data
            X_tr_np.shape[1],    # Number of input features
            epochs,              # Maximum epochs for training
            device,              # Device to use
            batch_size_default=batch  # Default batch size
        )
        
        # Calculate validation MSE using the best model if available
        if best_net is not None:
            val_mse = mean_squared_error(y_val_np, best_net.predict(X_val_np))
        else:
            val_mse = float('inf')  # Set to infinity if no model was trained

        if best_net is None:
            print(f"Grid search for {m} did not yield a best model. Skipping final evaluation.")
            metrics.append({
                'Model': m, 'Validation MSE': val_mse,
                'In-sample R2 (%)': np.nan, 'Success Ratio (%)': np.nan,
                'CER Gain vs HA (%)': np.nan, 'Best Hyperparameters': best_hp
            })
            continue
        
        # Save best hyperparameters
        joblib.dump(best_hp, out / "studies_or_best_params" / f"{m}_best_params.pkl")

        # --- Final Evaluation: Predict on the ENTIRE dataset ---
        # Refit the best model on the entire SCALED dataset.
        # X_ALL_scaled_np and y_ALL_scaled_np are already prepared and potentially truncated.
        X_ALL_scaled_np_for_refit = X_ALL_scaled_np
        y_ALL_scaled_np_for_refit = y_ALL_scaled_np.reshape(-1,1) # Ensure y is 2D for skorch

        print(f"Refitting best model for {m} on entire SCALED dataset (Shape X: {X_ALL_scaled_np_for_refit.shape}, Shape y: {y_ALL_scaled_np_for_refit.shape})")
        # Extract model-specific params from best_hp
        final_model_instance_params = {}
        for key, value in best_hp.items():
            if key.startswith('module__'):
                # Get the parameter name without the module__ prefix
                param_name = key[len('module__'):]
                final_model_instance_params[param_name] = value
                
        # Ensure all required model parameters are present
        # For Net1 these are: n_hidden1, activation_hidden, dropout

        # Use the existing GridNet for the final fit
        # Handle the case where optimizer could be a class or a string
        optimizer_param = best_hp.get('optimizer', torch.optim.Adam)
        if isinstance(optimizer_param, type):
            optimizer_class = optimizer_param  # It's already a class
        else:
            # It's a string, get the class from torch.optim
            optimizer_class = getattr(torch.optim, optimizer_param)
            
        # Create the module parameters dictionary, ensuring we include required parameters
        module_params = {
            'module__n_feature': X_ALL_scaled_np_for_refit.shape[1],
            'module__n_output': 1
        }
        
        # Add any module-specific parameters from best_hp that have 'module__' prefix
        for key, value in best_hp.items():
            if key.startswith('module__'):
                module_params[key] = value
        
        final_skorch_net = GridNet(
            module=model_class,
            max_epochs=epochs,
            batch_size=best_hp.get('batch_size', 256),
            optimizer=optimizer_class,
            lr=best_hp.get('lr', 0.01),
            optimizer__weight_decay=best_hp.get('optimizer__weight_decay', 0),
            l1_lambda=best_hp.get('l1_lambda', 0),
            iterator_train__shuffle=True,
            callbacks=None, # No early stopping for final fit on all data
            device=device,
            verbose=0,
            **module_params  # Pass all module parameters this way
        )
        final_skorch_net.fit(X_ALL_scaled_np_for_refit, y_ALL_scaled_np_for_refit)
        
        preds_ALL_scaled_np = final_skorch_net.predict(X_ALL_scaled_np_for_refit)
        preds_ALL_unscaled_np = scaler_y.inverse_transform(preds_ALL_scaled_np).ravel()
        
        final_predictions_all_models[m] = preds_ALL_unscaled_np

        # Save predictions
        # Use original X_ALL DataFrame's index if available, or generate a sequence if not
        try:
            # If X_ALL is still a DataFrame with an index
            date_index = X_ALL.index[:len(actual_full_unscaled_np)] if hasattr(X_ALL, 'index') else range(len(actual_full_unscaled_np))
        except:
            # Fallback to simple sequence
            date_index = range(len(actual_full_unscaled_np))
            
        pd.DataFrame({
            'Date': date_index,
            'Actual_Unscaled': actual_full_unscaled_np,
            'Predicted_Unscaled': preds_ALL_unscaled_np,
            'HA_Benchmark_Unscaled': y_pred_HA_expanding_np
        }).to_csv(out / "predictions" / f"{m}_fulldata_predictions.csv", index=False)

        # --- Calculate Metrics ---
        r2 = compute_in_r_square(
            actual_full_unscaled_np, 
            y_pred_HA_expanding_np, 
            preds_ALL_unscaled_np
        ) * 100
        
        success = compute_success_ratio(
            actual_full_unscaled_np, 
            preds_ALL_unscaled_np
        ) * 100

        # Ensure all inputs to compute_CER are 1D and same length
        # actual_full_unscaled_np, preds_ALL_unscaled_np, rf_full_aligned_np
        # y_pred_HA_expanding_np

        cer_model = compute_CER(
            actual_full_unscaled_np,
            preds_ALL_unscaled_np,
            rf_full_aligned_np,
            gamma_cer  # Pass as positional parameter
        ) * 100

        cer_ha = compute_CER(
            actual_full_unscaled_np,
            y_pred_HA_expanding_np, # HA's predictions
            rf_full_aligned_np,
            gamma_cer  # Pass as positional parameter
        ) * 100
        
        cer_gain_vs_ha = cer_model - cer_ha

        metrics.append({
            'Model': m,
            'Validation MSE': val_mse,
            'In-sample R2 (%)': r2,
            'Success Ratio (%)': success,
            'CER Model (%, ann.)': cer_model,
            'CER HA (%, ann.)': cer_ha,
            'CER Gain vs HA (%, ann.)': cer_gain_vs_ha,
            'Best Hyperparameters': str(best_hp)
        })
        print(f"--- Finished {m}: Val MSE={val_mse:.4f}, R2={r2:.2f}%, SR={success:.2f}%, CER Gain={cer_gain_vs_ha:.4f}pp ---")

    # --- Add HA benchmark row ---
    ha_sr_bench = compute_success_ratio(actual_full_unscaled_np, y_pred_HA_expanding_np) * 100
    
    # Pass gamma via positional parameter order to avoid duplicate keyword
    cer_ha_bench_val = compute_CER(
        actual_full_unscaled_np,
        y_pred_HA_expanding_np,
        rf_full_aligned_np,
        gamma_cer
    ) * 100

    metrics.append({
        'Model': 'HA_Benchmark',
        'Validation MSE': np.nan,
        'In-sample R2 (%)': 0.0,
        'Success Ratio (%)': ha_sr_bench,
        'CER Model (%, ann.)': cer_ha_bench_val,
        'CER HA (%, ann.)': cer_ha_bench_val,
        'CER Gain vs HA (%, ann.)': 0.0,
        'Best Hyperparameters': "N/A"
    })

    pd.DataFrame(metrics).to_csv(out / "final_metrics_grid_search.csv", index=False) # Changed filename
    print(f"\nResults saved to {out}/final_metrics_grid_search.csv")
