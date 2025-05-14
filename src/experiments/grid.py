# src/experiments/grid.py
from pathlib import Path
from datetime import datetime
import pandas as pd, torch, numpy as np, joblib
from src.utils.io import train_val_split, RF_ALL, X_ALL, Y_ALL
from src.utils.processing import scale_data
from src.utils.evaluation import compute_in_r_square, compute_success_ratio
from src.utils.metrics import compute_CER
from src.utils.training_grid import train_grid, GridNet
from src.configs.search_spaces import GRID as SPACE
from src.models import nns
import inspect

def run(
    models, 
    trials, # trials might not be used by grid search, but keep for consistency if cli passes it
    epochs, 
    threads, 
    batch, # <<< ADD batch
    device, # <<< ADD device
    gamma_cer=3.0, # Add gamma_cer with default 3.0
    # method="grid" # This can be removed if it's always "grid" for this file
    ):
    torch.set_num_threads(threads)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    # --- Output Path ---
    out_base = Path("./runs/0_Grid_Search_In_Sample") # <<< CHANGED from 2_ to 0_
    
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

    # --- Load Data (already done by global imports X_ALL, Y_ALL, RF_ALL) ---
    # X_ALL, Y_ALL, RF_ALL are the original unscaled DataFrames.
    print(f"Original data loaded. Shapes: X_ALL={X_ALL.shape}, Y_ALL={Y_ALL.shape}, RF_ALL={RF_ALL.shape}")


    # --- Scale Full Data ---
    # scale_data returns X_scaled_np, y_scaled_np, scaler_x, scaler_y
    X_ALL_scaled_np, y_ALL_scaled_np, scaler_x, scaler_y = scale_data(X_ALL, Y_ALL)
    
    # Convert scaled numpy arrays back to DataFrames with original index for splitting
    X_ALL_scaled_df = pd.DataFrame(X_ALL_scaled_np, index=X_ALL.index, columns=X_ALL.columns)
    y_ALL_scaled_df = pd.DataFrame(y_ALL_scaled_np, index=Y_ALL.index, columns=Y_ALL.columns)
    print(f"Full data scaled. Shapes: X_ALL_scaled_df={X_ALL_scaled_df.shape}, y_ALL_scaled_df={y_ALL_scaled_df.shape}")

    # --- Train/Validation Split on SCALED data ---
    # train_val_split now returns 4 items: X_tr_df, X_val_df, y_tr_df, y_val_df
    X_tr_df, X_val_df, y_tr_df, y_val_df = train_val_split(
        X_ALL_scaled_df, y_ALL_scaled_df, val_ratio=0.15, split_by_index=True
    )
    
    X_tr_np = X_tr_df.values
    y_tr_np = y_tr_df.values.reshape(-1,1) # skorch expects 2D y
    X_val_np = X_val_df.values
    y_val_np = y_val_df.values.reshape(-1,1) # skorch expects 2D y

    # For final metrics, we need unscaled actuals and corresponding risk-free rates
    actual_full_unscaled_np = Y_ALL.values.ravel() # Use the original unscaled Y_ALL
    
    # Align RF_ALL with the full dataset's index (X_ALL.index)
    rf_full_aligned_np = RF_ALL.reindex(X_ALL.index).fillna(method='ffill').fillna(method='bfill').values.ravel()

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
    # Use original Y_ALL for HA benchmark calculation
    y_pred_HA_expanding_series = Y_ALL.iloc[:, 0].expanding(min_periods=1).mean().shift(1)
    first_actual_val = Y_ALL.iloc[0, 0] # Use first value from original Y_ALL
    y_pred_HA_expanding_series = y_pred_HA_expanding_series.fillna(first_actual_val)
    # Ensure HA predictions align with the (potentially truncated) actual_full_unscaled_np
    y_pred_HA_expanding_np = y_pred_HA_expanding_series.values[:len(actual_full_unscaled_np)].ravel()


    metrics = []
    final_predictions_all_models = {} # To store predictions for each model

    for m in models:
        print(f"\n--- Running Grid Search for {m} ---")
        model_class = getattr(nns, m)
        
        if m not in SPACE:
            print(f"Warning: Search space for model '{m}' not found in GRID. Skipping.")
            continue
        search_space_model = SPACE[m]

        # train_grid returns best_net, best_hyperparams, best_validation_mse
        best_net, hp, val_mse = train_grid(
            model_class, X_tr_np, y_tr_np, X_val_np, y_val_np,
            search_space_model, epochs, device=device # <<< PASS device
        )

        if best_net is None:
            print(f"Grid search for {m} did not yield a best model. Skipping final evaluation.")
            metrics.append({
                'Model': m, 'Validation MSE': val_mse if val_mse is not None else np.nan,
                'In-sample R2 (%)': np.nan, 'Success Ratio (%)': np.nan,
                'CER Gain vs HA (pp, ann.)': np.nan, 'Best Hyperparameters': hp
            })
            continue
        
        joblib.dump(hp, out / "studies_or_best_params" / f"{m}_best_params.pkl")

        # --- Final Evaluation: Predict on the ENTIRE dataset ---
        # Refit the best model on the entire SCALED dataset.
        # X_ALL_scaled_np and y_ALL_scaled_np are already prepared and potentially truncated.
        X_ALL_scaled_np_for_refit = X_ALL_scaled_np
        y_ALL_scaled_np_for_refit = y_ALL_scaled_np.reshape(-1,1) # Ensure y is 2D for skorch

        print(f"Refitting best model for {m} on entire SCALED dataset (Shape X: {X_ALL_scaled_np_for_refit.shape}, Shape y: {y_ALL_scaled_np_for_refit.shape})")
        # final_model_for_eval instantiation seems fine, it uses hp for module params
        final_model_instance_params = {
            k: v for k,v in hp.items() 
            if k not in ['optimizer', 'lr', 'weight_decay', 'l1_lambda', 'batch_size']
        }
        # Ensure all necessary params for model_class are present in final_model_instance_params
        # or are defaults in the model_class itself.
        # Example: model_class might need 'activation_hidden' if not in hp.

        # Use the existing GridNet for the final fit
        final_skorch_net = GridNet( # <<< CHANGE FinalGridNetSkorch to GridNet
            module=model_class,
            module__n_feature=X_ALL_scaled_np_for_refit.shape[1],
            module__n_output=1,
            # Pass only module-specific HPs from hp, prefixed with module__
            **{f"module__{k}": v for k, v in final_model_instance_params.items()},
            max_epochs=epochs,
            batch_size=hp.get('batch_size', 256),
            optimizer=getattr(torch.optim, hp.get('optimizer', 'Adam')),
            lr=hp.get('lr', 0.01),
            optimizer__weight_decay=hp.get('weight_decay', 0),
            l1_lambda=hp.get('l1_lambda', 0),
            iterator_train__shuffle=True,
            callbacks=None, # No early stopping for final fit on all data
            device=device, # <<< PASS device
            verbose=0
        )
        final_skorch_net.fit(X_ALL_scaled_np_for_refit, y_ALL_scaled_np_for_refit)
        
        preds_ALL_scaled_np = final_skorch_net.predict(X_ALL_scaled_np_for_refit)
        preds_ALL_unscaled_np = scaler_y.inverse_transform(preds_ALL_scaled_np).ravel()
        
        final_predictions_all_models[m] = preds_ALL_unscaled_np

        # Save predictions
        # Ensure index for Date aligns with the length of actual_full_unscaled_np
        pd.DataFrame({
            'Date': X_ALL.index[:len(actual_full_unscaled_np)],
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
            gamma=gamma_cer
        ) * 100

        cer_ha = compute_CER(
            actual_full_unscaled_np,
            y_pred_HA_expanding_np, # HA's predictions
            rf_full_aligned_np,
            gamma=gamma_cer
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
            'Best Hyperparameters': str(hp)
        })
        print(f"--- Finished {m}: Val MSE={val_mse:.4f}, R2={r2:.2f}%, SR={success:.2f}%, CER Gain={cer_gain_vs_ha:.4f}pp ---")

    # --- Add HA benchmark row ---
    ha_sr_bench = compute_success_ratio(actual_full_unscaled_np, y_pred_HA_expanding_np) * 100
    cer_ha_bench_val = compute_CER(
        actual_full_unscaled_np,
        y_pred_HA_expanding_np,
        rf_full_aligned_np,
        gamma=gamma_cer
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
