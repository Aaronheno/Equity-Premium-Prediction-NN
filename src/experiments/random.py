from pathlib import Path;from datetime import datetime
import pandas as pd, torch, numpy as np, joblib
from src.utils.io import train_val_split, RF_ALL, X_ALL, Y_ALL
from src.utils.processing import scale_data
from src.utils.evaluation import compute_in_r_square, compute_success_ratio, compute_CER
from src.utils.metrics import compute_CER as metrics_compute_CER
from src.utils.training_grid import train_grid, GridNet
from src.utils.training_random import train_random
from src.configs.search_spaces import GRID, RANDOM as SPACE
from src.models import nns
from skorch.callbacks import EarlyStopping
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
    gamma_cer=3.0 # Default gamma for CER, changed to 3.0
):
    torch.set_num_threads(threads) # For PyTorch operations if not using CUDA much
    start_time_total = datetime.now()
    ts = start_time_total.strftime("%Y-%m-%d_%H-%M")

    # --- Output Path ---
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
    X_ALL_df = X_ALL
    y_ALL_df = Y_ALL
    RF_ALL_df = RF_ALL
    print(f"Data loaded. Shapes: X_ALL={X_ALL_df.shape}, y_ALL={y_ALL_df.shape}, RF_ALL={RF_ALL_df.shape}")

    # --- Scale Data ---
    # scale_data returns X_scaled_np, y_scaled_np, scaler_x, scaler_y
    X_ALL_scaled_np, y_ALL_scaled_np, scaler_x, scaler_y = scale_data(X_ALL_df, y_ALL_df)
    
    # For train_val_split, it's better to pass DataFrames if split_by_index=True
    # to preserve index information easily.
    X_ALL_scaled_df = pd.DataFrame(X_ALL_scaled_np, index=X_ALL_df.index, columns=X_ALL_df.columns)
    y_ALL_scaled_df = pd.DataFrame(y_ALL_scaled_np, index=y_ALL_df.index, columns=y_ALL_df.columns)

    # --- Train/Validation Split on SCALED data ---
    X_tr_df, X_val_df, y_tr_df, y_val_df = train_val_split(
        X_ALL_scaled_df, y_ALL_scaled_df, val_ratio=0.15, split_by_index=True
    )
    # Convert to numpy for skorch/PyTorch training
    X_tr_np = X_tr_df.values
    y_tr_np = y_tr_df.values.reshape(-1, 1) # Ensure 2D for skorch
    X_val_np = X_val_df.values
    y_val_np = y_val_df.values.reshape(-1, 1) # Ensure 2D for skorch

    print(f"Train shapes: X_tr={X_tr_np.shape}, y_tr={y_tr_np.shape}")
    print(f"Val shapes  : X_val={X_val_np.shape}, y_val={y_val_np.shape}")

    # --- Define actuals (unscaled) and risk-free for full period metrics ---
    actual_ALL_unscaled_np = y_ALL_df.values # Full period actuals, unscaled
    rf_ALL_np = RF_ALL_df.values              # Full period risk-free, unscaled

    # --- Benchmark: Historical Average (Expanding Window) on UNSCALED data ---
    # This benchmark is for the R-squared calculation against the full model predictions.
    y_pred_HA_expanding_series = y_ALL_df.iloc[:, 0].expanding(min_periods=1).mean().shift(1)
    first_actual_val = y_ALL_df.iloc[0, 0] # Use first actual value to fill initial NaN
    y_pred_HA_expanding_series = y_pred_HA_expanding_series.fillna(first_actual_val)
    y_pred_HA_expanding_np = y_pred_HA_expanding_series.values.reshape(-1, 1)
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
        best_net, best_hp, val_mse = train_random(
            model_class, X_tr_np, y_tr_np, X_val_np, y_val_np,
            search_space_model, epochs, trials, device
        )

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
        print(f"Refitting best model for {model_name_iter} on entire SCALED dataset (X_ALL_scaled_np)...")
        
        # Extract params for refit, ensure module__ prefix for skorch
        refit_module_params = {f"module__{k}": v for k, v in best_hp.items()
                               if k not in ["optimizer", "lr", "weight_decay", "l1_lambda", "batch_size"]}

        final_model_for_eval = GridNet( # Or the class used in train_random
            module=model_class,
            module__n_feature=X_ALL_scaled_np.shape[1],
            module__n_output=1,
            **refit_module_params,
            max_epochs=epochs, # Use same epochs as in search, or a different number for final fit
            batch_size=best_hp.get("batch_size", batch), # Use found or default
            optimizer=getattr(torch.optim, best_hp.get("optimizer", "Adam")),
            lr=best_hp.get("lr", 1e-3),
            optimizer__weight_decay=best_hp.get("weight_decay", 0.0),
            l1_lambda=best_hp.get("l1_lambda", 0.0),
            iterator_train__shuffle=True,
            callbacks=None, # Explicitly no callbacks for final in-sample refit
            # For final fit on ALL data, EarlyStopping on valid_loss is not applicable.
            # We train for the full 'epochs' specified.
            device=device,
            verbose=0
        )
        y_ALL_scaled_np_fit = y_ALL_scaled_np.reshape(-1,1) if y_ALL_scaled_np.ndim == 1 else y_ALL_scaled_np
        final_model_for_eval.fit(X_ALL_scaled_np, y_ALL_scaled_np_fit)
        
        preds_ALL_scaled_np = final_model_for_eval.predict(X_ALL_scaled_np)
        preds_ALL_unscaled_np = scaler_y.inverse_transform(preds_ALL_scaled_np)
        
        final_predictions_all_models[model_name_iter] = preds_ALL_unscaled_np

        # Save predictions
        pd.DataFrame({
            'Date': X_ALL_df.index, # Original dates
            'Actual_Unscaled': actual_ALL_unscaled_np.flatten(),
            'Predicted_Unscaled': preds_ALL_unscaled_np.flatten(),
            'HA_Benchmark_Unscaled': y_pred_HA_expanding_np.flatten()
        }).to_csv(out / "predictions" / f"{model_name_iter}_fulldata_predictions.csv", index=False)

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

        cer_model = metrics_compute_CER(
            actual_for_cer,         # Actual market returns
            preds_model_for_cer,    # Model's predicted market returns
            rf_for_cer,             # Risk-free rates
            gamma=gamma_cer
        ) * 100                     # Convert to percentage points

        cer_ha = metrics_compute_CER(
            actual_for_cer,         # Actual market returns
            preds_ha_for_cer,       # HA's "predicted" market returns
            rf_for_cer,             # Risk-free rates
            gamma=gamma_cer
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
        print(f"--- Finished {model_name_iter}: Val MSE={val_mse:.4f}, R2={r2_vs_ha:.2f}%, SR={sr:.2f}%, CER Gain={cer_gain_vs_ha:.4f}pp ---")

    # --- Add HA benchmark row to metrics summary ---
    ha_sr = compute_success_ratio(actual_ALL_unscaled_np, y_pred_HA_expanding_np) * 100
    
    # Recalculate cer_ha_bench consistently for the HA row
    # Ensure inputs are consistently shaped and lengthed as above
    actual_for_bench_cer = actual_ALL_unscaled_np.ravel()[:min_len] # Use min_len from model loop if applicable
    preds_ha_for_bench_cer = y_pred_HA_expanding_np.ravel()[:min_len]
    rf_for_bench_cer = rf_ALL_np.ravel()[:min_len]

    if len(actual_ALL_unscaled_np.ravel()) != min_len : # if min_len was used due to truncation
        print("Warning: HA benchmark CER calculation might use truncated series due to prior length mismatches.")


    cer_ha_bench = metrics_compute_CER(
        actual_for_bench_cer,
        preds_ha_for_bench_cer,
        rf_for_bench_cer,
        gamma=gamma_cer
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