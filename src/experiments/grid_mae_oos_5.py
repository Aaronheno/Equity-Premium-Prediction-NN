"""
Out-of-sample evaluation using grid search with MAE as the scoring function.

This module implements out-of-sample evaluation using grid search hyperparameters
optimized with Mean Absolute Error (MAE) as the validation metric.
"""
import sys
from pathlib import Path
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import datetime
import time
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import from project modules
from src.models import nns
from src.utils.metrics import compute_oos_r_square, compute_success_ratio, compute_MSFE_adjusted
from src.utils.metrics import compute_CER

def run(models=["Net1"], trials=None, epochs=30, threads=4, batch=256, 
        device='cpu', gamma_cer=3, oos_start_date=199001):
    """
    Run out-of-sample evaluation using grid search hyperparameters optimized with MAE.
    
    Parameters:
    -----------
    models : list
        List of model names to evaluate
    trials : int
        Not used for grid search, kept for API consistency
    epochs : int
        Number of epochs to train each model
    threads : int
        Number of threads to use
    batch : int
        Batch size for training
    device : str
        Device to use ('cpu' or 'cuda')
    gamma_cer : float
        Risk aversion parameter for CER calculation
    oos_start_date : int
        Start date for out-of-sample period (format: YYYYMM)
        
    Returns:
    --------
    None, results are saved to disk
    """
    print(f"Starting out-of-sample evaluation with MAE-optimized grid search parameters")
    print(f"Models: {models}")
    print(f"Epochs: {epochs}")
    print(f"Device: {device}")
    print(f"Batch size: {batch}")
    print(f"OOS start date: {oos_start_date}")
    
    # Set number of threads for torch
    torch.set_num_threads(threads)
    
    # Create timestamp for folder name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"./runs/5_MAE_OOS_grid_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data_path = Path('./data')
    predictor_raw = pd.read_excel(data_path / 'ml_equity_premium_data.xlsx', 
                                 sheet_name='result_predictor')
    predictor_raw.set_index('month', inplace=True)
    
    # Find OOS start index
    oos_start_idx = predictor_raw.index.get_loc(oos_start_date)
    
    # Risk-free rate for CER calculation
    rf = predictor_raw['TBL'].values / 1200  # Convert from annual % to monthly decimal
    
    # Save configuration
    with open(output_dir / "config.txt", "w") as f:
        f.write(f"Out-of-Sample Evaluation with MAE-Optimized Grid Search\n")
        f.write(f"=======================\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Models: {models}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Batch size: {batch}\n")
        f.write(f"Gamma CER: {gamma_cer}\n")
        f.write(f"OOS start date: {oos_start_date}\n")
        f.write(f"Threads: {threads}\n")
    
    # Find most recent in-sample MAE grid search results
    is_results_dirs = sorted([d for d in Path("./runs").glob("5_MAE_IS_grid_*")], reverse=True)
    
    if not is_results_dirs:
        print(f"Error: No in-sample MAE grid search results found. Run grid_mae_is_5.py first.")
        return
    
    is_results_dir = is_results_dirs[0]
    print(f"Using in-sample results from: {is_results_dir}")
    
    # Store all model results
    all_results = {}
    summary_data = []
    
    # Process each model
    for model_name in models:
        print(f"\nProcessing model: {model_name}")
        
        model_dir = output_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        # Check if model results exist from in-sample optimization
        model_is_dir = is_results_dir / model_name
        best_params_file = model_is_dir / "best_params.pkl"
        
        if not best_params_file.exists():
            print(f"Error: No in-sample results for {model_name} found at {best_params_file}")
            continue
        
        # Load best parameters
        with open(best_params_file, "rb") as f:
            is_results = pickle.load(f)
            best_params = is_results["best_params"]
        
        print(f"Using best parameters from in-sample optimization: {best_params}")
        
        # Load model class
        model_class = getattr(nns, model_name)
        
        # Initialize arrays for predictions and actuals
        y_pred_all = []
        y_actual_all = []
        dates_all = []
        
        # Process each time step in the OOS period
        for t in range(oos_start_idx, len(predictor_raw)):
            current_date = predictor_raw.index[t]
            print(f"  Processing {current_date}...", end="\r")
            
            # Use all data up to current time step for training
            X_train = predictor_raw.iloc[:t, 2:].values  # Skip log_equity_premium and equity_premium
            y_train = predictor_raw.iloc[:t]['log_equity_premium'].values
            
            # Create and train model with best parameters
            n_features = X_train.shape[1]
            model_config = {'n_feature': n_features, **best_params}
            model = model_class(**model_config)
            
            # Convert to tensors
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
            
            # Train the model
            criterion = nn.L1Loss()  # Use MAE loss for consistency
            optimizer = torch.optim.Adam(model.parameters(), lr=best_params.get('lr', 0.001))
            
            # Move model to device if using GPU
            if device == 'cuda' and torch.cuda.is_available():
                model = model.cuda()
                X_train_tensor = X_train_tensor.cuda()
                y_train_tensor = y_train_tensor.cuda()
            
            # Train for specified number of epochs
            model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()
            
            # Make prediction for the current time step
            model.eval()
            X_current = predictor_raw.iloc[t, 2:].values.reshape(1, -1)
            X_current_tensor = torch.tensor(X_current, dtype=torch.float32)
            if device == 'cuda' and torch.cuda.is_available():
                X_current_tensor = X_current_tensor.cuda()
            
            with torch.no_grad():
                prediction = model(X_current_tensor).cpu().numpy()[0, 0]
            
            # Store prediction
            y_pred_all.append(prediction)
            y_actual_all.append(predictor_raw.iloc[t]['log_equity_premium'])
            dates_all.append(current_date)
        
        print(f"  Completed processing {model_name}")
        
        # Calculate performance metrics
        y_pred = np.array(y_pred_all)
        y_actual = np.array(y_actual_all)
        
        # Historical average benchmark
        y_ha = []
        for i, t in enumerate(range(oos_start_idx, oos_start_idx + len(y_pred))):
            ha_window = predictor_raw.iloc[:t]['log_equity_premium'].values
            y_ha.append(np.mean(ha_window))
        y_ha = np.array(y_ha)
        
        # Calculate metrics
        oos_r2 = compute_oos_r_square(y_actual, y_ha, y_pred)
        msfe_adjusted, pvalue = compute_MSFE_adjusted(y_actual, y_ha, y_pred)
        success_ratio = compute_success_ratio(y_actual, y_pred)
        mse = mean_squared_error(y_actual, y_pred)
        mae = mean_absolute_error(y_actual, y_pred)
        
        # Calculate economic performance metrics
        rf_oos = rf[oos_start_idx:oos_start_idx + len(y_pred)]
        cer_gain = compute_CER(y_pred, rf_oos, gamma=gamma_cer) - compute_CER(y_ha, rf_oos, gamma=gamma_cer)
        
        # Store results
        model_results = {
            "model": model_name,
            "dates": dates_all,
            "y_pred": y_pred.tolist(),
            "y_actual": y_actual.tolist(),
            "y_ha": y_ha.tolist(),
            "oos_r2": oos_r2,
            "msfe_adjusted": msfe_adjusted,
            "msfe_pvalue": pvalue,
            "success_ratio": success_ratio,
            "cer_gain": cer_gain,
            "mse": mse,
            "mae": mae
        }
        
        all_results[model_name] = model_results
        
        # Append to summary data
        summary_data.append({
            "Model": model_name,
            "Method": "Grid Search (MAE)",
            "OOS R² (%)": oos_r2 * 100,
            "MSFE-adjusted": msfe_adjusted,
            "MSFE p-value": pvalue,
            "Success Ratio (%)": success_ratio * 100,
            "CER Gain (%)": cer_gain * 100,
            "MSE": mse,
            "MAE": mae
        })
        
        # Save model results
        with open(model_dir / "results.json", "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            json.dump(model_results, f)
        
        # Create a dataframe with the predictions
        pred_df = pd.DataFrame({
            "Date": dates_all,
            "Actual": y_actual,
            "Predicted": y_pred,
            "HA": y_ha
        })
        pred_df.to_csv(model_dir / "predictions.csv", index=False)
    
    # Create summary dataframe
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_dir / "summary.csv", index=False)
        
        # Create Excel file with results
        with pd.ExcelWriter(output_dir / "oos_results.xlsx") as writer:
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
    
    print(f"\nOut-of-sample evaluation complete. Results saved to {output_dir}")
    return output_dir
