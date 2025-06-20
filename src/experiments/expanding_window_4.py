"""
Expanding Window Analysis for Equity Premium Prediction

This experiment conducts expanding window out-of-sample evaluation with configurable
minimum window sizes. Features dual-level parallelization opportunities with
independent window processing and concurrent model optimization.

Threading Status: PARALLEL_READY (Window-level and model-level parallelism)
Hardware Requirements: CPU_INTENSIVE, CUDA_BENEFICIAL, HIGH_MEMORY_PREFERRED
Performance Notes:
    - Window parallelism: 2-4x speedup (multiple minimum window sizes)
    - Model parallelism: 8x speedup (8 models per window)
    - Memory usage: High due to expanding windows
    - CPU-intensive: Benefits from multi-core systems

Experiment Type: Expanding Window Out-of-Sample Analysis
Window Types: Minimum window sizes with continuous expansion
Models Supported: Net1, Net2, Net3, Net4, Net5, DNet1, DNet2, DNet3
HPO Methods: Grid Search, Random Search, Bayesian Optimization
Output Directory: runs/4_Expanding_Window_{method}_{timestamp}/

Critical Parallelization Opportunities:
    1. Independent window size processing (2-4x speedup)
    2. Model HPO within each window (8x speedup per window)
    3. Concurrent time period evaluation within windows
    4. Parallel metrics computation across windows and models

Threading Implementation Status:
    ❌ Sequential window processing (MAIN BOTTLENECK)
    ❌ Sequential model processing within windows
    ❌ Sequential time period evaluation
    ❌ Sequential metrics computation

Future Parallel Implementation:
    run_expanding_window(window_sizes, models, parallel_windows=True, parallel_models=True)
    
Expected Performance Gains:
    - Current: Sequential processing across all dimensions
    - With window parallelism: 2.7x speedup
    - With model parallelism: Additional 7x speedup
    - Combined on high-core systems: Up to 60-160x speedup

Expanding Window Advantages:
    - Captures increasing data availability over time
    - More stable than rolling windows for recent periods
    - Better long-term trend detection
    - Realistic real-world forecasting scenario
"""
import sys
from pathlib import Path
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import from project modules
from src.models import nns
from src.utils.metrics import compute_oos_r_square, compute_success_ratio, compute_MSFE_adjusted
from src.configs.search_spaces import GRID_OOS as GRID_SPACE
from src.configs.search_spaces import RANDOM_OOS as RANDOM_SPACE
from src.configs.search_spaces import BAYES_OOS as BAYES_SPACE

def run_expanding_window(
    model_names=["Net1"],
    window_sizes=[1, 3],  # Window sizes in years
    oos_start_date_int=199001,
    optimization_method="grid",  # "grid", "random", or "bayes"
    hpo_general_config=None,
    save_results=True,
):
    """
    Run expanding window analysis with hyperparameter optimization.
    
    Parameters:
    -----------
    model_names : list
        List of model names to evaluate
    window_sizes : list
        List of minimum window sizes in years (expands as time progresses)
    oos_start_date_int : int
        Start date for out-of-sample period (format: YYYYMM)
    optimization_method : str
        Hyperparameter optimization method ("grid", "random", or "bayes")
    hpo_general_config : dict
        Configuration for hyperparameter optimization
    save_results : bool
        Whether to save results to disk
        
    Returns:
    --------
    results : dict
        Dictionary with performance metrics for all models and window sizes
    """
    # Default HPO configuration if not provided
    if hpo_general_config is None:
        hpo_general_config = {
            'hpo_epochs': 30,
            'hpo_trials': 10,
            'hpo_device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'hpo_batch_size': 256
        }
    
    print(f"Starting expanding window analysis for {model_names} with {optimization_method} optimization")
    print(f"Minimum window sizes: {window_sizes} years")
    print(f"OOS start date: {oos_start_date_int}")
    print(f"Device: {hpo_general_config['hpo_device']}")
    
    # Load data
    data_path = Path('./data')
    predictor_raw = pd.read_excel(data_path / 'ml_equity_premium_data.xlsx', 
                                 sheet_name='result_predictor')
    predictor_raw.set_index('month', inplace=True)
    
    # Create timestamp for folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path(f"./runs/4_Expanding_Window_{optimization_method}_{timestamp}")
    
    if save_results:
        base_output_dir.mkdir(parents=True, exist_ok=True)
        # Save configuration
        with open(base_output_dir / "config.txt", "w") as f:
            f.write(f"Expanding Window Analysis\n")
            f.write(f"=======================\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Models: {model_names}\n")
            f.write(f"Optimization method: {optimization_method}\n")
            f.write(f"Minimum window sizes: {window_sizes} years\n")
            f.write(f"OOS start date: {oos_start_date_int}\n")
            f.write(f"HPO config: {hpo_general_config}\n")
    
    # Find OOS start index
    oos_start_idx = predictor_raw.index.get_loc(oos_start_date_int)
    
    # Store results for all models and window sizes
    all_results = {}
    
    # Process each model
    for model_name in model_names:
        print(f"\nProcessing model: {model_name}")
        model_class = getattr(nns, model_name)
        
        model_results = {}
        
        for min_window_size in window_sizes:
            print(f"  Minimum window size: {min_window_size} years")
            min_window_months = min_window_size * 12
            
            # Calculate window requirements
            # For valid expanding window, we need at least min_window_size data before OOS start
            min_required_idx = oos_start_idx - min_window_months
            if min_required_idx < 0:
                print(f"  Warning: Not enough data for {min_window_size} year window before OOS start")
                print(f"  Need at least {min_window_months} months before OOS start date")
                continue
            
            # Initialize performance metrics
            y_pred_all = []
            y_actual_all = []
            dates_all = []
            
            # Process each time step in the OOS period
            for t in range(oos_start_idx, len(predictor_raw)):
                current_date = predictor_raw.index[t]
                print(f"  Processing {current_date}...", end="\r")
                
                # Define expanding window with minimum size
                window_start_idx = max(0, oos_start_idx - min_window_months)  # Start from min window before OOS
                window_data = predictor_raw.iloc[window_start_idx:t]
                
                # Skip if we don't have enough data
                if len(window_data) < min_window_months:
                    print(f"  Warning: Skipping {current_date} - insufficient window data")
                    continue
                
                # Prepare training data
                X = window_data.iloc[:, 2:].values  # Skip log_equity_premium and equity_premium
                y = window_data['log_equity_premium'].values
                
                # Split into training and validation sets
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.15, random_state=42
                )
                
                # Run hyperparameter optimization based on the specified method
                if optimization_method == "grid":
                    # Use Grid Search
                    from src.utils.training_grid import run_grid_search
                    best_params = run_grid_search(
                        X_train, y_train, X_val, y_val, model_class, 
                        GRID_SPACE[model_name],
                        epochs=hpo_general_config['hpo_epochs'],
                        batch_size=hpo_general_config['hpo_batch_size'],
                        device=hpo_general_config['hpo_device']
                    )
                elif optimization_method == "random":
                    # Use Random Search
                    from src.utils.training_random import run_random_search
                    best_params = run_random_search(
                        X_train, y_train, X_val, y_val, model_class, 
                        RANDOM_SPACE[model_name],
                        epochs=hpo_general_config['hpo_epochs'],
                        n_trials=hpo_general_config['hpo_trials'],
                        batch_size=hpo_general_config['hpo_batch_size'],
                        device=hpo_general_config['hpo_device']
                    )
                elif optimization_method == "bayes":
                    # Use Bayesian Optimization
                    from src.utils.training_optuna import run_bayesian_optimization
                    best_params = run_bayesian_optimization(
                        X_train, y_train, X_val, y_val, model_class, 
                        BAYES_SPACE[model_name],
                        epochs=hpo_general_config['hpo_epochs'],
                        n_trials=hpo_general_config['hpo_trials'],
                        batch_size=hpo_general_config['hpo_batch_size'],
                        device=hpo_general_config['hpo_device']
                    )
                else:
                    raise ValueError(f"Unknown optimization method: {optimization_method}")
                
                # Train final model with best parameters
                n_features = X_train.shape[1]
                model_config = {'n_feature': n_features, **best_params}
                model = model_class(**model_config)
                
                # Convert to tensors
                X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
                y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
                
                # Train the model
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=best_params.get('lr', 0.001))
                
                # Move model to device if using GPU
                if hpo_general_config['hpo_device'] == 'cuda':
                    model = model.cuda()
                    X_train_tensor = X_train_tensor.cuda()
                    y_train_tensor = y_train_tensor.cuda()
                
                # Train for specified number of epochs
                model.train()
                for epoch in range(hpo_general_config['hpo_epochs']):
                    optimizer.zero_grad()
                    outputs = model(X_train_tensor)
                    loss = criterion(outputs, y_train_tensor)
                    loss.backward()
                    optimizer.step()
                
                # Make prediction for the current time step
                model.eval()
                X_current = predictor_raw.iloc[t, 2:].values.reshape(1, -1)
                X_current_tensor = torch.tensor(X_current, dtype=torch.float32)
                if hpo_general_config['hpo_device'] == 'cuda':
                    X_current_tensor = X_current_tensor.cuda()
                
                with torch.no_grad():
                    prediction = model(X_current_tensor).cpu().numpy()[0, 0]
                
                # Store prediction
                y_pred_all.append(prediction)
                y_actual_all.append(predictor_raw.iloc[t]['log_equity_premium'])
                dates_all.append(current_date)
            
            print(f"  Completed processing for {min_window_size} year expanding window")
            
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
            
            # Store results
            window_results = {
                'dates': dates_all,
                'y_pred': y_pred,
                'y_actual': y_actual,
                'y_ha': y_ha,
                'oos_r2': oos_r2,
                'msfe_adjusted': msfe_adjusted,
                'msfe_pvalue': pvalue,
                'success_ratio': success_ratio
            }
            
            model_results[min_window_size] = window_results
            
            # Save results for this window size
            if save_results:
                window_dir = base_output_dir / f"{model_name}_window{min_window_size}"
                window_dir.mkdir(exist_ok=True)
                
                # Save predictions
                pred_df = pd.DataFrame({
                    'Date': dates_all,
                    'Actual': y_actual,
                    'Predicted': y_pred,
                    'HA': y_ha
                })
                pred_df.to_csv(window_dir / "predictions.csv", index=False)
                
                # Save metrics
                metrics_df = pd.DataFrame({
                    'Metric': ['OOS R² (%)', 'MSFE-adjusted', 'MSFE p-value', 'Success Ratio (%)'],
                    'Value': [
                        oos_r2 * 100, 
                        msfe_adjusted, 
                        pvalue,
                        success_ratio * 100
                    ]
                })
                metrics_df.to_csv(window_dir / "metrics.csv", index=False)
        
        all_results[model_name] = model_results
    
    # Combine all results
    if save_results:
        # Create summary of results
        summary_data = []
        for model_name, model_results in all_results.items():
            for window_size, window_results in model_results.items():
                summary_data.append({
                    'Model': model_name,
                    'Min Window Size (Years)': window_size,
                    'Method': optimization_method.upper(),
                    'OOS R² (%)': window_results['oos_r2'] * 100,
                    'MSFE-adjusted': window_results['msfe_adjusted'],
                    'MSFE p-value': window_results['msfe_pvalue'],
                    'Success Ratio (%)': window_results['success_ratio'] * 100
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(base_output_dir / "summary_results.csv", index=False)
            
            # Create Excel file with multiple sheets
            with pd.ExcelWriter(base_output_dir / "expanding_window_results.xlsx") as writer:
                summary_df.to_excel(writer, sheet_name="Summary", index=False)
                
                # Add sheets for each window size
                window_sizes_available = set()
                for model_results in all_results.values():
                    window_sizes_available.update(model_results.keys())
                
                for window_size in sorted(window_sizes_available):
                    window_data = []
                    for model_name, model_results in all_results.items():
                        if window_size in model_results:
                            window_data.append({
                                'Model': model_name,
                                'Method': optimization_method.upper(),
                                'OOS R² (%)': model_results[window_size]['oos_r2'] * 100,
                                'MSFE-adjusted': model_results[window_size]['msfe_adjusted'],
                                'MSFE p-value': model_results[window_size]['msfe_pvalue'],
                                'Success Ratio (%)': model_results[window_size]['success_ratio'] * 100
                            })
                    
                    if window_data:
                        window_df = pd.DataFrame(window_data)
                        window_df.to_excel(writer, sheet_name=f"{window_size}yr Window", index=False)
    
    return all_results
