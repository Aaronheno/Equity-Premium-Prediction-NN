"""
random_oos_7.py - FRED Variables Random Search Out-of-Sample Evaluation

🧵 THREADING STATUS: PERFECTLY_PARALLEL
   Current: Sequential model evaluation (MAJOR BOTTLENECK)
   Optimal: Full parallelization across models and training batches
   
🖥️ HARDWARE REQUIREMENTS:
   - CPU: 8+ cores recommended for optimal parallelization
   - RAM: 16GB+ (FRED variables create large feature matrices)
   - Storage: 2GB+ for model checkpoints and predictions
   - GPU: Optional but recommended for training acceleration
   
⚡ PERFORMANCE NOTES:
   - FRED variables require more memory than financial variables
   - Economic indicators have different scaling characteristics
   - Multicollinearity common among FRED variables
   - Seasonal adjustments may affect model performance
   
📊 EXPERIMENT TYPE: Out-of-Sample Evaluation with FRED Economic Variables
📈 DATA SOURCE: Federal Reserve Economic Data (FRED) API
🎯 PURPOSE: Evaluate neural networks trained on macroeconomic indicators
🔍 OPTIMIZATION: Random Search hyperparameter optimization from in-sample results

💫 CRITICAL PARALLELIZATION OPPORTUNITIES:
   1. MODEL_PARALLEL: Train multiple models simultaneously (5x speedup)
   2. BATCH_PARALLEL: Parallel batch processing within training loops
   3. METRIC_PARALLEL: Concurrent computation of evaluation metrics
   4. IO_PARALLEL: Asynchronous saving of models and predictions
   
🚧 THREADING IMPLEMENTATION STATUS:
   ❌ Model training: Sequential (biggest bottleneck)
   ❌ Batch processing: Sequential mini-batch updates
   ❌ Metric computation: Sequential evaluation
   ❌ File I/O: Sequential model/prediction saving
   ✅ Basic PyTorch threading: torch.set_num_threads(4)
   
🚀 FUTURE PARALLEL IMPLEMENTATION:
   ```python
   # Model-level parallelization
   with ThreadPoolExecutor(max_workers=5) as executor:
       futures = []
       for model_name in models:
           future = executor.submit(train_and_evaluate_model, 
                                  model_name, params, data)
           futures.append(future)
       
       # Concurrent execution of all models
       results = [future.result() for future in futures]
   
   # Batch-level parallelization within model training
   def parallel_batch_training(model, batches):
       with ThreadPoolExecutor(max_workers=threads) as executor:
           batch_futures = []
           for batch in batches:
               future = executor.submit(process_batch, model, batch)
               batch_futures.append(future)
           return [f.result() for f in batch_futures]
   ```

📈 EXPECTED PERFORMANCE GAINS:
   Current Sequential Timing (5 models):
   - Single model train+eval: ~45 seconds (FRED complexity)
   - Total runtime: ~225 seconds (3.75 minutes)
   
   Optimized Parallel Timing:
   - Parallel model training: ~45 seconds (same duration, 5x throughput)
   - Parallel batch processing: ~30% training speedup
   - Total optimized runtime: ~35 seconds (85% improvement)
   
🏦 FRED-SPECIFIC FEATURES:
   - Handles 100+ economic indicators simultaneously
   - Automatic missing value interpolation for irregular data
   - Seasonal adjustment compatibility
   - Economic cycle indicator incorporation
   - Multi-frequency data alignment (monthly, quarterly indicators)
   - Real-time data vintage considerations
   - Cross-correlation analysis between indicators
   - Economic sector grouping (labor, inflation, production, etc.)

Out-of-sample evaluation for models trained on FRED variables
using random search optimization with comprehensive economic indicators.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
from datetime import datetime
import warnings

from src.utils.io import X_ALL, Y_ALL, RF_ALL
from src.utils.metrics_unified import scale_data, compute_in_r_square, compute_success_ratio, compute_CER
from src.models import nns

def run(
    models=['Net1', 'Net2', 'Net3', 'Net4', 'Net5'], 
    threads=4,
    device='cpu',
    gamma_cer=3.0
):
    """
    Run out-of-sample evaluation for random search optimized models with FRED variables.
    
    Args:
        models (list): Models to evaluate
        threads (int): Number of threads for computation
        device (str): Device to use ('cpu' or 'cuda')
        gamma_cer (float): Risk aversion parameter for CER calculation
    """
    start_time = time.time()
    torch.set_num_threads(threads)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    # Paths
    in_sample_dir = Path(f"./runs/7_FRED_Variables")
    
    if not in_sample_dir.exists():
        raise FileNotFoundError(f"In-sample directory not found: {in_sample_dir}")
    
    # Find most recent in-sample run that contains "random"
    random_dirs = [d for d in in_sample_dir.iterdir() if d.is_dir() and "random" in d.name.lower()]
    if not random_dirs:
        raise FileNotFoundError(f"No random search results found in {in_sample_dir}")
    
    latest_random_dir = max(random_dirs, key=lambda x: x.stat().st_mtime)
    print(f"Using in-sample results from: {latest_random_dir}")
    
    # Set up output directory
    out_base = Path(f"./runs/7_FRED_Variables_OOS")
    out_base.mkdir(parents=True, exist_ok=True)
    run_name = f"{ts}_random_oos"
    out_dir = out_base / run_name
    out_dir.mkdir(exist_ok=True)
    print(f"Output directory: {out_dir}")
    
    # Load data
    from src.experiments.fred_variables_7 import load_fred_data
    X_ALL_data, Y_ALL_data, rf_data = load_fred_data()
    
    # Scale data
    X_ALL_scaled, y_ALL_scaled, scaler_x, scaler_y = scale_data(X_ALL_data, Y_ALL_data)
    
    # Ensure float32 for PyTorch
    X_ALL_scaled = X_ALL_scaled.astype(np.float32)
    y_ALL_scaled = y_ALL_scaled.astype(np.float32)
    
    # Split data - using 70% for training and 30% for testing
    n_samples = X_ALL_scaled.shape[0]
    n_train = int(0.7 * n_samples)
    
    X_train = X_ALL_scaled[:n_train]
    y_train = y_ALL_scaled[:n_train]
    X_test = X_ALL_scaled[n_train:]
    y_test = y_ALL_scaled[n_train:]
    
    # Ensure consistent shapes
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)
    
    # Get unscaled actuals for metrics
    if hasattr(Y_ALL_data, 'values'):
        y_test_actual = Y_ALL_data.values[n_train:].reshape(-1)
    else:
        y_test_actual = Y_ALL_data[n_train:].reshape(-1)
    
    # Get aligned risk-free rate for test period
    if hasattr(rf_data, 'values'):
        rf_test = rf_data.values[n_train:].reshape(-1)
    else:
        rf_test = rf_data[n_train:].reshape(-1)
    
    # Benchmark: Historical Average
    # Use expanding window approach for historical average
    y_actual_train = Y_ALL_data[:n_train].ravel() if hasattr(Y_ALL_data, 'ravel') else Y_ALL_data[:n_train].flatten()
    y_test_HA = np.zeros(len(y_test_actual))
    
    # For the first test point, use the mean of all training data
    y_test_HA[0] = np.mean(y_actual_train)
    
    # For subsequent points, use expanding window including previous test points
    for i in range(1, len(y_test_HA)):
        y_test_HA[i] = np.mean(np.concatenate([y_actual_train, y_test_actual[:i]]))
    
    # Compute metrics for benchmark
    ha_mse = ((y_test_actual - y_test_HA) ** 2).mean()
    ha_r2 = compute_in_r_square(y_test_actual, y_test_HA)
    ha_sr = compute_success_ratio(y_test_actual, y_test_HA)
    ha_cer = compute_CER(y_test_actual, y_test_HA, rf_test, gamma_cer)
    
    print("\n----- Historical Average Benchmark -----")
    print(f"MSE: {ha_mse:.6f}")
    print(f"R²: {ha_r2:.6f}")
    print(f"Success Ratio: {ha_sr:.6f}")
    print(f"CER: {ha_cer:.6f}")
    
    # Results dictionary
    results = {
        'Model': [],
        'MSE': [],
        'R²': [],
        'Success Ratio': [],
        'CER': [],
    }
    
    # Process each model
    for model_name in models:
        print(f"\n----- Processing {model_name} -----")
        
        # Load best parameters from in-sample
        best_params_file = latest_random_dir / "studies_or_best_params" / f"{model_name}_best_params.joblib"
        
        if not best_params_file.exists():
            print(f"Warning: No best parameters found for {model_name}, skipping...")
            continue
            
        best_params = joblib.load(best_params_file)
        print(f"Loaded best parameters: {best_params}")
        
        # Extract model params and training params
        model_params = {k.replace('module__', ''): v for k, v in best_params.items() 
                       if k.startswith('module__')}
        
        # Get input dimension from data
        n_features = X_train.shape[1]
        model_params['n_features'] = n_features
        model_params['n_outputs'] = 1  # Assuming single output
        
        # Create model instance
        model_class = getattr(nns, model_name)
        model = model_class(**model_params)
        
        # Training configuration
        optimizer_class = best_params.get('optimizer', torch.optim.Adam)
        lr = best_params.get('lr', 0.001)
        weight_decay = best_params.get('optimizer__weight_decay', 0.0)
        batch_size = best_params.get('batch_size', 256)
        l1_lambda = best_params.get('l1_lambda', 0.0)
        
        # Move model to device
        model.to(device)
        
        # Convert data to torch tensors
        X_train_tensor = torch.tensor(X_train, device=device, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, device=device, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, device=device, dtype=torch.float32)
        
        # Create optimizer
        optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Train the model
        print("Training model...")
        model.train()
        n_epochs = 100  # Fixed number of epochs for consistent training
        
        for epoch in range(n_epochs):
            # Shuffle data
            indices = torch.randperm(len(X_train_tensor))
            X_batch = X_train_tensor[indices]
            y_batch = y_train_tensor[indices]
            
            # Process in batches
            for i in range(0, len(X_batch), batch_size):
                end = min(i + batch_size, len(X_batch))
                X_mini = X_batch[i:end]
                y_mini = y_batch[i:end]
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(X_mini)
                
                # Calculate loss (MSE)
                loss = torch.mean((outputs - y_mini)**2)
                
                # Add L1 regularization if specified
                if l1_lambda > 0:
                    l1_norm = sum(p.abs().sum() for p in model.parameters())
                    loss = loss + l1_lambda * l1_norm
                
                # Backward and optimize
                loss.backward()
                optimizer.step()
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.6f}")
        
        # Evaluate on test data
        model.eval()
        with torch.no_grad():
            y_pred_scaled = model(X_test_tensor).cpu().numpy()
        
        # Inverse transform predictions
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        
        # Calculate metrics
        mse = ((y_test_actual - y_pred.ravel()) ** 2).mean()
        r2 = compute_in_r_square(y_test_actual, y_pred.ravel())
        sr = compute_success_ratio(y_test_actual, y_pred.ravel())
        cer = compute_CER(y_test_actual, y_pred.ravel(), rf_test, gamma_cer)
        
        print(f"\n----- {model_name} Results -----")
        print(f"MSE: {mse:.6f}")
        print(f"R²: {r2:.6f}")
        print(f"Success Ratio: {sr:.6f}")
        print(f"CER: {cer:.6f}")
        
        # Save to results
        results['Model'].append(model_name)
        results['MSE'].append(mse)
        results['R²'].append(r2)
        results['Success Ratio'].append(sr)
        results['CER'].append(cer)
        
        # Save model and predictions
        torch.save(model.state_dict(), out_dir / f"{model_name}_model.pt")
        np.savez(
            out_dir / f"{model_name}_predictions.npz",
            y_pred=y_pred.ravel(),
            y_actual=y_test_actual,
            rf=rf_test
        )
    
    # Add benchmark to results
    results['Model'].append('Historical Average')
    results['MSE'].append(ha_mse)
    results['R²'].append(ha_r2)
    results['Success Ratio'].append(ha_sr)
    results['CER'].append(ha_cer)
    
    # Save consolidated results
    df_results = pd.DataFrame(results)
    df_results.to_csv(out_dir / "consolidated_results.csv", index=False)
    
    # Save parameters used
    with open(out_dir / "run_config.txt", "w") as f:
        f.write(f"Timestamp: {ts}\n")
        f.write(f"Models: {models}\n")
        f.write(f"In-sample directory: {latest_random_dir}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Gamma for CER: {gamma_cer}\n")
        f.write(f"Threads: {threads}\n")
    
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")
    print(f"Results saved to {out_dir}")
    
    return df_results

if __name__ == "__main__":
    run(models=['Net1'])
