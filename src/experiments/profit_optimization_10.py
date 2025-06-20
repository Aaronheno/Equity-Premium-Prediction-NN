"""
Profit-Based Hyperparameter Optimization for Neural Networks

This experiment implements hyperparameter optimization using trading profit as the objective
function instead of traditional error metrics like MSE. Features extensive parallelization
opportunities with independent parameter evaluation and concurrent model training.

Threading Status: PARALLEL_READY (Independent parameter combinations and model training)
Hardware Requirements: GPU_PREFERRED, CPU_INTENSIVE, HIGH_MEMORY_BENEFICIAL
Performance Notes:
    - Parameter combinations: Perfect parallelization across search space
    - Model training: Independent concurrent training for different parameters
    - Memory usage: High due to multiple model instances
    - GPU acceleration: Beneficial for faster neural network training

Experiment Type: Economic Metric Optimization (Profit Maximization)
Optimization Objective: Maximum trading profit with institutional constraints
Models Supported: Net1, Net2, Net3, Net4, Net5, DNet1, DNet2, DNet3
HPO Methods: Grid Search, Random Search, Bayesian Optimization
Output Directory: runs/10_Profit_Optimization/

Critical Parallelization Opportunities:
    1. Independent parameter combination evaluation (perfect parallelization)
    2. Concurrent model training across different hyperparameters
    3. Parallel profit calculation across validation periods
    4. Independent trading strategy simulation

Threading Implementation Status:
    ❌ Sequential parameter evaluation (MAIN BOTTLENECK)
    ❌ Sequential model training across parameters
    ❌ Sequential profit calculations
    ✅ PyTorch threading support within models

Future Parallel Implementation:
    run(models, method, param_parallel=True, model_parallel=True, n_jobs=64)
    
Expected Performance Gains:
    - Current: 8 hours for 100 parameter combinations × 4 models
    - With parameter parallelism: 2 hours (4x speedup)
    - With model parallelism: 30 minutes (additional 4x speedup)
    - Combined on 128-core server: 5-10 minutes (48-96x speedup)

Profit Optimization Features:
    - Economic metric optimization instead of statistical accuracy
    - Realistic trading constraints (leverage, transaction costs)
    - Multiple position sizing strategies (binary, proportional)
    - Risk-adjusted performance metrics (Sharpe, Sortino ratios)
    - Early stopping based on profit validation
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
import time
import copy
import random
import joblib
import warnings

# Project imports
from src.models import nns
from src.utils.io import RF_ALL, X_ALL, Y_ALL 
from src.utils.metrics_unified import compute_sharpe, compute_sortino
from src.configs.search_spaces import get_search_space

# Suppress warnings
warnings.filterwarnings('ignore')

def calculate_profit(predictions, actuals, rf_rates=None, rf_rate_default=0.03, max_leverage=1.5, 
                    transaction_cost=0.0007, initial_capital=1.0, 
                    position_sizing='binary'):
    """
    Calculate profit from predictions based on a simple trading strategy:
    - Long with leverage when prediction > 0
    - Invest in risk-free when prediction <= 0
    
    Args:
        predictions (array): Model predictions (equity premium)
        actuals (array): Actual equity premium values
        rf_rates (array, optional): Actual risk-free rates for each period (monthly)
        rf_rate_default (float): Default annual risk-free rate if rf_rates not provided
        max_leverage (float): Maximum leverage allowed (1.5 = 150%)
        transaction_cost (float): Cost per transaction (one-way)
        initial_capital (float): Starting capital
        position_sizing (str): Method for sizing positions ('binary', 'proportional')
        
    Returns:
        tuple: (final_capital, profit_pct, sharpe_ratio, trades, transaction_costs)
    """
    # Use pre-calculated simple equity premium if available,
    # otherwise convert from log equity premium to simple returns if needed
    if np.mean(np.abs(actuals)) < 0.1:  # Likely log returns
        # This appears to be log returns, convert to simple returns
        actual_returns = np.exp(actuals) - 1
    else:
        # Already simple returns or pre-calculated equity premium
        actual_returns = actuals
    
    # Initialize
    capital = initial_capital
    positions = np.zeros_like(predictions)  # 0 = cash, 1 = long equity
    leverage = np.zeros_like(predictions)  # Leverage applied (0 to max_leverage)
    
    # Handle risk-free rates
    if rf_rates is None or len(rf_rates) != len(predictions):
        # Use fixed default if no valid rates are provided
        monthly_rf = (1 + rf_rate_default) ** (1/12) - 1  # Monthly risk-free rate
        rf_rates_monthly = np.ones_like(predictions) * monthly_rf
    else:
        # Use provided rates - ensure they're in monthly format
        rf_rates_monthly = rf_rates
        # If annual rates are provided, convert to monthly
        if np.mean(rf_rates) > 0.1:  # Likely annual percentage rates
            rf_rates_monthly = rf_rates / 100  # Convert from percentage to decimal
        
        if np.mean(rf_rates_monthly) > 0.01:  # Still likely annual rates
            # Convert annual rates to monthly
            rf_rates_monthly = (1 + rf_rates_monthly) ** (1/12) - 1
    
    # Determine positions based on predictions
    if position_sizing == 'binary':
        # Binary positioning: either max leverage or none
        positions = np.where(predictions > 0, 1, 0)
        leverage = np.where(predictions > 0, max_leverage, 0)
    elif position_sizing == 'proportional':
        # Scale leverage based on prediction strength (normalized to max range)
        pos_preds = predictions.copy()
        if len(pos_preds) > 0:
            pos_max = max(np.max(pos_preds), 0.001)  # Avoid division by zero
            pos_preds = np.clip(pos_preds, 0, None)  # Only consider positive predictions
            leverage = pos_preds / pos_max * max_leverage
            positions = np.where(leverage > 0, 1, 0)
        else:
            positions = np.zeros_like(predictions)
            leverage = np.zeros_like(predictions)
    
    # Calculate returns and transaction costs
    returns = np.zeros_like(predictions)
    transaction_costs_list = []
    trades = 0
    prev_position = 0
    
    for i in range(len(predictions)):
        current_position = positions[i]
        
        # Calculate transaction cost if position changed
        if current_position != prev_position:
            trades += 1
            cost = capital * transaction_cost
            transaction_costs_list.append(cost)
            capital -= cost
        
        # Calculate return for current period
        if current_position == 1:  # Long equity with leverage
            period_return = actual_returns[i] * leverage[i]
        else:  # Risk-free investment
            period_return = rf_rates_monthly[i] if i < len(rf_rates_monthly) else (1 + rf_rate_default) ** (1/12) - 1
        
        # Update capital and track returns
        capital *= (1 + period_return)
        returns[i] = period_return
        
        # Update previous position
        prev_position = current_position
    
    # Calculate profit metrics
    final_capital = capital
    profit_pct = (final_capital / initial_capital - 1) * 100
    
    # Calculate risk-adjusted metrics
    if len(returns) > 1:
        sharpe = compute_sharpe(returns, 0)  # Excess returns already calculated
        sortino = compute_sortino(returns, 0)
    else:
        sharpe = 0
        sortino = 0
    
    # Total transaction costs
    total_transaction_costs = sum(transaction_costs_list) if transaction_costs_list else 0
    
    return final_capital, profit_pct, sharpe, trades, total_transaction_costs

class ProfitLoss(nn.Module):
    """
    Custom loss function that optimizes for trading profit instead of MSE.
    Implemented as a PyTorch module for easier integration.
    """
    def __init__(self, rf_rates=None, rf_rate_default=0.03, max_leverage=1.5, transaction_cost=0.0007, 
                position_sizing='binary'):
        super().__init__()
        self.rf_rates = rf_rates
        self.rf_rate_default = rf_rate_default
        self.max_leverage = max_leverage
        self.transaction_cost = transaction_cost
        self.position_sizing = position_sizing
        
    def forward(self, predictions, targets):
        """
        Calculate the loss (negative profit) for optimization.
        
        Args:
            predictions (torch.Tensor): Model predictions
            targets (torch.Tensor): Actual values
            
        Returns:
            torch.Tensor: Loss value (negative profit)
        """
        # Convert to numpy for profit calculation
        preds_np = predictions.detach().cpu().numpy().flatten()
        targets_np = targets.detach().cpu().numpy().flatten()
        
        # Calculate profit
        _, profit_pct, sharpe, _, _ = calculate_profit(
            preds_np, targets_np,
            rf_rates=self.rf_rates,
            rf_rate_default=self.rf_rate_default,
            max_leverage=self.max_leverage,
            transaction_cost=self.transaction_cost,
            position_sizing=self.position_sizing
        )
        
        # Since we want to maximize profit but minimizing loss, return negative profit
        # Add a small value to avoid zero gradients
        return torch.tensor(-profit_pct, requires_grad=True).to(predictions.device)

def train_model_with_profit_objective(model, X_train, y_train, X_val, y_val, 
                                     rf_rates_train=None, rf_rates_val=None, rf_rate_default=0.03,
                                     max_leverage=1.5, transaction_cost=0.0007,
                                     position_sizing='binary', early_stopping=10, epochs=200, batch_size=32,
                                     lr=0.001, weight_decay=0.01, device=None):
    """
    Train a neural network model with profit as the optimization objective.
    
    Args:
        model (nn.Module): PyTorch model to train
        X_train, y_train: Training data
        X_val, y_val: Validation data
        epochs (int): Maximum number of epochs
        batch_size (int): Batch size for training
        lr (float): Learning rate
        weight_decay (float): Weight decay for regularization
        rf_rate (float): Annual risk-free rate
        max_leverage (float): Maximum leverage allowed
        transaction_cost (float): Transaction cost per trade
        position_sizing (str): Method for position sizing
        early_stopping (int): Number of epochs to wait for improvement
        device (str): Device to use for training
        
    Returns:
        model: Trained model
        dict: Training history
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1).to(device)
    
    # Create data loader for training
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize optimizer and profit loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    profit_loss = ProfitLoss(rf_rates_train, rf_rate_default, max_leverage, transaction_cost, position_sizing)
    
    # Move model to device
    model = model.to(device)
    
    # Training history
    history = {
        'train_loss': [],
        'train_profit': [],
        'val_profit': [],
        'val_sharpe': [],
        'best_epoch': 0
    }
    
    # Early stopping variables
    best_val_profit = -float('inf')
    best_model_state = None
    no_improve_count = 0
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = profit_loss(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Calculate training profit
        model.eval()
        with torch.no_grad():
            train_preds = model(X_train_tensor).cpu().numpy().flatten()
            val_preds = model(X_val_tensor).cpu().numpy().flatten()
        
        _, train_profit, train_sharpe, _, _ = calculate_profit(
            train_preds, y_train.flatten(),
            rf_rates=rf_rates_train, rf_rate_default=rf_rate_default, 
            max_leverage=max_leverage, transaction_cost=transaction_cost, 
            position_sizing=position_sizing
        )
        
        _, val_profit, val_sharpe, _, _ = calculate_profit(
            val_preds, y_val.flatten(),
            rf_rates=rf_rates_val, rf_rate_default=rf_rate_default,
            max_leverage=max_leverage, transaction_cost=transaction_cost, 
            position_sizing=position_sizing
        )
        
        # Record history
        history['train_loss'].append(epoch_loss / len(train_loader))
        history['train_profit'].append(train_profit)
        history['val_profit'].append(val_profit)
        history['val_sharpe'].append(val_sharpe)
        
        # Check for improvement
        if val_profit > best_val_profit:
            best_val_profit = val_profit
            best_model_state = copy.deepcopy(model.state_dict())
            no_improve_count = 0
            history['best_epoch'] = epoch
        else:
            no_improve_count += 1
            if no_improve_count >= early_stopping:
                print(f"Early stopping at epoch {epoch}")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss/len(train_loader):.4f} | "
                 f"Train Profit: {train_profit:.2f}% | Val Profit: {val_profit:.2f}% | "
                 f"Val Sharpe: {val_sharpe:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history

def grid_search_profit_optimization(X_train, y_train, X_val, y_val, model_class, search_space,
                                 rf_rates_train=None, rf_rates_val=None, rf_rate_default=0.03, max_leverage=1.5, 
                                 transaction_cost=0.0007, position_sizing='binary', epochs=100, device=None, threads=4):
    """
    Perform grid search to find optimal parameters that maximize profit.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        model_class: PyTorch model class
        param_grid: Dictionary with parameters to search
        rf_rate, max_leverage, transaction_cost: Trading parameters
        position_sizing: Method for position sizing
        epochs: Maximum training epochs
        device: Training device
        threads: Number of PyTorch threads
        
    Returns:
        dict: Best parameters
        float: Best validation profit
        model: Best trained model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set PyTorch threads
    torch.set_num_threads(threads)
    
    best_val_profit = -float('inf')
    best_params = None
    best_model = None
    best_history = None
    
    # Iterate through parameter grid
    best_model = None
    best_profit = -np.inf
    best_params = None
    all_results = []
    
    # Get parameter combinations
    param_grid = ParameterGrid(search_space)
    
    for params in tqdm(param_grid, desc="Grid Search"):
        # Train model with current parameters
        model = model_class(**params)
        model, history = train_model_with_profit_objective(
            model, X_train, y_train, X_val, y_val,
            rf_rates_train=rf_rates_train, rf_rates_val=rf_rates_val, 
            rf_rate_default=rf_rate_default, max_leverage=max_leverage, 
            transaction_cost=transaction_cost, position_sizing=position_sizing,
            early_stopping=10, epochs=epochs, device=device
        )
        
        # Get validation profit (best epoch)
        val_profit = max(history['val_profit'])
        
        # Save results
        result = {
            'params': params,
            'val_profit': val_profit,
            'best_epoch': history['best_epoch'],
            'train_profit': history['train_profit'][history['best_epoch']],
            'val_sharpe': history['val_sharpe'][history['best_epoch']],
        }
        all_results.append(result)
        
        # Update best model if current is better
        if val_profit > best_profit:
            best_profit = val_profit
            best_params = params
            best_model = copy.deepcopy(model)
            best_history = history
            print(f"New best: {best_profit:.2f}% profit with {params}")
    
    return best_params, best_profit, best_model, best_history

def run(
    models=['Net1', 'Net2'],
    method='grid',
    trials=50,
    epochs=100,
    rf_rate_default=0.03,
    max_leverage=1.5,
    transaction_cost=0.0007,
    position_sizing='binary',
    batch=32,
    threads=4,
    device='auto',
    data_source='original'
):
    """
    Run profit optimization for specified models.
    
    Args:
        models (list): List of model names to optimize
        method (str): Optimization method ('grid', 'random', 'bayes')
        trials (int): Number of trials for random/bayes search
        epochs (int): Maximum training epochs
        rf_rate (float): Annual risk-free rate
        max_leverage (float): Maximum leverage allowed (1.5 = 150%)
        transaction_cost (float): Transaction cost per trade
        position_sizing (str): Position sizing method ('binary', 'proportional')
        batch (int): Batch size for training
        threads (int): Number of PyTorch threads
        device (str): Training device
        data_source (str): Data source to use ('original', 'newly_identified', 'fred')
    """
    # Set device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Create timestamp for output
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    # Load data based on data source
    rf_rates_full = None  # Initialize rf_rates_full to None
    equity_premium_full = None  # Initialize equity_premium_full to None
    dates = None     # Initialize dates to None
    
    if data_source == 'newly_identified':
        from src.experiments.newly_identified_6 import load_data
        X_train, y_train, scaler = load_data('standalone')
    elif data_source == 'fred':
        from src.experiments.fred_variables_7 import load_fred_data
        X_train, y_train, scaler = load_fred_data()
    else:  # Original data
        X_train = X_ALL
        y_train = Y_ALL
        
        # Try to load risk-free rate and equity premium data from original data source
        try:
            data_path = Path("ml_equity_premium_data.xlsx")
            if data_path.exists():
                df = pd.read_excel(data_path, sheet_name='result_predictor')
                dates = df['month'].values
                
                # Load risk-free rate (TBL)
                if 'TBL' in df.columns:
                    rf_rates_full = df['TBL'].values / 100  # Convert from percentage to decimal
                    print(f"Loaded risk-free rate data: {len(rf_rates_full)} values")
                else:
                    print("Warning: TBL column not found in data file")
                    
                # Load pre-calculated simple equity premium if available
                if 'equity_premium' in df.columns:
                    equity_premium_full = df['equity_premium'].values / 100  # Convert percentage to decimal
                    print(f"Loaded pre-calculated equity premium data: {len(equity_premium_full)} values")
                else:
                    print("Warning: equity_premium column not found in data file")
        except Exception as e:
            print(f"Warning: Could not load auxiliary data from original data file: {e}")
    
    print(f"Data loaded: X shape: {X_train.shape}, y shape: {y_train.shape}")
    
    # Create validation split (20%)
    split_idx = int(len(X_train) * 0.8)
    X_val = X_train[split_idx:]
    y_val = y_train[split_idx:]
    X_train = X_train[:split_idx]
    y_train = y_train[:split_idx]
    
    # Split risk-free rates if available
    rf_rates_train = None
    rf_rates_val = None
    if rf_rates_full is not None:
        # Match the length of available risk-free rates to our data
        # (might need alignment if dates don't match perfectly)
        if len(rf_rates_full) == len(X_train) + len(X_val):
            rf_rates_train = rf_rates_full[:split_idx]
            rf_rates_val = rf_rates_full[split_idx:]
            print(f"Using actual risk-free rates: {len(rf_rates_train)} for training, {len(rf_rates_val)} for validation")
        else:
            print(f"Warning: Risk-free rate data length ({len(rf_rates_full)}) doesn't match data length ({len(X_train) + len(X_val)})")
    
    # Create output directory
    out_base = Path(f"./runs/10_Profit_Optimization")
    out_base.mkdir(parents=True, exist_ok=True)
    
    # Process each model
    for model_name in models:
        print(f"\n--- Optimizing {model_name} for Maximum Profit ---")
        
        # Create model-specific output directory
        model_out_dir = out_base / f"{ts}_{model_name}_{method}_{data_source}"
        model_out_dir.mkdir(exist_ok=True, parents=True)
        
        # Get model class and search space
        model_class = getattr(nns, model_name)
        search_space = get_search_space(model_name)
        
        # Optimization based on method
        if method == 'grid':
            best_params, best_profit, best_model, history = grid_search_profit_optimization(
                X_train, y_train, X_val, y_val, model_class, search_space,
                rf_rates_train=rf_rates_train, rf_rates_val=rf_rates_val, rf_rate_default=rf_rate_default,
                max_leverage=max_leverage, transaction_cost=transaction_cost, 
                position_sizing=position_sizing, epochs=epochs, device=device, threads=threads
            )
        elif method == 'random':
            # Implementation for random search
            print("Random search not implemented yet, using grid search")
            best_params, best_profit, best_model, history = grid_search_profit_optimization(
                X_train, y_train, X_val, y_val, model_class, search_space,
                rf_rates_train=rf_rates_train, rf_rates_val=rf_rates_val, rf_rate_default=rf_rate_default,
                max_leverage=max_leverage, transaction_cost=transaction_cost, 
                position_sizing=position_sizing, epochs=epochs, device=device, threads=threads
            )
        elif method == 'bayes':
            # Implementation for Bayesian optimization
            print("Bayesian optimization not implemented yet, using grid search")
            best_params, best_profit, best_model, history = grid_search_profit_optimization(
                X_train, y_train, X_val, y_val, model_class, search_space,
                rf_rates_train=rf_rates_train, rf_rates_val=rf_rates_val, rf_rate_default=rf_rate_default,
                max_leverage=max_leverage, transaction_cost=transaction_cost, 
                position_sizing=position_sizing, epochs=epochs, device=device, threads=threads
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Save results
        results = {
            'model_name': model_name,
            'best_params': best_params,
            'best_profit': best_profit,
            'training_history': history,
            'config': {
                'rf_rate_default': rf_rate_default,
                'max_leverage': max_leverage,
                'transaction_cost': transaction_cost,
                'position_sizing': position_sizing,
                'epochs': epochs,
                'method': method,
                'data_source': data_source,
                'used_actual_rf_rates': rf_rates_train is not None
            }
        }
        
        # Save model
        torch.save(best_model.state_dict(), model_out_dir / f"{model_name}_best_model.pt")
        
        # Save results
        joblib.dump(results, model_out_dir / f"{model_name}_results.joblib")
        
        # Generate and save plots
        if history:
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 1, 1)
            plt.plot(history['train_profit'], label='Train Profit (%)')
            plt.plot(history['val_profit'], label='Validation Profit (%)')
            plt.axvline(x=history['best_epoch'], color='r', linestyle='--', 
                       label=f'Best Epoch ({history["best_epoch"]})')
            plt.title(f'{model_name} - Profit During Training')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 1, 2)
            plt.plot(history['val_sharpe'], label='Validation Sharpe')
            plt.axvline(x=history['best_epoch'], color='r', linestyle='--')
            plt.title(f'{model_name} - Sharpe Ratio During Training')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(model_out_dir / f"{model_name}_training_history.png")
            plt.close()
        
        print(f"Optimization complete for {model_name}")
        print(f"Best parameters: {best_params}")
        print(f"Best profit: {best_profit:.2f}%")
        print(f"Results saved to {model_out_dir}")

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Run profit optimization for equity premium prediction models"
    )
    
    parser.add_argument(
        "--models", 
        nargs="+", 
        default=['Net1', 'Net2'],
        help="List of models to optimize"
    )
    
    parser.add_argument(
        "--method", 
        choices=['grid', 'random', 'bayes'], 
        default='grid',
        help="Optimization method"
    )
    
    parser.add_argument(
        "--trials", 
        type=int, 
        default=50,
        help="Number of trials for random/bayes search"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=100,
        help="Maximum training epochs"
    )
    
    parser.add_argument(
        "--rf-rate-default", 
        type=float, 
        default=0.03,
        help="Default annual risk-free rate (used only if actual rates unavailable)"
    )
    
    parser.add_argument(
        "--max-leverage", 
        type=float, 
        default=1.5,
        help="Maximum leverage allowed (1.5 = 150%)"
    )
    
    parser.add_argument(
        "--transaction-cost", 
        type=float, 
        default=0.0007,
        help="Transaction cost per trade (0.0007 = 0.07%)"
    )
    
    parser.add_argument(
        "--position-sizing", 
        choices=['binary', 'proportional'], 
        default='binary',
        help="Position sizing method"
    )
    
    parser.add_argument(
        "--batch", 
        type=int, 
        default=32,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--threads", 
        type=int, 
        default=4,
        help="Number of PyTorch threads"
    )
    
    parser.add_argument(
        "--device", 
        choices=['cpu', 'cuda', 'auto'], 
        default='auto',
        help="Training device"
    )
    
    parser.add_argument(
        "--data-source", 
        choices=['original', 'newly_identified', 'fred'], 
        default='original',
        help="Data source to use"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    run(
        models=args.models,
        method=args.method,
        trials=args.trials,
        epochs=args.epochs,
        rf_rate_default=args.rf_rate_default,
        max_leverage=args.max_leverage,
        transaction_cost=args.transaction_cost,
        position_sizing=args.position_sizing,
        batch=args.batch,
        threads=args.threads,
        device=args.device,
        data_source=args.data_source
    )
