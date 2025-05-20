"""
profit_oos_10.py

This script performs out-of-sample evaluation of models that were optimized for profit.
It loads the best parameters found during profit optimization and evaluates the models 
on out-of-sample data, calculating profit metrics with realistic trading constraints:
- Long positions (up to 150% leverage) when predictions are positive
- Investment in risk-free assets when predictions are negative
- Realistic institutional trading costs
- Position sizing based on prediction strength

The script provides comprehensive performance metrics and visualizations for economic
interpretation of model results.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm
import time
import copy
import random
import joblib
import glob
import warnings

# Project imports
from src.models import nns
from src.utils.io import RF_ALL, X_ALL, Y_ALL
from src.experiments.profit_optimization_10 import calculate_profit
from src.utils.metrics_unified import compute_sharpe, compute_sortino, compute_max_drawdown

# Suppress warnings
warnings.filterwarnings('ignore')

def find_best_params(model_name, method='grid', data_source='original'):
    """
    Find the best parameters for a model from profit optimization results.
    
    Args:
        model_name (str): Name of the model
        method (str): Optimization method used
        data_source (str): Data source used
        
    Returns:
        dict or None: Best parameters if found, None otherwise
    """
    # Determine the appropriate directory prefix based on data source
    results_dir = Path("./runs/10_Profit_Optimization")
    
    if not results_dir.exists():
        print(f"Warning: Results directory {results_dir} does not exist")
        return None
    
    # Find most recent result that matches criteria
    result_dirs = sorted(
        glob.glob(f"{results_dir}/*_{model_name}_{method}_{data_source}"), 
        key=os.path.getmtime, 
        reverse=True
    )
    
    if not result_dirs:
        print(f"Warning: No results found for {model_name} with method {method} and data source {data_source}")
        return None
    
    # Load results from the first (most recent) matching directory
    result_file = Path(result_dirs[0]) / f"{model_name}_results.joblib"
    
    if not result_file.exists():
        print(f"Warning: Results file {result_file} does not exist")
        return None
    
    try:
        results = joblib.load(result_file)
        return results['best_params']
    except Exception as e:
        print(f"Error loading results from {result_file}: {e}")
        return None

def evaluate_model_oos(model, X_oos, y_oos, rf_rates=None, rf_rate_default=0.03, max_leverage=1.5, 
                      transaction_cost=0.0007, position_sizing='binary', dates=None):
    """
    Evaluate a model on out-of-sample data using profit metrics.
    
    Args:
        model (nn.Module): PyTorch model to evaluate
        X_oos (ndarray): Out-of-sample features
        y_oos (ndarray): Out-of-sample targets
        rf_rate (float): Annual risk-free rate
        max_leverage (float): Maximum leverage allowed
        transaction_cost (float): Transaction cost per trade
        position_sizing (str): Position sizing method
        dates (ndarray, optional): Dates for the out-of-sample period
        
    Returns:
        dict: Evaluation results including profit metrics and predictions
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Convert data to PyTorch tensors
    device = next(model.parameters()).device
    X_oos_tensor = torch.tensor(X_oos, dtype=torch.float32).to(device)
    
    # Generate predictions
    with torch.no_grad():
        predictions = model(X_oos_tensor).cpu().numpy().flatten()
    
    # Calculate profit metrics
    final_capital, profit_pct, sharpe, trades, total_tc = calculate_profit(
        predictions, y_oos.flatten(),
        rf_rates=rf_rates, rf_rate_default=rf_rate_default,
        max_leverage=max_leverage, transaction_cost=transaction_cost,
        position_sizing=position_sizing
    )
    
    # Calculate additional metrics
    # Use pre-calculated equity premium if available, otherwise convert
    if np.mean(np.abs(y_oos)) < 0.1:  # Likely log returns
        # This appears to be log returns, convert to simple returns
        actual_returns = np.exp(y_oos.flatten()) - 1
    else:
        # Already simple returns or pre-calculated equity premium
        actual_returns = y_oos.flatten()
    
    # Calculate position series
    positions = np.where(predictions > 0, 1, 0)
    leverage_vals = np.zeros_like(predictions)
    
    if position_sizing == 'binary':
        leverage_vals = np.where(predictions > 0, max_leverage, 0)
    elif position_sizing == 'proportional':
        pos_preds = predictions.copy()
        if len(pos_preds) > 0:
            pos_max = max(np.max(pos_preds), 0.001)
            pos_preds = np.clip(pos_preds, 0, None)
            leverage_vals = pos_preds / pos_max * max_leverage
    
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
    
    # Model strategy returns
    strategy_returns = np.zeros_like(positions, dtype=float)
    strategy_returns[positions == 1] = actual_returns[positions == 1] * leverage_vals[positions == 1]
    for i in range(len(strategy_returns)):
        if positions[i] == 0:  # Risk-free investment
            strategy_returns[i] = rf_rates_monthly[i] if i < len(rf_rates_monthly) else (1 + rf_rate_default) ** (1/12) - 1
    
    # Calculate transaction costs and apply them
    position_changes = np.diff(np.concatenate([[0], positions]))
    transaction_costs = np.zeros_like(strategy_returns)
    transaction_costs[position_changes != 0] = transaction_cost
    
    # Apply transaction costs to returns
    strategy_returns_net = strategy_returns - transaction_costs
    
    # Calculate equity curves (cumulative returns)
    equity_curve = np.cumprod(1 + strategy_returns_net)
    buy_hold_curve = np.cumprod(1 + actual_returns)
    
    # Calculate metrics
    sortino = compute_sortino(strategy_returns_net, 0)
    max_drawdown = compute_max_drawdown(equity_curve)
    
    # Create results dictionary
    results = {
        'predictions': predictions,
        'actual': y_oos.flatten(),
        'positions': positions,
        'leverage': leverage_vals,
        'strategy_returns': strategy_returns_net,
        'equity_curve': equity_curve,
        'buy_hold_curve': buy_hold_curve,
        'metrics': {
            'final_capital': final_capital,
            'profit_pct': profit_pct,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': max_drawdown,
            'trades': trades,
            'total_transaction_costs': total_tc,
            'average_transaction_cost': total_tc / trades if trades > 0 else 0,
            'trade_frequency': trades / len(y_oos) if len(y_oos) > 0 else 0
        }
    }
    
    # Add dates if provided
    if dates is not None:
        results['dates'] = dates
    
    return results

def plot_results(results, model_name, out_dir, show_plots=False):
    """
    Generate plots for out-of-sample evaluation results.
    
    Args:
        results (dict): Evaluation results
        model_name (str): Name of the model
        out_dir (Path): Output directory
        show_plots (bool): Whether to display plots
    """
    # Extract data
    predictions = results['predictions']
    actual = results['actual']
    positions = results['positions']
    equity_curve = results['equity_curve']
    buy_hold_curve = results['buy_hold_curve']
    metrics = results['metrics']
    leverage = results['leverage']
    dates = results.get('dates', np.arange(len(predictions)))
    
    # Convert dates to pandas datetime if they're numeric
    if not isinstance(dates[0], (pd.Timestamp, np.datetime64)):
        try:
            dates = pd.to_datetime(dates, format='%Y%m')
        except:
            pass  # Keep as is if conversion fails
    
    # Create figure directory
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Predictions vs Actual
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label='Actual', color='blue', alpha=0.7)
    plt.plot(dates, predictions, label='Predicted', color='red', linestyle='--')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title(f'{model_name}: Predictions vs Actual')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / f"{model_name}_predictions.png")
    if not show_plots:
        plt.close()
    
    # 2. Equity Curve
    plt.figure(figsize=(12, 6))
    plt.plot(dates, equity_curve, label='Strategy', color='green')
    plt.plot(dates, buy_hold_curve, label='Buy & Hold', color='blue', linestyle='--')
    plt.title(f'{model_name}: Equity Curve (Starting with $1)')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / f"{model_name}_equity_curve.png")
    if not show_plots:
        plt.close()
    
    # 3. Leverage and Positions
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # 3a. Leverage over time
    ax1.plot(dates, leverage, color='purple')
    ax1.set_title(f'{model_name}: Applied Leverage')
    ax1.set_ylabel('Leverage')
    ax1.grid(True, alpha=0.3)
    
    # 3b. Positions and signals
    ax2.plot(dates, predictions, color='red', alpha=0.5, label='Prediction Signal')
    ax2.fill_between(dates, 0, 1, where=positions > 0, color='green', alpha=0.3, label='Long Position')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_title('Trading Positions')
    ax2.set_ylabel('Signal Strength')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / f"{model_name}_positions.png")
    if not show_plots:
        plt.close()
    
    # 4. Performance Metrics Summary
    plt.figure(figsize=(10, 6))
    metrics_to_plot = {
        'Profit (%)': metrics['profit_pct'],
        'Sharpe Ratio': metrics['sharpe'],
        'Sortino Ratio': metrics['sortino'],
        'Max Drawdown (%)': metrics['max_drawdown'] * 100,
    }
    
    bars = plt.bar(
        range(len(metrics_to_plot)), 
        list(metrics_to_plot.values()),
        color=sns.color_palette("viridis", len(metrics_to_plot))
    )
    
    plt.xticks(range(len(metrics_to_plot)), list(metrics_to_plot.keys()), rotation=45)
    plt.title(f'{model_name}: Performance Metrics')
    plt.grid(True, alpha=0.3)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.02,
            f'{height:.2f}',
            ha='center', va='bottom'
        )
    
    plt.tight_layout()
    plt.savefig(fig_dir / f"{model_name}_metrics.png")
    if not show_plots:
        plt.close()
    
    # 5. Return Distribution
    plt.figure(figsize=(10, 6))
    
    strategy_returns = np.diff(np.log(equity_curve))
    buy_hold_returns = np.diff(np.log(buy_hold_curve))
    
    sns.histplot(strategy_returns, kde=True, stat="density", linewidth=0, color="green", alpha=0.6, label="Strategy")
    sns.histplot(buy_hold_returns, kde=True, stat="density", linewidth=0, color="blue", alpha=0.6, label="Buy & Hold")
    
    plt.title(f'{model_name}: Return Distribution')
    plt.xlabel('Log Returns')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / f"{model_name}_return_distribution.png")
    if not show_plots:
        plt.close()

def run(
    models=['Net1', 'Net2'],
    method='grid',
    rf_rate_default=0.03,
    max_leverage=1.5,
    transaction_cost=0.0007,
    position_sizing='binary',
    oos_start_date=200001,
    threads=4,
    device='auto',
    data_source='original',
    show_plots=False
):
    """
    Run out-of-sample evaluation for profit-optimized models.
    
    Args:
        models (list): List of model names to evaluate
        method (str): Optimization method used ('grid', 'random', 'bayes')
        rf_rate (float): Annual risk-free rate
        max_leverage (float): Maximum leverage allowed (1.5 = 150%)
        transaction_cost (float): Transaction cost per trade
        position_sizing (str): Position sizing method ('binary', 'proportional')
        oos_start_date (int): Start date for out-of-sample period (YYYYMM)
        threads (int): Number of PyTorch threads
        device (str): Evaluation device
        data_source (str): Data source to use ('original', 'newly_identified', 'fred')
        show_plots (bool): Whether to display plots
    """
    # Set device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    # Set PyTorch threads
    torch.set_num_threads(threads)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Create timestamp for output
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    # Load data based on data source
    if data_source == 'newly_identified':
        from src.experiments.newly_identified_6 import load_data
        X_data, y_data, _ = load_data('standalone')
    elif data_source == 'fred':
        from src.experiments.fred_variables_7 import load_fred_data
        X_data, y_data, _ = load_fred_data()
    else:  # Original data
        X_data = X_ALL
        y_data = Y_ALL
    
    print(f"Data loaded: X shape: {X_data.shape}, y shape: {y_data.shape}")
    
    # Load dates, risk-free rates, and equity premium (if available)
    dates = None
    rf_rates = None
    equity_premium = None
    if data_source == 'original':
        try:
            # Try to load additional data from the original dataset
            data_path = Path("ml_equity_premium_data.xlsx")
            if data_path.exists():
                df = pd.read_excel(data_path, sheet_name='result_predictor')
                dates = df['month'].values
                
                # Risk-free rate (TBL column)
                if 'TBL' in df.columns:
                    rf_rates = df['TBL'].values / 100  # Convert percentage to decimal
                    print(f"Loaded risk-free rate data: {len(rf_rates)} values")
                else:
                    print("Warning: TBL column not found in data file")
                
                # Pre-calculated equity premium
                if 'equity_premium' in df.columns:
                    equity_premium = df['equity_premium'].values / 100  # Convert percentage to decimal
                    print(f"Loaded pre-calculated equity premium data: {len(equity_premium)} values")
                    
                    # If we have equity premium values, use them instead of log_equity_premium
                    if len(equity_premium) == len(y_data):
                        print("Using pre-calculated equity premium values instead of log returns")
                        # Use the equity premium values directly
                        y_data = equity_premium
                else:
                    print("Warning: equity_premium column not found in data file")
        except Exception as e:
            print(f"Warning: Could not load auxiliary data from original data file: {e}")
    
    # Create out-of-sample split based on date
    if dates is not None and oos_start_date is not None:
        # Find index where dates >= oos_start_date
        oos_indices = np.where(dates >= oos_start_date)[0]
        if len(oos_indices) > 0:
            oos_idx = oos_indices[0]
            print(f"OOS period starts at index {oos_idx} (date: {dates[oos_idx]})")
            
            # Split data
            X_train = X_data[:oos_idx]
            y_train = y_data[:oos_idx]
            X_oos = X_data[oos_idx:]
            y_oos = y_data[oos_idx:]
            
            # Split dates and risk-free rates
            if dates is not None:
                dates_oos = dates[oos_idx:]
            else:
                dates_oos = None
                
            if rf_rates is not None:
                rf_rates_oos = rf_rates[oos_idx:]
                print(f"Using actual risk-free rates for OOS evaluation: {len(rf_rates_oos)} values")
            else:
                rf_rates_oos = None
        else:
            print(f"Warning: No dates found >= {oos_start_date}, using last 30% as OOS")
            oos_idx = int(len(X_data) * 0.7)
            X_train = X_data[:oos_idx]
            y_train = y_data[:oos_idx]
            X_oos = X_data[oos_idx:]
            y_oos = y_data[oos_idx:]
            
            if dates is not None:
                dates_oos = dates[oos_idx:]
            else:
                dates_oos = None
                
            if rf_rates is not None:
                rf_rates_oos = rf_rates[oos_idx:]
            else:
                rf_rates_oos = None
    else:
        # No dates available, split based on percentage (70/30)
        oos_idx = int(len(X_data) * 0.7)
        X_train = X_data[:oos_idx]
        y_train = y_data[:oos_idx]
        X_oos = X_data[oos_idx:]
        y_oos = y_data[oos_idx:]
        dates_oos = np.arange(len(y_oos))  # Use indices as dates
        rf_rates_oos = None  # No risk-free rates available
    
    print(f"Data split: Train: {len(X_train)}, OOS: {len(X_oos)}")
    
    # Create output directory
    out_base = Path(f"./runs/10_Profit_OOS")
    out_base.mkdir(parents=True, exist_ok=True)
    
    # Create run directory
    run_dir = out_base / f"{ts}_{oos_start_date}_{method}_{data_source}"
    run_dir.mkdir(exist_ok=True, parents=True)
    
    # Create combined results DataFrame
    combined_results = pd.DataFrame()
    
    # Process each model
    for model_name in models:
        print(f"\n--- Evaluating {model_name} OOS Performance ---")
        
        # Find best parameters from in-sample optimization
        best_params = find_best_params(model_name, method, data_source)
        
        if best_params is None:
            print(f"Warning: No best parameters found for {model_name}, skipping")
            continue
        
        print(f"Found best parameters for {model_name}: {best_params}")
        
        # Create model with best parameters
        model_class = getattr(nns, model_name)
        model = model_class(**best_params).to(device)
        
        # Create output directory for this model
        model_out_dir = run_dir / model_name
        model_out_dir.mkdir(exist_ok=True, parents=True)
        
        # Train model on all training data
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1).to(device)
        
        # Initialize model with best parameters
        model = model_class(**best_params).to(device)
        
        # Train model on full training data
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
        loss_fn = nn.MSELoss()
        
        model.train()
        for epoch in range(200):  # Train for 200 epochs max
            optimizer.zero_grad()
            y_pred = model(X_train_tensor)
            loss = loss_fn(y_pred, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/200, Loss: {loss.item():.8f}")
        
        # Evaluate on out-of-sample data
        results = evaluate_model_oos(
            model, X_oos, y_oos,
            rf_rates=rf_rates_oos,
            rf_rate_default=rf_rate_default,
            max_leverage=max_leverage,
            transaction_cost=transaction_cost,
            position_sizing=position_sizing,
            dates=dates_oos
        )
        
        # Save results
        joblib.dump(results, model_out_dir / f"{model_name}_oos_results.joblib")
        
        # Generate plots
        plot_results(results, model_name, model_out_dir, show_plots)
        
        # Create summary of results
        metrics = results['metrics']
        summary = pd.DataFrame({
            'Model': [model_name],
            'Profit (%)': [metrics['profit_pct']],
            'Sharpe Ratio': [metrics['sharpe']],
            'Sortino Ratio': [metrics['sortino']],
            'Max Drawdown (%)': [metrics['max_drawdown'] * 100],
            'Trades': [metrics['trades']],
            'Trade Frequency': [metrics['trade_frequency']],
            'Transaction Costs': [metrics['total_transaction_costs']]
        })
        
        # Add to combined results
        combined_results = pd.concat([combined_results, summary], ignore_index=True)
        
        print(f"OOS evaluation complete for {model_name}")
        print(f"Profit: {metrics['profit_pct']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe']:.4f}")
        print(f"Results saved to {model_out_dir}")
    
    # Save combined results
    if not combined_results.empty:
        combined_results.to_csv(run_dir / "combined_results.csv", index=False)
        
        # Generate comparative plots
        if len(models) > 1:
            plt.figure(figsize=(12, 8))
            metrics_to_compare = ['Profit (%)', 'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown (%)']
            
            for i, metric in enumerate(metrics_to_compare):
                plt.subplot(2, 2, i+1)
                sns.barplot(x='Model', y=metric, data=combined_results)
                plt.title(metric)
                plt.xticks(rotation=45)
                plt.tight_layout()
            
            plt.savefig(run_dir / "comparative_metrics.png")
            if not show_plots:
                plt.close()
    
    print(f"All models evaluated. Combined results saved to {run_dir}")
    return combined_results

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Run out-of-sample evaluation for profit-optimized models"
    )
    
    parser.add_argument(
        "--models", 
        nargs="+", 
        default=['Net1', 'Net2'],
        help="List of models to evaluate"
    )
    
    parser.add_argument(
        "--method", 
        choices=['grid', 'random', 'bayes'], 
        default='grid',
        help="Optimization method used"
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
        "--oos-start-date", 
        type=int, 
        default=200001,
        help="Start date for out-of-sample period (YYYYMM)"
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
        help="Evaluation device"
    )
    
    parser.add_argument(
        "--data-source", 
        choices=['original', 'newly_identified', 'fred'], 
        default='original',
        help="Data source to use"
    )
    
    parser.add_argument(
        "--show-plots", 
        action="store_true",
        help="Display plots in addition to saving them"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    run(
        models=args.models,
        method=args.method,
        rf_rate_default=args.rf_rate_default,
        max_leverage=args.max_leverage,
        transaction_cost=args.transaction_cost,
        position_sizing=args.position_sizing,
        oos_start_date=args.oos_start_date,
        threads=args.threads,
        device=args.device,
        data_source=args.data_source,
        show_plots=args.show_plots
    )
