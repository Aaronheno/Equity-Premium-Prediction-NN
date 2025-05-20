"""
gamma_sensitivity_9.py

This script performs sensitivity analysis on the Certainty Equivalent Return (CER)
metric by evaluating model performance across a range of risk aversion (gamma)
values. This helps understand how robust different prediction models are to
varying investor risk preferences.

The analysis:
1. Takes existing model predictions from previous runs
2. Calculates CER across a spectrum of gamma values
3. Visualizes sensitivity curves for different models
4. Provides insights on model performance stability under different risk attitudes
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
from matplotlib.ticker import MaxNLocator
from typing import Dict, List, Tuple, Union, Optional
import joblib
import glob
import re
import warnings

# Ensure consistent visualization
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('colorblind')

def calculate_cer(returns, gamma=2.0, rf_rates=None, rf_rate_default=0.03):
    """
    Calculate Certainty Equivalent Return (CER) for a series of returns.
    
    Args:
        returns (np.ndarray): Series of returns
        gamma (float): Risk aversion coefficient
        rf_rates (np.ndarray): Risk-free rates (optional)
        rf_rate_default (float): Default risk-free rate (used if actual rates unavailable)
        
    Returns:
        float: CER value
    """
    # For numerical stability, handle extreme gamma values differently
    if gamma < 0.000001:  # Very close to 0
        return np.mean(returns)
    
    # Standard CER formula: rf + E[r-rf] - (gamma/2) * Var[r]
    if rf_rates is not None:
        excess_returns = returns - rf_rates
    else:
        excess_returns = returns - rf_rate_default
    
    mean_excess_return = np.mean(excess_returns)
    variance = np.var(excess_returns)
    
    cer = rf_rate_default + mean_excess_return - (gamma / 2) * variance
    return cer

def load_predictions(model_name, method, run_dir=None, start_date=None, data_source='original'):
    """
    Load prediction results for the specified model and method.
    
    Args:
        model_name (str): Name of the model (e.g., 'Net1')
        method (str): Method used (e.g., 'bayes_oos')
        run_dir (str, optional): Specific run directory to look in
        start_date (int, optional): OOS start date in YYYYMM format
        data_source (str): Data source ('original', 'newly_identified', or 'fred')
        
    Returns:
        dict: Dictionary with predictions, actuals, model name, method, and run directory
    """
    # Determine run directory
    if run_dir is None:
        # Find latest run for given method
        if method in ['grid', 'random', 'bayes']:
            base_dir = Path(f"./runs/1_OOS_Analysis/{method}")
        else:
            # For methods like mae_grid, find appropraite directory using mapping
            method_parts = method.split('_')
            method_num = None
            if len(method_parts) > 1 and method_parts[-1].isdigit():
                method_num = method_parts[-1]
            elif any(m for m in method_parts if m in ['grid', 'random', 'bayes']):
                # Assume format like mae_grid_5
                base_method = next(m for m in method_parts if m in ['grid', 'random', 'bayes'])
                if base_method in ['grid', 'random', 'bayes'] and method_parts[-1].isdigit():
                    method_num = method_parts[-1]
                    method = base_method
            
            if method_num:
                base_dir = Path(f"./runs/{method_num}_OOS_Analysis/{method}")
            else:
                base_dir = Path(f"./runs/1_OOS_Analysis/{method}")
        
        if not base_dir.exists():
            print(f"Run directory {base_dir} not found")
            return None
        
        # Find most recent run for this model
        run_dirs = sorted([d for d in base_dir.glob(f"*{model_name}*") if d.is_dir()], reverse=True)
        if not run_dirs:
            print(f"No run directories found for {model_name} with method {method}")
            return None
        
        run_dir = run_dirs[0]
    else:
        run_dir = Path(run_dir)
        if not run_dir.exists():
            print(f"Specified run directory {run_dir} not found")
            return None
    
    # Load predictions and actual values from results file
    result_file = None
    for file in run_dir.glob(f"*{model_name}*results*.joblib"):
        result_file = file
        break
    
    if result_file is None:
        print(f"No results file found in {run_dir}")
        return None
    
    try:
        results = joblib.load(result_file)
        if 'oos_predictions' in results and 'oos_actuals' in results:
            predictions = results['oos_predictions']
            actuals = results['oos_actuals']
        elif 'predictions' in results and 'actuals' in results:
            predictions = results['predictions']
            actuals = results['actuals']
        else:
            print(f"Required keys not found in results file: {result_file}")
            return None
            
        # Try to get risk-free rates if available
        rf_rates = None
        if 'rf_rates' in results:
            rf_rates = results['rf_rates']
        
        # Filter by start_date if provided
        dates = None
        if 'dates' in results:
            dates = results['dates']
            
        if start_date is not None and dates is not None:
            mask = dates >= start_date
            predictions = predictions[mask]
            actuals = actuals[mask]
            dates = dates[mask]
            if rf_rates is not None and len(rf_rates) == len(predictions):
                rf_rates = rf_rates[mask]
        elif start_date is not None and dates is None:
            print(f"No dates found in results, cannot filter by start_date")
        
        # Return as dictionary
        result_dict = {
            'predictions': predictions,
            'actuals': actuals,
            'model_name': model_name,
            'method': method,
            'run_dir': str(run_dir)
        }
        
        # Add rf_rates if available
        if rf_rates is not None:
            result_dict['rf_rates'] = rf_rates
            
        # Add dates if available
        if dates is not None:
            result_dict['dates'] = dates
            
        return result_dict
    except Exception as e:
        print(f"Error loading results from {result_file}: {e}")
        return None

def calculate_returns(predictions, actuals, rf_rate=0):
    """
    Calculate realized returns from predictions and actual values.
    
    This function converts log equity premium predictions into trading returns
    by determining when to invest in equities vs risk-free assets.
    
    Args:
        predictions (np.ndarray): Predicted log equity premium
        actuals (np.ndarray): Actual log equity premium
        rf_rate (float): Risk-free rate (annual)
        
    Returns:
        tuple: (market_timing_returns, buy_hold_returns)
    """
    # Convert from log equity premium to simple returns
    actual_returns = np.exp(actuals) - 1
    
    # Calculate market timing returns (invest in equity when prediction > 0)
    # Otherwise invest in risk-free asset
    market_timing_signals = predictions > 0
    market_timing_returns = np.zeros_like(actual_returns)
    
    # When signal is positive, invest in equity
    market_timing_returns[market_timing_signals] = actual_returns[market_timing_signals]
    
    # When signal is negative, invest in risk-free asset
    # Monthly risk-free rate
    monthly_rf = (1 + rf_rate) ** (1/12) - 1
    market_timing_returns[~market_timing_signals] = monthly_rf
    
    # Buy-and-hold strategy (always invest in equity)
    buy_hold_returns = actual_returns
    
    return market_timing_returns, buy_hold_returns

def run_gamma_sensitivity_analysis(
    models,
    methods,
    gamma_values=None,
    rf_rate_default=0.03,
    run_dir=None,
    start_date=None,
    data_source='original',
    n_jobs=1
):
    """
    Run gamma sensitivity analysis to evaluate how CER changes with different gamma values.
    
    Args:
        models (List[str]): List of models to analyze
        methods (List[str]): List of methods to use
        gamma_values (List[float], optional): List of gamma values to test
        rf_rate_default (float): Default risk-free rate (used if actual rates unavailable)
        run_dir (str, optional): Specific run directory
        start_date (int, optional): OOS start date in YYYYMM format
        data_source (str): Data source ('original', 'newly_identified', or 'fred')
        n_jobs (int): Number of parallel jobs
        
    Returns:
        tuple: (cer_results, benchmark_cer, actual_values_dict, dates_dict)
    """
    # Default gamma values if not specified
    if gamma_values is None:
        gamma_values = np.concatenate([
            np.linspace(1, 5, 12),    # Low to moderate risk aversion (more resolution)
            np.linspace(5, 12, 8)     # Moderate to high risk aversion
        ])
    
    # Set up timestamp for output
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    # Set up output directory
    out_base = Path(f"./runs/9_Gamma_Sensitivity")
    out_base.mkdir(parents=True, exist_ok=True)
    
    # Create output directory name based on parameters
    method_str = "_".join(methods)
    model_str = "_".join(models)
    run_name = f"{ts}_{data_source}_{method_str}_{model_str}"
    out_dir = out_base / run_name
    out_dir.mkdir(exist_ok=True, parents=True)
    print(f"Output directory: {out_dir}")
    
    # Dictionary to store results
    cer_results = {}
    actual_values_dict = {}
    dates_dict = {}
    benchmark_cer = {}
    
    # Calculate CER for each model across gamma values
    for method in methods:
        cer_results[method] = {}
        actual_values_dict[method] = {}
        dates_dict[method] = {}
        benchmark_cer[method] = {}
        
        for model in models:
            # Load predictions
            try:
                pred_dict = load_predictions(model, method, run_dir, start_date, data_source)
                if pred_dict is None:
                    continue
                
                actuals = pred_dict['actuals']
                predictions = pred_dict['predictions']
                rf_rates = pred_dict.get('rf_rates', None)
                
                # Store actual values and dates for later reference
                actual_values_dict[method][model] = actuals
                if 'dates' in pred_dict:
                    dates_dict[method][model] = pred_dict['dates']
                
                # Calculate returns
                model_returns, buy_hold_returns = calculate_returns(predictions, actuals)
                
                # Calculate CER for each gamma value
                model_cer_values = []
                buy_hold_cer_values = []
                
                for gamma in gamma_values:
                    model_cer = calculate_cer(model_returns, gamma, rf_rates, rf_rate_default)
                    buy_hold_cer = calculate_cer(buy_hold_returns, gamma, rf_rates, rf_rate_default)
                    
                    model_cer_values.append(model_cer)
                    buy_hold_cer_values.append(buy_hold_cer)
                
                # Store results
                cer_results[method][model] = model_cer_values
                benchmark_cer[method][model] = buy_hold_cer_values
                
            except Exception as e:
                print(f"Error loading predictions for {model} with {method}: {e}")
                continue
    
    # Save results to CSV
    results_df = pd.DataFrame(index=gamma_values)
    benchmark_df = pd.DataFrame(index=gamma_values)
    
    for method in methods:
        for model in models:
            if model in cer_results[method]:
                col_name = f"{method}_{model}"
                results_df[col_name] = cer_results[method][model]
                benchmark_df[col_name] = benchmark_cer[method][model]
    
    results_df.to_csv(out_dir / "gamma_sensitivity_cer.csv")
    benchmark_df.to_csv(out_dir / "gamma_sensitivity_benchmark.csv")
    
    # Generate plots
    generate_sensitivity_plots(gamma_values, cer_results, benchmark_cer, out_dir, methods, models)
    generate_sensitivity_heatmap(gamma_values, cer_results, benchmark_cer, out_dir, methods, models)
    
    return cer_results, benchmark_cer, actual_values_dict, dates_dict

def generate_sensitivity_plots(gamma_values, cer_results, benchmark_cer, out_dir, methods, models):
    """
    Generate plots showing how CER changes with gamma for each model.
    
    Args:
        gamma_values (list): List of gamma values tested
        cer_results (dict): Dictionary with CER results for each model
        benchmark_cer (dict): Dictionary with benchmark CER results
        out_dir (Path): Output directory
        methods (list): List of methods used
        models (list): List of models analyzed
    """
    # Prepare a grid of plots, one for each method
    n_methods = len(methods)
    fig, axes = plt.subplots(n_methods, 1, figsize=(10, 5 * n_methods), sharex=True)
    
    # If there's only one method, wrap axes in a list for consistent indexing
    if n_methods == 1:
        axes = [axes]
    
    for i, method in enumerate(methods):
        ax = axes[i]
        
        # Plot CER for each model
        for model in models:
            if model in cer_results[method]:
                ax.plot(gamma_values, cer_results[method][model], label=f"{model}", linewidth=2)
        
        # Plot benchmark (buy-and-hold) for reference
        for model in models:
            if model in benchmark_cer[method]:
                ax.plot(gamma_values, benchmark_cer[method][model], label=f"Buy & Hold", 
                        linestyle='--', color='black', alpha=0.7)
                # Only need to plot benchmark once
                break
        
        ax.set_title(f"CER Sensitivity to Gamma - {method}", fontsize=14)
        ax.set_ylabel("CER", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
    
    # Set common x-axis label
    axes[-1].set_xlabel("Risk Aversion (γ)", fontsize=12)
    axes[-1].set_xscale('log')
    
    # Add a footnote about interpretation
    plt.figtext(0.5, 0.01, 
                "Higher CER values indicate better risk-adjusted performance.\n"
                "Steeper CER slope indicates higher sensitivity to risk aversion.", 
                ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(out_dir / "gamma_sensitivity_curves.png", dpi=300)
    plt.close()
    
    # Generate individual plots for each method
    for method in methods:
        plt.figure(figsize=(10, 6))
        
        # Plot CER for each model
        for model in models:
            if model in cer_results[method]:
                plt.plot(gamma_values, cer_results[method][model], label=f"{model}", linewidth=2)
        
        # Plot benchmark (buy-and-hold) for reference
        for model in models:
            if model in benchmark_cer[method]:
                plt.plot(gamma_values, benchmark_cer[method][model], label=f"Buy & Hold", 
                       linestyle='--', color='black', alpha=0.7)
                # Only need to plot benchmark once
                break
        
        plt.title(f"CER Sensitivity to Gamma - {method}", fontsize=14)
        plt.ylabel("CER", fontsize=12)
        plt.xlabel("Risk Aversion (γ)", fontsize=12)
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        
        # Add a footnote about interpretation
        plt.figtext(0.5, 0.01, 
                  "Higher CER values indicate better risk-adjusted performance.\n"
                  "Steeper CER slope indicates higher sensitivity to risk aversion.", 
                  ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        plt.savefig(out_dir / f"gamma_sensitivity_{method}.png", dpi=300)
        plt.close()

def generate_sensitivity_heatmap(gamma_values, cer_results, benchmark_cer, out_dir, methods, models):
    """
    Generate heatmaps showing CER advantage over benchmark across gamma values.
    
    Args:
        gamma_values (list): List of gamma values tested
        cer_results (dict): Dictionary with CER results for each model
        benchmark_cer (dict): Dictionary with benchmark CER results
        out_dir (Path): Output directory
        methods (list): List of methods used
        models (list): List of models analyzed
    """
    for method in methods:
        # Calculate CER advantage (model CER - benchmark CER)
        cer_advantage = {}
        
        for model in models:
            if model in cer_results[method] and model in benchmark_cer[method]:
                cer_advantage[model] = np.array(cer_results[method][model]) - np.array(benchmark_cer[method][model])
        
        if not cer_advantage:
            continue
        
        # Create DataFrame for heatmap
        heatmap_data = pd.DataFrame(cer_advantage, index=gamma_values)
        
        # Generate heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data.T, cmap='RdBu_r', center=0, 
                   cbar_kws={'label': 'CER Advantage over Buy & Hold'})
        
        plt.title(f"CER Advantage by Model and Risk Aversion - {method}", fontsize=14)
        plt.ylabel("Model", fontsize=12)
        plt.xlabel("Risk Aversion (γ)", fontsize=12)
        
        plt.tight_layout()
        plt.savefig(out_dir / f"gamma_sensitivity_heatmap_{method}.png", dpi=300)
        plt.close()
        
        # Save data
        heatmap_data.to_csv(out_dir / f"cer_advantage_{method}.csv")
        
        # Generate crossover point analysis
        plt.figure(figsize=(10, 6))
        
        # For each model, find where its CER crosses the benchmark
        for model in cer_advantage:
            # Find points where advantage changes sign (crossover points)
            crossovers = []
            for i in range(len(gamma_values) - 1):
                if (cer_advantage[model][i] >= 0 and cer_advantage[model][i+1] < 0) or \
                   (cer_advantage[model][i] <= 0 and cer_advantage[model][i+1] > 0):
                    # Linear interpolation to find more precise crossover point
                    x1, x2 = gamma_values[i], gamma_values[i+1]
                    y1, y2 = cer_advantage[model][i], cer_advantage[model][i+1]
                    crossover = x1 - y1 * (x2 - x1) / (y2 - y1)
                    crossovers.append(crossover)
            
            # Plot the crossover points
            for crossover in crossovers:
                plt.axvline(x=crossover, color='gray', linestyle=':', alpha=0.7)
                plt.text(crossover, plt.ylim()[1] * 0.9, f"{model}: γ={crossover:.2f}", 
                       rotation=90, verticalalignment='top')
        
        # Plot the CER advantage curves
        for model in cer_advantage:
            plt.plot(gamma_values, cer_advantage[model], label=model)
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.title(f"CER Advantage Curves - {method}", fontsize=14)
        plt.ylabel("CER Advantage", fontsize=12)
        plt.xlabel("Risk Aversion (γ)", fontsize=12)
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        
        plt.tight_layout()
        plt.savefig(out_dir / f"cer_advantage_curves_{method}.png", dpi=300)
        plt.close()

def run(
    models=['Net1', 'Net2', 'Net3', 'Net4'],
    methods=['grid', 'random', 'bayes'],
    gamma_range=None,
    rf_rate_default=0.03,
    run_dir=None,
    start_date=None,
    data_source='original',
    max_leverage=1.5,
    transaction_cost=0.0007,
    position_sizing='binary',
    show_plots=False
):
    """
    Main function to run gamma sensitivity analysis.
    
    Args:
        models (List[str]): List of models to analyze
        methods (List[str]): List of methods to analyze
        gamma_range (str, optional): Custom range of gamma values in format "start,end,num_points"
        rf_rate_default (float): Default annual risk-free rate (used only if actual rates unavailable)
        run_dir (str, optional): Specific run directory
        start_date (int, optional): OOS start date in YYYYMM format
        data_source (str): Data source ('original', 'newly_identified', or 'fred')
        max_leverage (float): Maximum leverage allowed
        transaction_cost (float): Transaction cost as a fraction of trade value
        position_sizing (str): Position sizing strategy ('binary' or 'continuous')
        show_plots (bool): Whether to display plots
    """
    # Parse custom gamma range if provided
    if gamma_range:
        try:
            parts = gamma_range.split(',')
            start, end, num = float(parts[0]), float(parts[1]), int(parts[2])
            gamma_values = np.linspace(start, end, num)
        except Exception as e:
            print(f"Error parsing gamma range: {e}")
            print("Using default gamma values")
            gamma_values = None
    else:
        gamma_values = None
    
    # Run the analysis
    cer_results, benchmark_cer, actual_values_dict, dates_dict = run_gamma_sensitivity_analysis(
        models=models,
        methods=methods,
        gamma_values=gamma_values,
        rf_rate_default=rf_rate_default,
        run_dir=run_dir,
        start_date=start_date,
        data_source=data_source,
        n_jobs=1
    )
    
    print("Gamma sensitivity analysis completed.")
    
    return cer_results, benchmark_cer

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Run gamma sensitivity analysis for equity premium prediction models")
    
    parser.add_argument("--models", nargs='+', default=['Net1', 'Net2', 'Net3', 'Net4'],
                      help="Models to analyze (e.g., Net1 Net2 Net3)")
    
    parser.add_argument("--methods", nargs='+', default=['grid', 'random', 'bayes'],
                      help="Methods to analyze (e.g., bayes_oos random_oos)")
    
    parser.add_argument("--gamma-range", type=str, default=None,
                      help="Custom range of gamma values in format 'start,end,num_points' (e.g., '0.5,10,20')")
    
    parser.add_argument(
        "--rf-rate-default", 
        type=float, 
        default=0.03,
        help="Default annual risk-free rate (used only if actual rates unavailable)"
    )
    
    parser.add_argument("--run-dir", type=str, default=None,
                      help="Specific run directory to analyze")
    
    parser.add_argument("--start-date", type=int, default=None,
                      help="OOS start date in YYYYMM format (e.g., 200001)")
    
    parser.add_argument("--data-source", type=str, default='original',
                      choices=['original', 'newly_identified', 'fred'],
                      help="Data source to use")
    
    parser.add_argument("--max-leverage", type=float, default=1.5,
                      help="Maximum leverage allowed")
    
    parser.add_argument("--transaction-cost", type=float, default=0.0007,
                      help="Transaction cost as a fraction of trade value")
    
    parser.add_argument("--position-sizing", type=str, default='binary',
                      choices=['binary', 'continuous'],
                      help="Position sizing strategy")
    
    parser.add_argument("--show-plots", action='store_true',
                      help="Whether to display plots")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    run(
        models=args.models,
        methods=args.methods,
        gamma_range=args.gamma_range,
        rf_rate_default=args.rf_rate_default,
        run_dir=args.run_dir,
        start_date=args.start_date,
        data_source=args.data_source,
        max_leverage=args.max_leverage,
        transaction_cost=args.transaction_cost,
        position_sizing=args.position_sizing,
        show_plots=args.show_plots
    )
