"""
Economic value analysis for equity premium prediction models.

This module provides tools to evaluate the economic value of forecasting models
through market timing performance metrics like average returns and Sharpe ratios.
It also includes functionality to convert OOS results to the format needed for analysis.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re
import sys
from datetime import datetime

# =====================================================================
# OOS Results Conversion Functions
# =====================================================================

def extract_method_from_path(path_str):
    """Extract HPO method (grid, random, bayes) from directory path using the numeric naming scheme."""
    path = Path(path_str)
    
    # Check the parent folder of the timestamp folder
    # For example, from 'runs/1_Grid_Search_OOS/20250517_021419_grid_search_oos/oos_all_predictions...'
    # we need to extract the '1_Grid_Search_OOS' part
    parent_dir = path.parent.parent.name.lower()  # This gets the stage folder (1_Grid_Search_OOS)
    
    # Extract method based on the parent directory name (the stage folder)
    if 'grid' in parent_dir:
        return 'GS'
    elif 'random' in parent_dir:
        return 'RS'
    elif 'bayes' in parent_dir:
        return 'BO'
    else:
        # If we can't determine from parent, try the timestamp folder itself
        timestamp_dir = path.parent.name.lower()  # Timestamp folder like '20250517_021419_grid_search_oos'
        if 'grid' in timestamp_dir:
            return 'GS'
        elif 'random' in timestamp_dir:
            return 'RS'
        elif 'bayes' in timestamp_dir:
            return 'BO'
        else:
            return 'Unknown'

def convert_oos_results_to_economic_format(oos_results_dir, output_file=None, verbose=True):
    """
    Convert OOS results to format needed for economic value analysis
    
    Parameters:
    -----------
    oos_results_dir : str or Path
        Directory containing OOS results (csv files)
    output_file : str or Path, optional
        Path to save the combined predictions
    verbose : bool, default=True
        Whether to print progress information
        
    Returns:
    --------
    combined_df : pd.DataFrame
        DataFrame with combined predictions in format needed for economic analysis
    """
    # Find all prediction files
    oos_results_dir = Path(oos_results_dir)
    # Use ** glob pattern to search recursively through all subdirectories
    prediction_files = list(oos_results_dir.glob("**/oos_all_predictions_raw_with_actuals.csv"))
    
    if verbose:
        print(f"Found {len(prediction_files)} OOS prediction files")
    
    if not prediction_files:
        print(f"Error: No prediction files found in {oos_results_dir}")
        return pd.DataFrame()
    
    all_data = {}
    date_index = None
    
    for file_path in prediction_files:
        # Extract method from path
        method = extract_method_from_path(file_path)
        
        if verbose:
            print(f"Processing {file_path} (Method: {method})")
        
        # Read the file
        try:
            # Load the data with more explicit error handling
            df = pd.read_csv(file_path)
            
            if verbose:
                print(f"  Columns in file: {df.columns.tolist()}")
            
            # Convert date column to datetime and set as index
            # First check for explicit 'Date' column
            if 'Date' in df.columns:
                date_col = 'Date'
            # Then check if 'Unnamed: 0' could be a date column (common in CSV exports)
            elif 'Unnamed: 0' in df.columns:
                # Check if Unnamed: 0 contains numeric dates in YYYYmm format
                try:
                    # Check the first few values to see if they're integers in YYYYmm format
                    values = df['Unnamed: 0'].head()
                    if all(isinstance(x, (int, np.int64)) for x in values) and all(len(str(x)) == 6 for x in values):
                        # These appear to be YYYYmm format dates
                        date_col = 'Unnamed: 0'
                        if verbose:
                            print(f"  Using 'Unnamed: 0' as date column (YYYYmm format)")
                    else:
                        # Try standard date parsing
                        test_dates = pd.to_datetime(df['Unnamed: 0'])
                        date_col = 'Unnamed: 0'
                        if verbose:
                            print(f"  Using 'Unnamed: 0' as date column (standard format)")
                except Exception as e:
                    date_col = None
                    if verbose:
                        print(f"  WARNING: 'Unnamed: 0' is not a valid date column: {e}")
            else:
                date_col = None
            
            # Set index if we found a date column
            if date_col:
                try:
                    # Check if the values look like YYYYmm format integers
                    if all(isinstance(x, (int, np.int64)) for x in df[date_col].head()):
                        # Keep as integers for YYYYmm format consistency with other data
                        # Just set as index without converting to datetime
                        df.set_index(date_col, inplace=True)
                        if verbose:
                            print(f"  Using YYYYmm integers as index: {df.index.min()} to {df.index.max()}, length: {len(df.index)}")
                    else:
                        # Standard datetime parsing for other date formats
                        df[date_col] = pd.to_datetime(df[date_col])
                        df.set_index(date_col, inplace=True)
                        if verbose:
                            print(f"  Date range: {df.index.min()} to {df.index.max()}, length: {len(df.index)}")
                    
                    # Store date index for later use
                    if date_index is None:
                        date_index = df.index
                except Exception as e:
                    if verbose:
                        print(f"  WARNING: Failed to set {date_col} as index: {e}")
            elif verbose:
                print(f"  WARNING: No date column found in file")
            
            # Extract model predictions and rename columns
            for col in df.columns:
                # Skip columns we don't need
                if not col.startswith('Predicted_') or col.endswith('_CF') or col.endswith('_HA'):
                    continue
                
                # Extract model name from column
                model_match = re.match(r'Predicted_([A-Za-z0-9]+)(?:_|$)', col)
                if model_match:
                    model_name = model_match.group(1)
                    new_col_name = f"{model_name}_{method}"
                    
                    # Store in all_data dict
                    all_data[new_col_name] = df[col].values
                    
            # Add HA (historical average) predictions
            ha_cols = [c for c in df.columns if c.endswith('_HA')]
            if ha_cols and 'HA' not in all_data:
                all_data['HA'] = df[ha_cols[0]].values
                
            # Add actual values
            actual_cols = [c for c in df.columns if 'Actual' in c]
            if actual_cols and 'Actual' not in all_data:
                all_data['Actual'] = df[actual_cols[0]].values
                if verbose:
                    print(f"  Found actual values column: {actual_cols[0]}")
            elif verbose and not actual_cols:
                print(f"  WARNING: No actual value columns found in this file")
                print(f"  Available columns: {df.columns.tolist()}")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc(file=sys.stderr)
    
    # Combine all results into a DataFrame
    if all_data and date_index is not None:
        # Print debug information about collected data
        if verbose:
            print("\nData collection summary:")
            print(f"Total models found: {len(all_data) - ('Actual' in all_data)}")
            print(f"Actual values found: {'Actual' in all_data}")
            for k, v in all_data.items():
                print(f"  {k}: length={len(v)}, non-null={np.sum(~np.isnan(v))}, range=[{np.nanmin(v):.4f}, {np.nanmax(v):.4f}]")
        
        # Handle possible length mismatch
        min_len = min(len(v) for v in all_data.values())
        if min_len < len(date_index):
            if verbose:
                print(f"WARNING: Truncating date_index from {len(date_index)} to {min_len} to match data length")
            date_index = date_index[:min_len]
        
        # Construct DataFrame with explicit error handling
        try:
            combined_df = pd.DataFrame({k: v[:min_len] for k, v in all_data.items()}, index=date_index)
            
            if verbose:
                print(f"\nSuccessfully created combined DataFrame with shape {combined_df.shape}")
                print(f"Columns: {combined_df.columns.tolist()}")
                print(f"Index type: {type(combined_df.index[0])}")
            
            # If the index contains YYYYmm integers, preserve that format when saving to CSV
            # by resetting the index to a standard column
            if output_file:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Reset index to ensure the YYYYmm format is preserved
                output_df = combined_df.reset_index()
                if verbose:
                    print(f"Saving with index column preserved as: {output_df.columns[0]}")
                output_df.to_csv(output_file, index=False)
                if verbose:
                    print(f"Saved combined predictions to {output_file}")
            
            return combined_df
        except Exception as e:
            print(f"Error constructing DataFrame: {e}")
            import traceback
            traceback.print_exc(file=sys.stderr)
            return pd.DataFrame()
    else:
        print("Error: Could not construct combined DataFrame")
        if not all_data:
            print("No data was collected from prediction files")
        if date_index is None:
            print("No date index was found in any prediction file")
        return pd.DataFrame()

def find_all_oos_runs(base_dir):
    """
    Find all OOS run directories using the numeric naming scheme
    
    Parameters:
    -----------
    base_dir : str or Path
        Base directory to search for OOS runs
        
    Returns:
    --------
    oos_dirs : list
        List of directories containing OOS runs
    """
    base_dir = Path(base_dir)
    
    # Find all directories that follow the numeric naming convention for OOS results
    oos_dirs = []
    
    # Stage 1: Out-of-sample results with numeric prefix
    oos_dirs.extend(list(base_dir.glob("1_Grid_Search_OOS*")))
    oos_dirs.extend(list(base_dir.glob("1_Random_Search_OOS*")))
    oos_dirs.extend(list(base_dir.glob("1_Bayes_Search_OOS*")))
    
    # Also include MAE-based OOS predictions (stage 5)
    oos_dirs.extend(list(base_dir.glob("5_Grid_MAE_OOS*")))
    oos_dirs.extend(list(base_dir.glob("5_Random_MAE_OOS*")))
    oos_dirs.extend(list(base_dir.glob("5_Bayes_MAE_OOS*")))
    
    # Include window-based analysis results (stages 3-4)
    oos_dirs.extend(list(base_dir.glob("3_Rolling_Window*")))
    oos_dirs.extend(list(base_dir.glob("4_Expanding_Window*")))
    
    # Filter to only include directories that have the OOS predictions file
    valid_oos_dirs = []
    for dir_path in oos_dirs:
        if (dir_path / "oos_all_predictions_raw_with_actuals.csv").exists():
            valid_oos_dirs.append(dir_path)
    
    return valid_oos_dirs

# =====================================================================
# Economic Value Analysis Functions
# =====================================================================

def calculate_market_timing_performance(predictions_df, actual_returns, risk_free_rates, window_sizes=[1, 3]):
    """
    Calculate market timing performance metrics for forecasting models
    
    Parameters:
    -----------
    predictions_df : pd.DataFrame
        DataFrame with date index and columns for different model predictions (log equity premium)
    actual_returns : pd.Series or np.array
        Actual simple returns (not log) aligned with predictions_df
    risk_free_rates : pd.Series or np.array
        Risk-free rates aligned with predictions_df
    window_sizes : list
        List of expanding window sizes (in years) to evaluate
        
    Returns:
    --------
    market_timing_performance : dict
        Dictionary of DataFrames with market timing performance metrics
    """
    results = {}
    
    # Calculate buy-and-hold performance
    buy_hold_avg_return = np.mean(actual_returns) * 12 * 100  # Annualized
    buy_hold_sharpe = np.mean(actual_returns - risk_free_rates) / np.std(actual_returns - risk_free_rates) * np.sqrt(12)  # Annualized
    
    # Dictionary to store results for different window sizes
    window_results = {}
    
    for window_size in window_sizes:
        window_label = f"{window_size} Year Expanding Window"
        window_results[window_label] = {}
        
        # Add buy-and-hold strategy
        window_results[window_label]["BH"] = {
            "Method": "-",
            "Average Return %": buy_hold_avg_return,
            "Sharpe Ratio": buy_hold_sharpe
        }
        
        # Process HA (historical average) if available
        if 'HA' in predictions_df.columns:
            # Convert log equity premium to simple returns for HA
            log_eq_premium_ha = predictions_df['HA'].values
            simple_returns_ha = np.exp(log_eq_premium_ha) * (1 + risk_free_rates) - 1
            
            # Apply market timing for HA
            timing_signal_ha = simple_returns_ha * actual_returns > 0
            return_portfolio_ha = actual_returns * timing_signal_ha
            avg_return_ha = np.mean(return_portfolio_ha) * 12 * 100  # Annualized
            excess_return_ha = return_portfolio_ha - risk_free_rates
            sharpe_ratio_ha = np.mean(excess_return_ha) / np.std(excess_return_ha) * np.sqrt(12)  # Annualized
            
            window_results[window_label]["HA"] = {
                "Method": "-",
                "Average Return %": avg_return_ha,
                "Sharpe Ratio": sharpe_ratio_ha
            }
        
        # Convert log equity premium predictions to simple returns for all models
        for col in predictions_df.columns:
            if col == 'HA' or 'Actual' in col:
                continue
                
            model_name = col.split('_')[0]  # Extract model name
            method = "Unknown"
            
            # Extract method from column name
            if "_GS" in col:
                method = "GS"
            elif "_RS" in col:
                method = "RS"
            elif "_BO" in col:
                method = "BO"
            else:
                # Skip columns that don't have method information
                continue
                
            # Convert log equity premium to simple returns
            log_eq_premium = predictions_df[col].values
            simple_returns = np.exp(log_eq_premium) * (1 + risk_free_rates) - 1
            
            # Apply expanding window approach if needed
            if window_size > 1:
                # For expanding window, we'd typically need date information
                # This is a simplified approach - in a real implementation,
                # we would filter based on dates
                window_size_months = window_size * 12
                simple_returns = simple_returns[window_size_months-1:]
                window_actual_returns = actual_returns[window_size_months-1:]
                window_risk_free_rates = risk_free_rates[window_size_months-1:]
            else:
                window_actual_returns = actual_returns
                window_risk_free_rates = risk_free_rates
                
            # Calculate timing signal
            timing_signal = simple_returns * window_actual_returns > 0
            return_portfolio = window_actual_returns * timing_signal
            avg_return = np.mean(return_portfolio) * 12 * 100  # Annualized
            excess_return = return_portfolio - window_risk_free_rates
            
            # Handle edge case where all excess returns are the same
            if np.std(excess_return) == 0:
                sharpe_ratio = 0
            else:
                sharpe_ratio = np.mean(excess_return) / np.std(excess_return) * np.sqrt(12)  # Annualized
            
            # Store results
            model_key = f"{model_name}"
            if model_key not in window_results[window_label]:
                window_results[window_label][model_key] = {
                    "Method": method,
                    "Average Return %": avg_return,
                    "Sharpe Ratio": sharpe_ratio
                }
    
    # Convert nested dictionaries to DataFrame
    result_dfs = {}
    for window_label, window_data in window_results.items():
        df = pd.DataFrame.from_dict(window_data, orient='index')
        # Ensure consistent column order
        df = df[["Method", "Average Return %", "Sharpe Ratio"]]
        # Sort the DataFrame without using complex key functions that can cause errors with certain index types
        # First, convert the index to strings for safer handling
        df.index = df.index.astype(str)
        
        # Create a sort order dictionary
        sort_order = {}
        # First priority - special models
        if 'HA' in df.index:
            sort_order['HA'] = 0
        if 'BH' in df.index:
            sort_order['BH'] = 1
            
        # Second priority - Net models, then DNet models
        net_models = [idx for idx in df.index if idx.startswith('Net')]
        dnet_models = [idx for idx in df.index if idx.startswith('DNet')]
        
        # Add Net models with numeric sorting
        for model in sorted(net_models, key=lambda x: int(x[3:]) if x[3:].isdigit() else 999):
            sort_order[model] = len(sort_order) + 10
            
        # Add DNet models with numeric sorting
        for model in sorted(dnet_models, key=lambda x: int(x[4:]) if x[4:].isdigit() else 999):
            sort_order[model] = len(sort_order) + 10
            
        # Add any remaining models
        for model in df.index:
            if model not in sort_order:
                sort_order[model] = len(sort_order) + 100
                
        # Sort using the dictionary
        df = df.loc[sorted(df.index, key=lambda x: sort_order.get(x, 999))]
        result_dfs[window_label] = df
        
    return result_dfs

def visualize_performance(performance_dfs, output_path=None):
    """
    Create visualizations of market timing performance
    
    Parameters:
    -----------
    performance_dfs : dict
        Dictionary of DataFrames with performance metrics, keyed by window label
    output_path : str or Path, optional
        Path to save visualizations
    """
    for window_label, df in performance_dfs.items():
        # Create figure
        plt.figure(figsize=(14, 10))
        
        # Create color map based on Method column
        method_colors = {
            "-": "gray",
            "GS": "green",
            "RS": "blue",
            "BO": "purple",
            "Unknown": "lightgray"
        }
        
        # Plot average returns
        plt.subplot(2, 1, 1)
        bar_plot = sns.barplot(x=df.index, y="Average Return %", hue="Method", data=df, 
                              palette=method_colors, dodge=False)
        plt.title(f"Average Returns (%) - {window_label}", fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title="Method")
        
        # Add values on top of bars
        for p in bar_plot.patches:
            bar_plot.annotate(f"{p.get_height():.1f}",
                             (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='bottom', fontsize=9)
        
        # Plot Sharpe ratios
        plt.subplot(2, 1, 2)
        bar_plot = sns.barplot(x=df.index, y="Sharpe Ratio", hue="Method", data=df, 
                              palette=method_colors, dodge=False)
        plt.title(f"Sharpe Ratios - {window_label}", fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title="Method")
        
        # Add values on top of bars
        for p in bar_plot.patches:
            bar_plot.annotate(f"{p.get_height():.2f}",
                             (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if output_path:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path / f"market_timing_{window_label.replace(' ', '_')}.png", dpi=300)
            print(f"Saved plot to {output_path / f'market_timing_{window_label.replace(' ', '_')}.png'}")
        
        plt.close()

def load_data_for_economic_analysis(data_path, oos_start_date=199001):
    """
    Load necessary data for economic value analysis
    
    Parameters:
    -----------
    data_path : str or Path
        Path to data directory
    oos_start_date : int
        Start date for out-of-sample period (format: YYYYMM)
        
    Returns:
    --------
    data_dict : dict
        Dictionary with loaded data
    """
    data_path = Path(data_path)
    data_dict = {}
    
    try:
        # Load predictor data for risk-free rate
        predictor_raw = pd.read_excel(data_path / 'ml_equity_premium_data.xlsx', 
                                      sheet_name='PredictorData1926-2023')
        predictor_raw.set_index(keys='yyyymm', inplace=True)
        
        # Extract risk-free rate (lagged)
        n_rows = predictor_raw.shape[0]
        # Filter to start from oos_start_date (adjust if necessary)
        oos_start_str = str(oos_start_date)
        risk_free_lag = predictor_raw['Rfree'][0:(n_rows - 1)]
        
        # Store index for later alignment
        risk_free_index = risk_free_lag.index
        
        # Try to filter by date if possible
        if oos_start_str in risk_free_lag.index:
            risk_free_lag = risk_free_lag.loc[oos_start_str:]
            print(f"Found risk-free rates starting from {oos_start_str}, length: {len(risk_free_lag)}")
        else:
            print(f"Warning: OOS start date {oos_start_str} not found in risk_free_lag index.")
        
        # Get actual simple returns
        predictor_df = pd.read_excel(data_path / 'ml_equity_premium_data.xlsx', 
                                     sheet_name='result_predictor')
        predictor_df.set_index(keys='month', inplace=True)
        
        # Calculate the next month from oos_start_date
        oos_year = int(oos_start_str[:4])
        oos_month = int(oos_start_str[4:])
        oos_next_month = oos_month + 1 if oos_month < 12 else 1
        oos_next_year = oos_year if oos_month < 12 else oos_year + 1
        oos_next_date = f"{oos_next_year}{oos_next_month:02d}"
        
        # Get equity premium data starting from the next month after OOS start
        if oos_next_date in predictor_df.index:
            equity_premium = predictor_df.equity_premium.loc[oos_next_date:]
            print(f"Found equity premium data starting from {oos_next_date}, length: {len(equity_premium)}")
        else:
            print(f"Warning: Next OOS date {oos_next_date} not found in equity premium index.")
            equity_premium = predictor_df.equity_premium
        
        # Ensure risk_free_lag and equity_premium are aligned before adding
        # Filter both to have matching indices
        common_index = risk_free_index.intersection(equity_premium.index)
        if len(common_index) == 0:
            print(f"Error: No common dates found between risk-free rates and equity premium data.")
            return data_dict
            
        print(f"Found {len(common_index)} common dates for analysis.")
        risk_free_lag_aligned = risk_free_lag.loc[common_index].to_numpy()
        equity_premium_aligned = equity_premium.loc[common_index].to_numpy()
        
        # Now we can safely add the aligned arrays
        actual_simple_returns = equity_premium_aligned + risk_free_lag_aligned
        
        # Historical average forecasting as benchmark - using the same common indices for consistency
        log_equity_premium = predictor_df['log_equity_premium']
        
        # Build expanding window averages for historical average
        ha_values = []
        for date in common_index:
            # Find all dates up to this one
            prior_dates = log_equity_premium.index[log_equity_premium.index <= date]
            if len(prior_dates) > 0:
                # Calculate expanding window mean
                ha_values.append(log_equity_premium.loc[prior_dates].mean())
            else:
                ha_values.append(np.nan)
        
        y_pred_HA = np.array(ha_values)
        
        # Store in data_dict with aligned indices to ensure consistency
        data_dict['risk_free_rates'] = risk_free_lag_aligned
        data_dict['actual_simple_returns'] = actual_simple_returns
        data_dict['historical_average_log_ep'] = y_pred_HA
        data_dict['date_index'] = common_index  # Store the common date index for reference
        
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc(file=sys.stderr)
        
    return data_dict

def run_economic_value_analysis(predictions_file, raw_data_path, output_path=None, window_sizes=[1, 3], oos_start_date=199001, force=False, verbose=False, oos_results_dir="./runs"):
    # Import datetime inside the function to avoid UnboundLocalError
    from datetime import datetime
    """
    Run complete economic value analysis
    
    Parameters:
    -----------
    predictions_file : str or Path
        Path to file with model predictions
    raw_data_path : str or Path
        Path to raw data files
    output_path : str or Path, optional
        Path to save outputs. If None, will use './runs/2_Economic_Value_Analysis/YYYYMMDD_HHMMSS'
    window_sizes : list
        List of expanding window sizes (in years) to evaluate
    oos_start_date : int
        Start date for out-of-sample period (format: YYYYMM)
        
    Returns:
    --------
    performance : dict
        Dictionary of DataFrames with market timing performance metrics
    """
    # Set default output path if none provided
    if output_path is None:
        output_base = Path("./runs/2_Economic_Value_Analysis")
        
        # Check if we already have results for this predictions file
        predictions_file_path = Path(predictions_file)
        predictions_filename = predictions_file_path.name
        
        # Look for existing analysis with this predictions file
        existing_analyses = []
        for analysis_dir in output_base.glob("*"):
            if analysis_dir.is_dir():
                metadata_file = analysis_dir / "analysis_metadata.txt"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, "r") as f:
                            content = f.read()
                            if f"Predictions File: {predictions_file}" in content:
                                # Found matching analysis
                                existing_analyses.append(analysis_dir)
                    except:
                        pass
        
        if existing_analyses and not force:
            # Use the most recent existing analysis
            existing_analyses.sort(reverse=True)  # Sort by name (which has timestamp)
            print(f"Found existing analysis for these predictions at {existing_analyses[0]}")
            print(f"Use --force to re-analyze or a different --output-path to create a new analysis")
            return None
        elif verbose and force and existing_analyses:
            print(f"Forcing re-analysis of predictions (found {len(existing_analyses)} existing analyses)")
        
        # Create new timestamped directory
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_base / f"{timestamp}_oos{oos_start_date}"
    # Load prediction results
    if verbose:
        print(f"Loading predictions from {predictions_file}")
    
    # First read without setting index to check the format
    temp_df = pd.read_csv(predictions_file)
    first_col = temp_df.columns[0]
    
    if verbose:
        print(f"First column in predictions file: {first_col}")
        print(f"First few values: {temp_df[first_col].head().tolist()}")
    
    # Check if the first column contains YYYYmm format integers
    try:
        # Try to convert first column values to integers
        values = [int(x) for x in temp_df[first_col].head()]
        if all(len(str(x)) == 6 for x in values):
            # These look like YYYYmm format dates, read without converting to datetime
            if verbose:
                print(f"Detected YYYYmm format integers in the first column")
            predictions_df = pd.read_csv(predictions_file)
            predictions_df.set_index(first_col, inplace=True)
        else:
            # Standard CSV reading with index in first column
            predictions_df = pd.read_csv(predictions_file, index_col=0)
    except:
        # If conversion fails, just read normally
        predictions_df = pd.read_csv(predictions_file, index_col=0)
    
    # Load necessary data
    if verbose:
        print(f"Loading data for economic analysis from {raw_data_path}")
    data_dict = load_data_for_economic_analysis(raw_data_path, oos_start_date)
    
    if not data_dict or 'actual_simple_returns' not in data_dict:
        print("Error: Could not load necessary data for analysis.")
        return {}
    
    actual_returns = data_dict['actual_simple_returns']
    risk_free_rates = data_dict['risk_free_rates']
    
    # Add HA predictions if not already in predictions_df
    if 'HA' not in predictions_df.columns and 'historical_average_log_ep' in data_dict and 'date_index' in data_dict:
        # Convert prediction index to string for matching with date_index
        if verbose:
            print(f"Prediction DataFrame has {len(predictions_df)} rows with index: {predictions_df.index[:5]}")
            print(f"Historical average data has {len(data_dict['historical_average_log_ep'])} entries")
            print(f"Date index from data_dict: {data_dict['date_index'][:5]}..., length: {len(data_dict['date_index'])}")
        
        # The issue is we need to align the dates between predictions_df and historical_average_log_ep
        # First, we'll use the actual values and risk_free_rates that we already have in the predictions file
        # rather than trying to use the ones from the data_dict
        if 'ActualLogEP' in predictions_df.columns and 'LaggedRF' in predictions_df.columns:
            if verbose:
                print("Using ActualLogEP and LaggedRF from prediction files instead of data_dict")
            actual_returns = predictions_df['ActualMktRet'].values
            risk_free_rates = predictions_df['LaggedRF'].values
            
            # And we'll use the HA values from the prediction files if available
            if 'HA' in predictions_df.columns:
                if verbose:
                    print("Using HA values from prediction files")
            else:
                print("Warning: No HA column in prediction files and cannot align with data_dict dates. Using first HA prediction file available.")
                # Try to use the HA column from the first prediction file as a fallback
                try:
                    first_file = list(Path(oos_results_dir).glob("**/oos_all_predictions_raw_with_actuals.csv"))[0]
                    temp_df = pd.read_csv(first_file)
                    if 'HA' in temp_df.columns:
                        predictions_df['HA'] = temp_df['HA'].values[:len(predictions_df)]
                        if verbose:
                            print(f"Added HA values from {first_file}")
                except Exception as e:
                    print(f"Error adding HA values: {e}")
        else:
            print("Warning: Cannot align actual returns and risk-free rates. Economic value analysis may not be accurate.")
            # This is a fallback that likely won't work well, but we'll try it anyway
            try:
                # Create a temporary Series with the historical average values
                ha_series = pd.Series(data_dict['historical_average_log_ep'], index=data_dict['date_index'])
                # Try to reindex it to match the predictions_df index
                ha_aligned = ha_series.reindex(predictions_df.index)
                # Add to predictions_df
                predictions_df['HA'] = ha_aligned
                if verbose:
                    print("Added reindexed historical average predictions to the dataset")
            except Exception as e:
                print(f"Error adding historical average predictions: {e}")
    
    # Verify lengths match
    min_len = min(len(actual_returns), len(risk_free_rates))
    for col in predictions_df.columns:
        min_len = min(min_len, len(predictions_df[col]))
    
    if min_len < len(actual_returns):
        print(f"Warning: Truncating data to length {min_len} to ensure alignment")
        actual_returns = actual_returns[:min_len]
        risk_free_rates = risk_free_rates[:min_len]
        for col in predictions_df.columns:
            predictions_df[col] = predictions_df[col].values[:min_len]
    
    # Calculate market timing performance
    if verbose:
        print(f"Calculating market timing performance with window sizes: {window_sizes}")
        print(f"Number of predictions: {len(predictions_df)}")
        print(f"Models evaluated: {[col for col in predictions_df.columns if col != 'Actual']}")
    
    performance = calculate_market_timing_performance(
        predictions_df, actual_returns, risk_free_rates, window_sizes
    )
    
    # Save results
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save summary metadata
    with open(output_path / "analysis_metadata.txt", "w") as f:
        f.write(f"Economic Value Analysis\n")
        f.write(f"=====================\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"OOS Start Date: {oos_start_date}\n")
        f.write(f"Window Sizes: {window_sizes}\n")
        f.write(f"Predictions File: {predictions_file}\n")
        f.write(f"Data Path: {raw_data_path}\n")
    
    # Save individual CSV files
    for window_label, df in performance.items():
        df.to_csv(output_path / f"market_timing_performance_{window_label.replace(' ', '_')}.csv")
        print(f"Saved results to {output_path / f'market_timing_performance_{window_label.replace(' ', '_')}.csv'}")
    
    # Create Excel file with multiple sheets
    with pd.ExcelWriter(output_path / "market_timing_performance.xlsx", engine='openpyxl') as writer:
        for window_label, df in performance.items():
            sheet_name = window_label[:31]  # Excel limits sheet names to 31 chars
            df.to_excel(writer, sheet_name=sheet_name)
        print(f"Saved combined results to {output_path / 'market_timing_performance.xlsx'}")
    
    # Create visualizations
    visualize_performance(performance, output_path)
    
    return performance
