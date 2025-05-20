# src/utils/io.py
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import traceback
import sys # For printing to stderr

# Define the path to the data file relative to the project root
_DATA_FILE_XLSX = Path("data") / "ml_equity_premium_data.xlsx"

# Define columns to use as predictors.
# These should match the columns available in the 'result_predictor' source
# after excluding 'month', 'log_equity_premium', 'equity_premium'.
# Updated to include LTY, DE, and VOL_* predictors.
_PREDICTOR_COLUMNS = [
    'DP', 'DY', 'EP', 'SVAR', 'BM', 'NTIS', 'TBL', 'LTR', 'TMS', 'DFY', 'DFR', 'INFL', 'LTY', 'DE',
    'MA_1_9', 'MA_1_12', 'MA_2_9', 'MA_2_12', 'MA_3_9', 'MA_3_12',
    'MOM_1', 'MOM_2', 'MOM_3', 'MOM_6', 'MOM_9', 'MOM_12',
    'VOL_1_9', 'VOL_1_12', 'VOL_2_9', 'VOL_2_12', 'VOL_3_9', 'VOL_3_12'
]

_TARGET_COL = 'log_equity_premium'


def _load_raw_data_from_excel(file_path=_DATA_FILE_XLSX):
    """
    Loads necessary raw data sheets from the Excel file.
    - 'result_predictor': Contains the primary target (log_equity_premium) and pre-calculated predictors.
    - 'PredictorData1926-2023': Contains CRSP S&P500 value-weighted return ('CRSP_SPvw')
                                and risk-free rate ('Rfree').
    """
    try:
        # Load the sheet with predictors and the primary target variable
        # Assuming 'month' column is YYYYMM integer format
        df_result_predictor = pd.read_excel(file_path, sheet_name='result_predictor')
        df_result_predictor['month'] = pd.to_datetime(df_result_predictor['month'].astype(str), format='%Y%m')
        df_result_predictor = df_result_predictor.sort_values(by='month').reset_index(drop=True)

        # Load the sheet with raw market and risk-free rate data
        # Assuming 'yyyymm' column is YYYYMM integer format
        df_goyal_raw = pd.read_excel(file_path, sheet_name='PredictorData1926-2023')
        df_goyal_raw['month'] = pd.to_datetime(df_goyal_raw['yyyymm'].astype(str), format='%Y%m')
        df_goyal_raw = df_goyal_raw.sort_values(by='month').reset_index(drop=True)
        
        # Select and rename necessary columns for CER calculation
        # CRSP_SPvw is R_{m,t+1} (total market return at t+1)
        # Rfree is R_{f,t+1} (risk-free rate at t+1, needs lagging for CER)
        df_market_rf = df_goyal_raw[['month', 'CRSP_SPvw', 'Rfree']].copy()
        df_market_rf.rename(columns={'CRSP_SPvw': 'market_return_tplus1', 
                                     'Rfree': 'risk_free_rate_tplus1'}, inplace=True)

        return df_result_predictor, df_market_rf

    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}", file=sys.stderr)
        print("Please ensure 'data/ml_equity_premium_data.xlsx' exists and contains the required sheets.", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        raise
    except Exception as e:
        print(f"Error loading data from Excel: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        raise


def load_and_prepare_oos_data(oos_start_year_month_int, predictor_cols=None):
    """
    Loads and prepares all necessary data for Out-of-Sample (OOS) evaluation.

    Args:
        oos_start_year_month_int (int): The starting month for OOS evaluation (e.g., 195701).
        predictor_cols (list, optional): List of predictor column names. 
                                         If None, uses _PREDICTOR_COLUMNS.

    Returns:
        dict: A dictionary containing:
            'dates_all_t_np': (N-1,) array of all dates (yyyymm integer format) at time t.
            'predictor_array_for_oos': (N-1, k+1) np.ndarray. First col is y_{t+1} (log_equity_premium), 
                                       next k cols are X_t (predictors).
            'actual_log_ep_all_np': (N-1,) array of y_{t+1} (log_equity_premium), aligned with predictor_array_for_oos.
            'actual_market_returns_all_np': (N-1,) array of R_{m,t+1} (simple total market return), aligned.
            'lagged_risk_free_rates_all_np': (N-1,) array of R_{f,t} (simple risk-free rate), aligned.
            'historical_average_all_np': (N-1,) array of HA forecasts for y_{t+1}, aligned.
            'oos_start_idx_in_arrays': Index in the (N-1) length arrays where OOS period begins.
            'predictor_names': List of predictor names used.
    """
    if predictor_cols is None:
        predictor_names = list(_PREDICTOR_COLUMNS) # Ensure it's a mutable list if modified later
    else:
        predictor_names = list(predictor_cols)

    df_result_predictor, df_market_rf = _load_raw_data_from_excel()

    # Merge the two dataframes on 'month' (which is now datetime object)
    df_merged = pd.merge(df_result_predictor, df_market_rf, on='month', how='inner')
    
    # Ensure all specified predictor columns are present
    missing_cols = [col for col in predictor_names if col not in df_merged.columns]
    if missing_cols:
        raise ValueError(f"Missing predictor columns in the merged data: {missing_cols}")

    # --- Prepare the main predictor array for OOS loop: [y_{t+1}, X_t] ---
    # y_{t+1} is log_equity_premium at row i (from df_merged)
    # X_t are predictors from row i-1 (from df_merged)
    
    # Target variable (log_equity_premium_{t+1})
    log_ep_tplus1 = df_merged[_TARGET_COL].values[1:] # Shape: (N-1,)

    # Predictors (X_t) - these are from the previous period
    X_t_df = df_merged[predictor_names].iloc[:-1, :] # Shape: (N-1, k_predictors)
    
    # Dates for X_t (and for which y_t+1 is being predicted)
    # These dates correspond to time 't'.
    # Original 'month' in df_merged is datetime. Convert to YYYYMM int for oos_start_date comparison.
    dates_t_for_xt_ytplus1 = df_merged['month'].dt.strftime('%Y%m').astype(int).values[:-1] # Shape: (N-1,)

    # Construct the array: [y_{t+1}, X_t]
    predictor_array_for_oos = np.concatenate(
        [log_ep_tplus1.reshape(-1, 1), X_t_df.values], axis=1
    ) # Shape: (N-1, 1 + k_predictors)
    
    # --- Other series, aligned to the (N-1) length of predictor_array_for_oos ---
    # actual_log_ep_all_np is y_{t+1}
    actual_log_ep_all_np = log_ep_tplus1.copy() # Shape: (N-1,)

    # market_return_tplus1 is R_{m,t+1} (simple total market return at t+1)
    actual_market_returns_all_np = df_merged['market_return_tplus1'].values[1:] # Shape: (N-1,)

    # risk_free_rate_tplus1 is R_{f,t+1}. We need R_{f,t} for CER.
    # So, R_{f,t} is risk_free_rate_tplus1 from the previous row.
    lagged_risk_free_rates_all_np = df_merged['risk_free_rate_tplus1'].values[:-1] # Shape: (N-1,)

    # Historical Average (HA) forecast for log_equity_premium
    # HA for y_{k+1} (which is actual_log_ep_all_np[k]) uses mean of (y_0, ..., y_k) from original series.
    # original_log_ep_series is y_0, y_1, ..., y_{N-1} from df_merged.
    original_log_ep_series = df_merged[_TARGET_COL].values # Length N
    
    historical_average_all_np = np.full(len(actual_log_ep_all_np), np.nan) # Length N-1
    if len(original_log_ep_series) > 0:
        expanding_sum = np.cumsum(original_log_ep_series)
        expanding_count = np.arange(1, len(original_log_ep_series) + 1)
        # HA for y_{k+1} (target at index k in actual_log_ep_all_np) uses data up to y_k
        # original_log_ep_series[0] to original_log_ep_series[k]
        # So, for actual_log_ep_all_np[k], HA is mean(original_log_ep_series[0]...original_log_ep_series[k])
        for k_idx in range(len(historical_average_all_np)): # k_idx from 0 to N-2
            # HA for actual_log_ep_all_np[k_idx] (which is y_{k_idx+1})
            # uses original_log_ep_series[0] up to original_log_ep_series[k_idx]
            historical_average_all_np[k_idx] = expanding_sum[k_idx] / expanding_count[k_idx]

    # Determine OOS start index
    oos_start_idx_mask = dates_t_for_xt_ytplus1 >= oos_start_year_month_int
    if not np.any(oos_start_idx_mask):
        raise ValueError(f"OOS start date {oos_start_year_month_int} is after the last available data point "
                         f"({dates_t_for_xt_ytplus1[-1] if len(dates_t_for_xt_ytplus1)>0 else 'N/A'}).")
    oos_start_idx_in_arrays = np.where(oos_start_idx_mask)[0][0]

    # Sanity check lengths
    expected_len = len(df_merged) - 1
    if expected_len <= 0:
        raise ValueError("Data processing resulted in zero or negative length arrays. Check input data.")
        
    assert len(predictor_array_for_oos) == expected_len, "Length mismatch for predictor_array_for_oos"
    assert len(actual_log_ep_all_np) == expected_len, "Length mismatch for actual_log_ep_all_np"
    assert len(actual_market_returns_all_np) == expected_len, "Length mismatch for actual_market_returns_all_np"
    assert len(lagged_risk_free_rates_all_np) == expected_len, "Length mismatch for lagged_risk_free_rates_all_np"
    assert len(historical_average_all_np) == expected_len, "Length mismatch for historical_average_all_np"
    assert len(dates_t_for_xt_ytplus1) == expected_len, "Length mismatch for dates_t_for_xt_ytplus1"
    
    return {
        'dates_all_t_np': dates_t_for_xt_ytplus1, 
        'predictor_array_for_oos': predictor_array_for_oos,
        'actual_log_ep_all_np': actual_log_ep_all_np,
        'actual_market_returns_all_np': actual_market_returns_all_np,
        'lagged_risk_free_rates_all_np': lagged_risk_free_rates_all_np,
        'historical_average_all_np': historical_average_all_np,
        'oos_start_idx_in_arrays': oos_start_idx_in_arrays,
        'predictor_names': predictor_names
    }

# --- Existing train_val_split ---
# (No changes needed here for OOS, it's used by in-sample HPO if called)
def train_val_split(X, y, val_ratio=0.15, split_by_index=True):
    """Splits data into training and validation sets."""
    n_total = len(X)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val

    if n_train <= 0 or n_val < 0: # n_val can be 0 if val_ratio is very small or n_total is small
        print(f"Warning: train_val_split resulted in n_train={n_train}, n_val={n_val}. Adjusting.", file=sys.stderr)
        if n_total > 1 and n_val == 0 : # Ensure val set has at least 1 if possible
            n_val = 1
            n_train = n_total - 1
        elif n_total <=1 : # Cannot split
            print(f"Error: Cannot split data with {n_total} samples.", file=sys.stderr)
            return X, X, y, y # Or raise error, or return (X, None, y, None)

    if split_by_index:
        if isinstance(X, pd.DataFrame) and not X.index.is_monotonic_increasing:
            print("Warning: X index is not monotonic increasing. Sorting by index before splitting.", file=sys.stderr)
            X = X.sort_index()
        if isinstance(y, (pd.Series, pd.DataFrame)) and not y.index.is_monotonic_increasing:
            print("Warning: y index is not monotonic increasing. Sorting by index before splitting.", file=sys.stderr)
            y = y.sort_index()
        
        X_train = X.iloc[:n_train]
        X_val = X.iloc[n_train:]
        y_train = y.iloc[:n_train]
        y_val = y.iloc[n_train:]
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_ratio, shuffle=False, random_state=42)

    print(f"Data split (by index={split_by_index}): Train={len(X_train)}, Validation={len(X_val)} (Ratio: {val_ratio})")
    return X_train, X_val, y_train, y_val


# --- Global constants for lazy loading (for existing in-sample scripts) ---
# These are kept for backward compatibility. New OOS scripts should use load_and_prepare_oos_data().
try:
    _DF_RESULT_PREDICTOR_GLOBAL, _DF_MARKET_RF_GLOBAL = _load_raw_data_from_excel()
    
    X_ALL_DF_GLOBAL = _DF_RESULT_PREDICTOR_GLOBAL[_PREDICTOR_COLUMNS].copy()
    Y_ALL_SERIES_GLOBAL = _DF_RESULT_PREDICTOR_GLOBAL[_TARGET_COL].copy()
    
    _temp_df_for_rf_global = pd.merge(
        _DF_RESULT_PREDICTOR_GLOBAL[['month', _TARGET_COL]], 
        _DF_MARKET_RF_GLOBAL[['month', 'risk_free_rate_tplus1']], 
        on='month', 
        how='left'
    )
    RF_ALL_SERIES_GLOBAL = _temp_df_for_rf_global['risk_free_rate_tplus1'].shift(1).bfill()
    
    X_ALL = X_ALL_DF_GLOBAL.values
    Y_ALL = Y_ALL_SERIES_GLOBAL.values.reshape(-1, 1)
    RF_ALL = RF_ALL_SERIES_GLOBAL.values 
    
    Y_ALL_UNSCALED = Y_ALL_SERIES_GLOBAL.copy()
    RF_ALL_UNSCALED = RF_ALL_SERIES_GLOBAL.copy()
    
    if len(Y_ALL_SERIES_GLOBAL) > 0:
        _expanding_sum_global = np.cumsum(Y_ALL_SERIES_GLOBAL.values)
        _expanding_count_global = np.arange(1, len(Y_ALL_SERIES_GLOBAL) + 1)
        HISTORICAL_AVERAGE_EXPANDING = _expanding_sum_global / _expanding_count_global
    else:
        HISTORICAL_AVERAGE_EXPANDING = np.array([])

except Exception as e:
    print(f"Could not pre-load global data X_ALL, Y_ALL, RF_ALL in io.py: {e}", file=sys.stderr)
    print("In-sample scripts relying on these globals might fail.", file=sys.stderr)
    X_ALL, Y_ALL, RF_ALL, Y_ALL_UNSCALED, RF_ALL_UNSCALED, HISTORICAL_AVERAGE_EXPANDING = None, None, None, None, None, None