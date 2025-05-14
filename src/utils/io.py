from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import traceback

# Define the path to the data file relative to the project root
_DATA_FILE = Path("data") / "ml_equity_premium_data.xlsx"

# Define columns to use as predictors (ensure these match your Excel sheet)
# Exclude 'month', 'log_equity_premium', 'equity_premium'
# Include any extra features you added (like DE, LTY if they are in your sheet)
_COLS = [
    'DP', 'DY', 'EP', 'SVAR', 'BM', 'NTIS', 'TBL', 'LTR', 'TMS', 'DFY', 'DFR', 'INFL',
    'MA_1_9', 'MA_1_12', 'MA_2_9', 'MA_2_12', 'MA_3_9', 'MA_3_12',
    'MOM_1', 'MOM_2', 'MOM_3', 'MOM_6', 'MOM_9', 'MOM_12',
    # Add 'DE', 'LTY' here if they are in your 'result_predictor' sheet and you want to use them
    # 'DE', 'LTY'
]


def _load_data_from_excel(file_path):
    """
    Loads predictors and target from the 'result_predictor' sheet,
    and the risk-free rate from the 'PredictorData1926-2023' sheet,
    aligning them by date AND ensuring features are lagged relative to the target.
    """
    predictor_date_col = "month"
    rf_date_col = "yyyymm"
    date_format = "%Y%m" # Assuming dates are like 192701

    try:
        # --- Load 'result_predictor' sheet ---
        df_predictors_raw = pd.read_excel(file_path, sheet_name="result_predictor")
        if predictor_date_col not in df_predictors_raw.columns:
            raise KeyError(f"Date column '{predictor_date_col}' not found in sheet 'result_predictor'")
        # Convert date column and set as index
        df_predictors_raw[predictor_date_col] = pd.to_datetime(df_predictors_raw[predictor_date_col], format=date_format)
        df_predictors_raw.set_index(predictor_date_col, inplace=True)
        df_predictors_raw.sort_index(inplace=True) # Ensure data is sorted by date

        # --- Load 'PredictorData1926-2023' sheet for Risk-Free rate ---
        # Assuming the sheet name is consistent with the paper's original data source name
        rf_sheet_name = "PredictorData1926-2023" # Adjust if your sheet name is different
        df_rf_raw = pd.read_excel(file_path, sheet_name=rf_sheet_name)
        if rf_date_col not in df_rf_raw.columns:
             raise KeyError(f"Date column '{rf_date_col}' not found in sheet '{rf_sheet_name}'.")
        if "Rfree" not in df_rf_raw.columns:
             raise KeyError(f"Column 'Rfree' not found in sheet '{rf_sheet_name}'.")
        # Convert date column and set as index
        df_rf_raw[rf_date_col] = pd.to_datetime(df_rf_raw[rf_date_col], format=date_format)
        df_rf_raw.set_index(rf_date_col, inplace=True)
        df_rf_raw.sort_index(inplace=True) # Ensure data is sorted by date
        # Extract only the Risk-Free rate column
        rf_series = df_rf_raw["Rfree"].astype("float32") # Keep as Series for now

        # --- Create Lagged Features and Aligned Target/RF ---
        # Features X(t-1): Select predictor columns and shift them forward by 1 period
        # Ensure all columns in _COLS actually exist in df_predictors_raw
        missing_cols = [col for col in _COLS if col not in df_predictors_raw.columns]
        if missing_cols:
            raise KeyError(f"Predictor columns not found in 'result_predictor' sheet: {missing_cols}")
        X = df_predictors_raw[_COLS].shift(1).astype("float32")

        # Target y(t): Select the target column (no shift needed)
        if "log_equity_premium" not in df_predictors_raw.columns:
             raise KeyError("Target column 'log_equity_premium' not found in 'result_predictor' sheet.")
        y = df_predictors_raw[["log_equity_premium"]].astype("float32")

        # Risk-Free Rate rf(t-1): Align with X(t-1) and y(t) index, then shift
        # We need rf(t-1) for CER calculation later if using lagged features
        # Reindex first to match the dates in y and X, then shift
        rf = rf_series.reindex(y.index).shift(1).astype("float32")
        rf = rf.to_frame(name="Rfree") # Convert back to DataFrame

        # --- Combine and Drop NaNs ---
        # Combine y(t), X(t-1), rf(t-1) based on the index of y(t) and X(t-1)
        # The shift(1) introduces NaN in the first row for X and rf.
        combined = pd.concat([y, X, rf], axis=1)
        combined.dropna(inplace=True) # This removes the first row where X and rf are NaN

        # Separate back into final X, y, rf
        X = combined[_COLS]
        y = combined[["log_equity_premium"]]
        rf = combined[["Rfree"]]
        # --- End Lagging and Alignment ---


        if X.empty or y.empty or rf.empty:
            raise ValueError("Data is empty after loading, alignment, lagging, and NaN drop. Check input file and date columns.")

        print(f"Data loaded and aligned (X lagged). Shapes: X={X.shape}, y={y.shape}, rf={rf.shape}")
        # Add debug prints to verify alignment
        print("\n--- Head of Aligned Data (Post Lagging/DropNA) ---")
        print(combined.head())
        print("\n--- Tail of Aligned Data (Post Lagging/DropNA) ---")
        print(combined.tail())
        # --- End Debug Prints ---
        return X, y, rf

    except FileNotFoundError:
        print(f"Error: Excel file not found at {file_path}")
        raise
    except KeyError as e:
        print(f"Error: Problem finding a required column or sheet name: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        traceback.print_exc() # Print full traceback for unexpected errors
        raise

# --- Load data ONCE globally ---
try:
    import traceback # Make sure traceback is imported
    X_ALL, Y_ALL, RF_ALL = _load_data_from_excel(_DATA_FILE)
except Exception as e:
    print("!!! CRITICAL ERROR DURING GLOBAL DATA LOADING !!!")
    # Decide how to handle - exit? Or let subsequent code fail?
    # For robustness, maybe create empty DataFrames to allow later code to fail more gracefully?
    X_ALL, Y_ALL, RF_ALL = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    # Or re-raise the exception:
    # raise e
# --- End global loading ---


def train_val_split(X, y, val_ratio=0.15, split_by_index=True):
    """Splits data into training and validation sets."""
    n_total = len(X)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val

    if split_by_index:
        # Ensure data is sorted by index (time) before splitting
        if not X.index.is_monotonic_increasing:
            print("Warning: Data index is not monotonic increasing. Sorting by index before splitting.")
            X = X.sort_index()
            y = y.sort_index() # Sort y by the same index

        X_train = X.iloc[:n_train]
        X_val = X.iloc[n_train:]
        y_train = y.iloc[:n_train]
        y_val = y.iloc[n_train:]
    else:
        # Random split (less common for time series)
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_ratio, random_state=42, shuffle=False) # Keep shuffle=False for consistency

    print(f"Data split (by index={split_by_index}): Train={len(X_train)}, Validation={len(X_val)} (Ratio: {val_ratio})")
    return X_train, X_val, y_train, y_val

# --- Scaler Function ---
# (scale_data function remains unchanged)
def scale_data(X_df, y_df):
    """Scales features (X) and target (y) using StandardScaler."""
    scaler_x = StandardScaler()
    X_scaled = scaler_x.fit_transform(X_df)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y_df) # y_df should be DataFrame (e.g., [[]])

    # Convert scaled numpy arrays back to DataFrames with original index and columns
    X_scaled_df = pd.DataFrame(X_scaled, index=X_df.index, columns=X_df.columns)
    y_scaled_df = pd.DataFrame(y_scaled, index=y_df.index, columns=y_df.columns)

    print("Data scaling complete.")
    return X_scaled_df, y_scaled_df, scaler_x, scaler_y





