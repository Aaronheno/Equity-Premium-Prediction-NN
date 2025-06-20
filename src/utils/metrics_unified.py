"""
Unified metrics and utility functions for equity premium prediction.

This module consolidates functions from metrics.py, evaluation.py, and processing.py
for calculating performance metrics and processing data in equity premium prediction.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# --- R-squared Metrics ---

def compute_oos_r_square(actual, benchmark, predicted):
    """
    Computes the out-of-sample R-squared relative to a benchmark forecast.
    Formula: 1 - (MSE_model / MSE_benchmark)
    
    Args:
        actual (array-like): Actual values
        benchmark (array-like): Benchmark predictions (e.g., historical average)
        predicted (array-like): Model predictions
        
    Returns:
        float: Out-of-sample R-squared value
    """
    # Ensure inputs are numpy arrays
    actual = np.asarray(actual)
    benchmark = np.asarray(benchmark)
    predicted = np.asarray(predicted)
    
    mse_benchmark = mean_squared_error(actual, benchmark)
    if mse_benchmark == 0:  # Avoid division by zero if benchmark is perfect
        return -np.inf if mean_squared_error(actual, predicted) > 0 else 1.0
    mse_predicted = mean_squared_error(actual, predicted)
    return 1 - (mse_predicted / mse_benchmark)

# Alias for backward compatibility with existing code
compute_in_r_square = compute_oos_r_square

# --- Success Ratio ---

def compute_success_ratio(actual, predicted):
    """
    Computes the percentage of times the predicted sign matches the actual sign.
    
    Args:
        actual (array-like): Actual values
        predicted (array-like): Model predictions
        
    Returns:
        float: Success ratio (between 0 and 1)
    """
    # Ensure inputs are numpy arrays
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    return np.mean(np.sign(actual) == np.sign(predicted))

# --- MSFE-adjusted Statistic ---

def compute_MSFE_adjusted(actual, benchmark, predicted, c=0):
    """
    Computes the MSFE-adjusted statistic based on Clark and West (2007).
    
    Args:
        actual (array-like): Actual values
        benchmark (array-like): Benchmark predictions (e.g., historical average)
        predicted (array-like): Model predictions
        c (float): Adjustment constant (typically 0)
        
    Returns:
        tuple: (MSFE-adjusted statistic, p-value)
    """
    # Ensure inputs are numpy arrays
    actual = np.asarray(actual)
    benchmark = np.asarray(benchmark)
    predicted = np.asarray(predicted)
    
    # Calculate forecast errors
    err_benchmark = actual - benchmark
    err_predicted = actual - predicted
    
    # Calculate squared errors
    sq_err_benchmark = err_benchmark ** 2
    sq_err_predicted = err_predicted ** 2
    
    # Calculate adjusted term
    adj_term = (benchmark - predicted) ** 2
    
    # Calculate MSFE-adjusted statistic components
    f_t = sq_err_benchmark - sq_err_predicted + adj_term
    mean_f = np.mean(f_t)
    std_f = np.std(f_t, ddof=1) / np.sqrt(len(f_t))
    
    # Compute MSFE-adjusted statistic
    msfe_adj = mean_f / std_f if std_f > 0 else 0
    
    # Approximate p-value (one-sided test: H0: msfe_adj <= 0, H1: msfe_adj > 0)
    from scipy.stats import norm
    p_value = 1 - norm.cdf(msfe_adj)
    
    return msfe_adj, p_value

# --- Certainty Equivalent Return (CER) Calculation ---

def compute_CER(predicted_returns, risk_free_rates=None, gamma=3, freq=12):
    """
    Calculates the annualized Certainty Equivalent Return (CER) for a strategy
    based on predicted returns, following Campbell & Thompson (2008) approach.
    
    This function supports two distinct calling patterns for backward compatibility:
    
    Pattern 1 (from metrics.py):
      compute_CER(actual_returns, predicted_returns, risk_free_rates, gamma=3.0)
      
    Pattern 2 (from evaluation.py):
      compute_CER(pred, rf, gamma=5, freq=12)
    
    The function will try to detect which pattern is used based on parameters.
    
    Args:
        predicted_returns (array-like): In Pattern 1, this is actual_returns.
                                        In Pattern 2, this is predicted returns.
        risk_free_rates (array-like): In Pattern 1, this is predicted_returns.
                                      In Pattern 2, this is risk-free rates.
                                      If None, simple CER calculation is used.
        gamma (float): Risk aversion coefficient (default: 3)
        freq (int): Number of periods per year (default: 12 for monthly data)
        
    Returns:
        float: Annualized CER value
    """
    # Detect calling pattern
    if risk_free_rates is None:
        # Simple CER calculation (Pattern 2 without risk-free rate)
        pred = np.asarray(predicted_returns).ravel()
        mu_p = np.mean(pred)
        sigma_p_sq = np.var(pred)
        cer = mu_p - 0.5 * gamma * sigma_p_sq
        return cer * freq * 100  # Annualize and convert to percentage
    
    # Check if third parameter is passed (Pattern 1)
    if hasattr(gamma, "__len__") and not isinstance(gamma, str):
        # Pattern 1: compute_CER(actual_returns, predicted_returns, risk_free_rates, gamma)
        actual_returns = np.asarray(predicted_returns).ravel()
        predicted_returns = np.asarray(risk_free_rates).ravel()
        risk_free_rates = np.asarray(gamma).ravel()
        gamma = freq if isinstance(freq, (int, float)) else 3.0
        
        if not (len(actual_returns) == len(predicted_returns) == len(risk_free_rates)):
            raise ValueError("Input arrays must have the same length.")
        if len(actual_returns) == 0:
            return 0.0
        
        # Calculate predicted excess returns
        predicted_excess_returns = predicted_returns - risk_free_rates
        
        # Calculate portfolio weights
        weights = (1 / gamma) * predicted_excess_returns
        weights = np.clip(weights, 0, 1)
        
        # Calculate portfolio returns
        portfolio_returns = weights * actual_returns + (1 - weights) * risk_free_rates
        
        # Calculate portfolio excess returns
        portfolio_excess_returns = portfolio_returns - risk_free_rates
        
        # Calculate mean and variance
        mu_p = np.mean(portfolio_excess_returns)
        sigma_p_sq = np.var(portfolio_excess_returns)
        
        # Calculate CER
        cer = mu_p - 0.5 * gamma * sigma_p_sq
        
        # Annualize
        return cer * 12
    else:
        # Pattern 2: compute_CER(pred, rf, gamma, freq)
        pred = np.asarray(predicted_returns).ravel()
        rf = np.asarray(risk_free_rates).ravel()
        
        if pred.shape != rf.shape:
            raise ValueError(f"Shapes of pred {pred.shape} and rf {rf.shape} must match after ravel()")
        if len(pred) == 0:
            return np.nan
        
        # Calculate excess return
        rp = pred - rf
        
        # Calculate statistics for base period
        mu_p = np.mean(rp)
        sigma_p_sq = np.var(rp)
        
        # Annualize
        mu_p_ann = mu_p * freq
        sigma_p_sq_ann = sigma_p_sq * freq
        
        # Calculate annualized CER in percentage
        return (mu_p_ann - 0.5 * gamma * sigma_p_sq_ann) * 100

# --- Data Processing ---

def scale_data(X, y):
    """
    Scales features (X) and target (y) using StandardScaler.
    
    Args:
        X (pd.DataFrame or np.ndarray): Feature data
        y (pd.Series, pd.DataFrame, or np.ndarray): Target data
        
    Returns:
        tuple: (X_scaled, y_scaled, scaler_x, scaler_y)
               Scaled data preserves input format (DataFrame or array) and the fitted scalers
    """
    # Ensure y is a 2D numpy array for scaler compatibility
    y_is_df = isinstance(y, pd.DataFrame)
    y_is_series = isinstance(y, pd.Series)
    y_index = y.index if (y_is_df or y_is_series) else None
    y_columns = y.columns if y_is_df else None
    
    if y_is_df or y_is_series:
        y_array = y.values.reshape(-1, 1)
    elif isinstance(y, np.ndarray):
        y_array = y.reshape(-1, 1)
    else:
        raise TypeError("Unsupported type for y. Expected pandas Series/DataFrame or numpy array.")
    
    # Remember X format
    X_is_df = isinstance(X, pd.DataFrame)
    X_index = X.index if X_is_df else None
    X_columns = X.columns if X_is_df else None
    
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    # Fit and transform
    X_np = X.values if X_is_df else X
    X_scaled_np = scaler_x.fit_transform(X_np)
    y_scaled_np = scaler_y.fit_transform(y_array)
    
    # Return in the same format as input
    if X_is_df:
        X_scaled = pd.DataFrame(X_scaled_np, index=X_index, columns=X_columns)
    else:
        X_scaled = X_scaled_np
    
    if y_is_df:
        y_scaled = pd.DataFrame(y_scaled_np, index=y_index, columns=y_columns)
    elif y_is_series:
        y_scaled = pd.Series(y_scaled_np.flatten(), index=y_index, name=y.name)
    else:
        y_scaled = y_scaled_np
    
    print("Data scaling complete.")
    return X_scaled, y_scaled, scaler_x, scaler_y

# --- New CER Functions for OOS Evaluation ---

def compute_CER_binary(actual_market_returns, predictions_log_ep, risk_free_rates, gamma=3):
    """
    Binary (0/1) Campbell-Thompson portfolio rule for CER calculation.
    This is the approach used in the main paper results.
    
    Portfolio allocation rule:
    - w_t = 1 if prediction > 0 (invest 100% in market)
    - w_t = 0 if prediction <= 0 (invest 100% in risk-free)
    
    Args:
        actual_market_returns (array-like): Actual market returns (not excess returns)
        predictions_log_ep (array-like): Predicted log equity premiums
        risk_free_rates (array-like): Risk-free rates (aligned with market returns)
        gamma (float): Risk aversion coefficient (default: 3)
        
    Returns:
        float: Annualized CER value
    """
    # Ensure numpy arrays
    actual_market_returns = np.asarray(actual_market_returns).ravel()
    predictions_log_ep = np.asarray(predictions_log_ep).ravel()
    risk_free_rates = np.asarray(risk_free_rates).ravel()
    
    # Validate inputs
    if not (len(actual_market_returns) == len(predictions_log_ep) == len(risk_free_rates)):
        raise ValueError("All input arrays must have the same length.")
    if len(actual_market_returns) == 0:
        return np.nan
    
    # Binary weights based on prediction sign
    weights = (predictions_log_ep > 0).astype(float)
    
    # Calculate portfolio returns
    portfolio_returns = weights * actual_market_returns + (1 - weights) * risk_free_rates
    
    # Calculate portfolio excess returns
    portfolio_excess_returns = portfolio_returns - risk_free_rates
    
    # Calculate mean and variance
    mu_p = np.mean(portfolio_excess_returns)
    sigma_p_sq = np.var(portfolio_excess_returns)
    
    # Calculate CER
    cer = mu_p - (gamma / 2) * sigma_p_sq
    
    # Annualize (monthly to annual)
    return cer * 12

def compute_CER_proportional(actual_market_returns, predictions_log_ep, risk_free_rates, gamma=3):
    """
    Proportional Campbell-Thompson portfolio rule for CER calculation.
    This is an alternative approach for robustness comparison.
    
    Portfolio allocation rule:
    - w_t = (1/γ) * predicted_log_equity_premium
    - w_t is clipped to [0, 1] (no leverage, no short-selling)
    
    Args:
        actual_market_returns (array-like): Actual market returns (not excess returns)
        predictions_log_ep (array-like): Predicted log equity premiums
        risk_free_rates (array-like): Risk-free rates (aligned with market returns)
        gamma (float): Risk aversion coefficient (default: 3)
        
    Returns:
        float: Annualized CER value
    """
    # Ensure numpy arrays
    actual_market_returns = np.asarray(actual_market_returns).ravel()
    predictions_log_ep = np.asarray(predictions_log_ep).ravel()
    risk_free_rates = np.asarray(risk_free_rates).ravel()
    
    # Validate inputs
    if not (len(actual_market_returns) == len(predictions_log_ep) == len(risk_free_rates)):
        raise ValueError("All input arrays must have the same length.")
    if len(actual_market_returns) == 0:
        return np.nan
    
    # Proportional weights based on predicted log equity premium
    # Note: For small values, log equity premium ≈ excess return
    weights = (1 / gamma) * predictions_log_ep
    weights = np.clip(weights, 0, 1)
    
    # Calculate portfolio returns
    portfolio_returns = weights * actual_market_returns + (1 - weights) * risk_free_rates
    
    # Calculate portfolio excess returns
    portfolio_excess_returns = portfolio_returns - risk_free_rates
    
    # Calculate mean and variance
    mu_p = np.mean(portfolio_excess_returns)
    sigma_p_sq = np.var(portfolio_excess_returns)
    
    # Calculate CER
    cer = mu_p - (gamma / 2) * sigma_p_sq
    
    # Annualize (monthly to annual)
    return cer * 12
