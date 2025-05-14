import numpy as np
from sklearn.metrics import mean_squared_error

# --- R-squared vs Benchmark ---
def compute_in_r_square(actual, benchmark, predicted):
    """
    Computes the in-sample R-squared relative to a benchmark forecast.
    Formula: 1 - (MSE_model / MSE_benchmark)
    """
    mse_benchmark = mean_squared_error(actual, benchmark)
    if mse_benchmark == 0: # Avoid division by zero if benchmark is perfect
        return -np.inf if mean_squared_error(actual, predicted) > 0 else 1.0
    mse_predicted = mean_squared_error(actual, predicted)
    return 1 - (mse_predicted / mse_benchmark)

# --- Success Ratio ---
def compute_success_ratio(actual, predicted):
    """
    Computes the percentage of times the predicted sign matches the actual sign.
    """
    return np.mean(np.sign(actual) == np.sign(predicted))

# --- Certainty Equivalent Return (CER) Calculation ---
def compute_CER(actual_returns, predicted_returns, risk_free_rates, gamma=3.0):
    """
    Calculates the annualized Certainty Equivalent Return (CER) for a strategy
    based on predicted returns, following Campbell & Thompson (2008) approach.

    Args:
        actual_returns (np.ndarray): 1D array of actual realized returns (e.g., y_ALL_unscaled).
        predicted_returns (np.ndarray): 1D array of predicted returns for the same period.
        risk_free_rates (np.ndarray): 1D array of risk-free rates for the same period.
        gamma (float): Coefficient of relative risk aversion. Default is 3.

    Returns:
        float: Annualized CER value for the strategy.
    """
    # Ensure inputs are 1D arrays
    actual_returns = np.asarray(actual_returns).ravel()
    predicted_returns = np.asarray(predicted_returns).ravel()
    risk_free_rates = np.asarray(risk_free_rates).ravel()

    if not (len(actual_returns) == len(predicted_returns) == len(risk_free_rates)):
        raise ValueError("Input arrays must have the same length.")
    if len(actual_returns) == 0:
        return 0.0 # Or handle as appropriate

    # 1. Calculate predicted EXCESS returns (prediction for rt+1 - rf,t)
    # Assumes predicted_returns are for t+1 and risk_free_rates are for t
    predicted_excess_returns = predicted_returns - risk_free_rates

    # 2. Calculate portfolio weights based on predictions
    # Simple mean-variance weight: w = (1/gamma) * (predicted_excess_return / variance_prediction)
    # Since we don't predict variance, use a common simplification:
    # Assume constant variance or ignore it for weight calculation relative to gamma.
    # w_t = (1 / gamma) * predicted_excess_return_t
    weights = (1 / gamma) * predicted_excess_returns

    # Apply constraints to weights (e.g., long-only, max 100% allocation)
    # Adjust constraints if your methodology differs (e.g., allowing leverage > 1 or shorting < 0)
    weights = np.clip(weights, 0, 1)

    # 3. Calculate ACTUAL realized portfolio returns based on weights and ACTUAL market returns
    # Portfolio Return = weight_t * actual_return_{t+1} + (1 - weight_t) * risk_free_rate_t
    # Note: actual_returns are rt+1, risk_free_rates are rf,t
    portfolio_returns = weights * actual_returns + (1 - weights) * risk_free_rates

    # 4. Calculate ACTUAL realized portfolio EXCESS returns
    portfolio_excess_returns = portfolio_returns - risk_free_rates

    # 5. Calculate mean (mu_p) and variance (sigma_p_sq) of the portfolio's excess returns
    mu_p = np.mean(portfolio_excess_returns)
    sigma_p_sq = np.var(portfolio_excess_returns)

    # 6. Calculate the CER value for this portfolio strategy
    # CER = mu_p - 0.5 * gamma * sigma_p_sq
    cer = mu_p - 0.5 * gamma * sigma_p_sq

    # 7. Annualize the CER value (assuming monthly data)
    annualized_cer = cer * 12

    return annualized_cer 