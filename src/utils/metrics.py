# Compatibility layer that imports from the unified metrics module
# This maintains backward compatibility while using the improved implementation
from src.utils.metrics_unified import (
    compute_oos_r_square,
    compute_in_r_square,
    compute_success_ratio,
    compute_CER,
    compute_MSFE_adjusted
)

# Leave original docstrings for backward compatibility
compute_oos_r_square.__doc__ = """
Computes the out-of-sample R-squared relative to a benchmark forecast.
Formula: 1 - (MSE_model / MSE_benchmark)
"""

compute_in_r_square.__doc__ = """
Computes the in-sample R-squared relative to a benchmark forecast.
Formula: 1 - (MSE_model / MSE_benchmark)
"""

compute_success_ratio.__doc__ = """
Computes the percentage of times the predicted sign matches the actual sign.
"""

compute_CER.__doc__ = """
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

compute_MSFE_adjusted.__doc__ = """
Computes the MSFE-adjusted statistic based on Clark and West (2007).
"""