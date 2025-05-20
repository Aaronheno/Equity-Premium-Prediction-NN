import numpy as np
from scipy.stats import norm


def PT_test(actual, forecast):
    """
    Implements the Directional Accuracy Test of Pesaran and Timmerman, 1992.
    Reference:
    Pesaran, M.H. and Timmermann, A. 1992, A simple nonparametric test of predictive performance,
    Journal of Business and Economic Statistics, 10(4), 461–465.

    :param actual: a column vector of actual values
    :param forecast: a column vector of the forecasted values.
    :return: a tuple of three elements, the first element is the success ratio,
    the second element is the PT statistic and the third one is the corresponding p-value.
    """
    n = actual.shape[0]
    if n != forecast.shape[0]:
        raise ValueError('Length of forecast and actual must be the same')
    x_t = np.zeros(n).reshape((-1, 1))
    z_t = np.zeros(n).reshape((-1, 1))
    y_t = np.zeros(n).reshape((-1, 1))
    x_t[actual > 0] = 1.0
    y_t[forecast > 0] = 1.0
    p_y = np.mean(y_t)
    p_x = np.mean(x_t)
    z_t[forecast * actual > 0] = 1
    p_hat = np.mean(z_t)
    p_star = p_y * p_x + (1 - p_y) * (1 - p_x)
    p_hat_var = (p_star * (1 - p_star)) / n
    p_star_var = ((2 * p_y - 1) ** 2 * (p_x * (1 - p_x))) / n + \
                 ((2 * p_x - 1) ** 2 * (p_y * (1 - p_y))) / n + \
                 (4 * p_x * p_y * (1 - p_x) * (1 - p_y)) / n ** 2
    
    # Handle edge cases to prevent warnings
    denominator = p_hat_var - p_star_var
    if denominator <= 0:
        # Edge case: Perfect prediction or degenerate case
        # Return a neutral statistic (0) if denominator is non-positive
        stat = 0.0
    else:
        stat = (p_hat - p_star) / np.sqrt(denominator)
    p_value = 1 - norm.cdf(stat)
    return p_hat, stat, p_value

def CW_test(actual, forecast_1, forecast_2):
    """
    Performs the Clark and West (2007) test to compare forecasts from nested models.
    Reference:
    [1] T.E. Clark and K.D. West (2007). "Approximately Normal Tests
    for Equal Predictive Accuracy in Nested Models." Journal of
    Econometrics 138, 291-311
    [2] He M, Zhang Y, Wen D, et al. Forecasting crude oil prices:
    A scaled PCA approach[J]. Energy Economics, 2021, 97: 105189.

    :param actual:  a column vector of actual values
    :param forecast_1:  a column vector of forecasts for restricted model
    :param forecast_2:  a column vector of forecasts for unrestricted model
    :return: a tuple of two elements, the first element is the MSPE_adjusted
    statistic, while the second one is the corresponding p-value
    """
    e_1 = actual - forecast_1
    e_2 = actual - forecast_2
    f_hat = np.square(e_1) - (np.square(e_2) - np.square(forecast_1 - forecast_2))
    Y_f = f_hat
    X_f = np.ones(f_hat.shape[0]).reshape(-1, 1)
    beta_f = np.linalg.inv(X_f.transpose() @ X_f) * (X_f.transpose() @ Y_f)
    e_f = Y_f - X_f * beta_f
    sig2_e = (e_f.transpose() @ e_f) / (Y_f.shape[0] - 1)
    cov_beta_f = sig2_e * np.linalg.inv(X_f.transpose() @ X_f)
    
    # Handle edge cases to prevent warnings
    # For matrix/array comparisons, we need to check if any/all elements meet the condition
    # In this case, we need to ensure covariance matrix is positive definite
    try:
        # Try to compute with a safety check on the covariance matrix
        if isinstance(cov_beta_f, np.ndarray) and cov_beta_f.size > 1:
            # For arrays/matrices, check if any element is non-positive
            if np.any(cov_beta_f <= 0):
                MSPE_adjusted = np.zeros_like(beta_f)
            else:
                MSPE_adjusted = beta_f / np.sqrt(cov_beta_f)
        else:
            # For scalar values, use direct comparison
            if cov_beta_f <= 0:
                MSPE_adjusted = np.zeros_like(beta_f)
            else:
                MSPE_adjusted = beta_f / np.sqrt(cov_beta_f)
    except:
        # If any issues with the computation, return neutral statistic
        MSPE_adjusted = np.zeros_like(beta_f)
    p_value = 1 - norm.cdf(MSPE_adjusted)
    return MSPE_adjusted[0][0], p_value[0][0]