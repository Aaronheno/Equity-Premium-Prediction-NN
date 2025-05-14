# src/utils/evaluation.py
import numpy as np
from sklearn.metrics import mean_squared_error

def compute_in_r_square(actual, benchmark, pred):
    """In sample R squared"""
    # Ensure inputs are numpy arrays
    actual = np.asarray(actual)
    benchmark = np.asarray(benchmark)
    pred = np.asarray(pred)
    return 1 - mean_squared_error(actual, pred) / mean_squared_error(actual, benchmark)

def compute_success_ratio(actual, pred):
    """Success ratio"""
    # Ensure inputs are numpy arrays
    actual = np.asarray(actual)
    pred = np.asarray(pred)
    return np.mean(np.sign(actual) == np.sign(pred))

def compute_CER(pred, rf, gamma=5, freq=12):
    """
    Certainty Equivalent Return (Annualized Percentage).
    Assumes pred and rf are decimal returns for the base period (e.g., monthly).
    freq is the number of periods per year (e.g., 12 for monthly).
    """
    # Ensure inputs are numpy arrays
    pred = np.asarray(pred).ravel() # Ensure 1D
    rf = np.asarray(rf).ravel()     # Ensure 1D

    if pred.shape != rf.shape:
        raise ValueError(f"Shapes of pred {pred.shape} and rf {rf.shape} must match after ravel()")
    if len(pred) == 0:
        return np.nan # Handle empty input

    rp = pred - rf  # Predicted excess return for the base period

    # Calculate statistics for the base period
    mu_p = np.mean(rp)
    sigma_p_sq = np.var(rp) # Use variance

    # Annualize (simple scaling approximation)
    mu_p_ann = mu_p * freq
    sigma_p_sq_ann = sigma_p_sq * freq

    # Calculate annualized CER and convert to percentage
    cer_ann_pct = (mu_p_ann - 0.5 * gamma * sigma_p_sq_ann) * 100
    return cer_ann_pct
