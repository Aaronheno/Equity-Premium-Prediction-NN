import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

def scale_data(X, y):
    """
    Scales features (X) and target (y) using StandardScaler.

    Args:
        X (pd.DataFrame or np.ndarray): Feature data.
        y (pd.Series or pd.DataFrame or np.ndarray): Target data. # Allow DataFrame for y

    Returns:
        tuple: (X_scaled, y_scaled, scaler_x, scaler_y)
               Scaled data as numpy arrays and the fitted scalers.
    """
    # Ensure y is a 2D numpy array for scaler compatibility
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y_array = y.values.reshape(-1, 1)
    elif isinstance(y, np.ndarray): # If it's already a numpy array
        y_array = y.reshape(-1, 1) # Ensure it's 2D
    else:
        raise TypeError("Unsupported type for y. Expected pandas Series/DataFrame or numpy array.")


    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    # Fit and transform
    # Use .values if X is a DataFrame to get numpy array
    X_np = X.values if isinstance(X, pd.DataFrame) else X
    X_scaled = scaler_x.fit_transform(X_np)
    y_scaled = scaler_y.fit_transform(y_array) # Use the prepared y_array

    print("Data scaling complete.")
    return X_scaled, y_scaled, scaler_x, scaler_y 