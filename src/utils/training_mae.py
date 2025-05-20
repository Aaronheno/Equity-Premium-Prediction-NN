"""
Hyperparameter optimization utilities using Mean Absolute Error (MAE) as the scoring function.

This module contains functions for grid search, random search, and Bayesian optimization
with MAE as the validation metric.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import ParameterGrid
import optuna
import random
import itertools
from tqdm import tqdm

def run_grid_search_mae(X_train, y_train, X_val, y_val, model_class, param_grid, 
                     epochs=30, batch_size=256, device='cpu'):
    """
    Perform grid search for hyperparameter optimization using MAE as scoring function.
    
    Parameters:
    -----------
    X_train : array-like
        Training feature data
    y_train : array-like
        Training target data
    X_val : array-like
        Validation feature data
    y_val : array-like
        Validation target data
    model_class : class
        Neural network model class
    param_grid : dict
        Dictionary with hyperparameters
    epochs : int
        Number of epochs to train
    batch_size : int
        Batch size for training
    device : str
        Device to use ('cpu' or 'cuda')
        
    Returns:
    --------
    best_params : dict
        Best hyperparameters based on validation MAE
    """
    print(f"Running grid search with MAE scoring (epochs={epochs}, batch_size={batch_size}, device={device})")
    
    # Generate all parameter combinations
    param_combinations = list(ParameterGrid(param_grid))
    print(f"Total combinations to evaluate: {len(param_combinations)}")
    
    best_mae = float('inf')
    best_params = None
    
    # Convert data to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    
    # Move data to device
    if device == 'cuda' and torch.cuda.is_available():
        X_train_tensor = X_train_tensor.cuda()
        y_train_tensor = y_train_tensor.cuda()
        X_val_tensor = X_val_tensor.cuda()
        y_val_tensor = y_val_tensor.cuda()
    
    # Evaluate each parameter combination
    for i, params in enumerate(tqdm(param_combinations, desc="Grid Search")):
        # Create and configure model
        model_params = {'n_feature': X_train.shape[1], **params}
        model = model_class(**model_params)
        
        # Move model to device
        if device == 'cuda' and torch.cuda.is_available():
            model = model.cuda()
        
        # Define loss function and optimizer
        criterion = nn.L1Loss()  # Using L1Loss for MAE optimization
        optimizer = torch.optim.Adam(model.parameters(), lr=params.get('lr', 0.001))
        
        # Train the model
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
        
        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_outputs_np = val_outputs.cpu().numpy()
            y_val_np = y_val_tensor.cpu().numpy()
            val_mae = mean_absolute_error(y_val_np, val_outputs_np)
        
        # Update best parameters if this combination is better
        if val_mae < best_mae:
            best_mae = val_mae
            best_params = params
            print(f"New best MAE: {best_mae:.6f}, params: {best_params}")
    
    print(f"Grid search complete. Best MAE: {best_mae:.6f}")
    print(f"Best parameters: {best_params}")
    
    return best_params


def run_random_search_mae(X_train, y_train, X_val, y_val, model_class, param_space, 
                       epochs=30, n_trials=10, batch_size=256, device='cpu'):
    """
    Perform random search for hyperparameter optimization using MAE as scoring function.
    
    Parameters:
    -----------
    X_train : array-like
        Training feature data
    y_train : array-like
        Training target data
    X_val : array-like
        Validation feature data
    y_val : array-like
        Validation target data
    model_class : class
        Neural network model class
    param_space : dict
        Dictionary with hyperparameter ranges
    epochs : int
        Number of epochs to train
    n_trials : int
        Number of random combinations to try
    batch_size : int
        Batch size for training
    device : str
        Device to use ('cpu' or 'cuda')
        
    Returns:
    --------
    best_params : dict
        Best hyperparameters based on validation MAE
    """
    print(f"Running random search with MAE scoring (trials={n_trials}, epochs={epochs}, device={device})")
    
    best_mae = float('inf')
    best_params = None
    
    # Convert data to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    
    # Move data to device
    if device == 'cuda' and torch.cuda.is_available():
        X_train_tensor = X_train_tensor.cuda()
        y_train_tensor = y_train_tensor.cuda()
        X_val_tensor = X_val_tensor.cuda()
        y_val_tensor = y_val_tensor.cuda()
    
    # Generate random parameter combinations
    for trial in tqdm(range(n_trials), desc="Random Search"):
        # Sample random parameters
        params = {}
        for param_name, param_range in param_space.items():
            if isinstance(param_range, list):
                params[param_name] = random.choice(param_range)
            elif isinstance(param_range, tuple) and len(param_range) == 2:
                # Assume continuous range (min_val, max_val)
                min_val, max_val = param_range
                if isinstance(min_val, int) and isinstance(max_val, int):
                    # Integer parameter
                    params[param_name] = random.randint(min_val, max_val)
                else:
                    # Float parameter
                    params[param_name] = random.uniform(min_val, max_val)
        
        # Create and configure model
        model_params = {'n_feature': X_train.shape[1], **params}
        model = model_class(**model_params)
        
        # Move model to device
        if device == 'cuda' and torch.cuda.is_available():
            model = model.cuda()
        
        # Define loss function and optimizer
        criterion = nn.L1Loss()  # Using L1Loss for MAE optimization
        optimizer = torch.optim.Adam(model.parameters(), lr=params.get('lr', 0.001))
        
        # Train the model
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
        
        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_outputs_np = val_outputs.cpu().numpy()
            y_val_np = y_val_tensor.cpu().numpy()
            val_mae = mean_absolute_error(y_val_np, val_outputs_np)
        
        # Update best parameters if this combination is better
        if val_mae < best_mae:
            best_mae = val_mae
            best_params = params
            print(f"New best MAE: {best_mae:.6f}, params: {best_params}")
    
    print(f"Random search complete. Best MAE: {best_mae:.6f}")
    print(f"Best parameters: {best_params}")
    
    return best_params


def objective_mae(trial, X_train, y_train, X_val, y_val, model_class, param_space, epochs, device):
    """
    Objective function for Bayesian optimization using MAE as the metric.
    
    Parameters:
    -----------
    trial : optuna.Trial
        Optuna trial object
    X_train, y_train, X_val, y_val : array-like
        Training and validation data
    model_class : class
        Neural network model class
    param_space : dict
        Dictionary with hyperparameter ranges for Bayesian optimization
    epochs : int
        Number of training epochs
    device : str
        Device to train on ('cpu' or 'cuda')
        
    Returns:
    --------
    val_mae : float
        Validation mean absolute error (to be minimized)
    """
    # Sample parameters from the parameter space
    params = {}
    for param_name, param_range in param_space.items():
        if isinstance(param_range, list):
            params[param_name] = trial.suggest_categorical(param_name, param_range)
        elif isinstance(param_range, tuple) and len(param_range) == 2:
            min_val, max_val = param_range
            if isinstance(min_val, int) and isinstance(max_val, int):
                params[param_name] = trial.suggest_int(param_name, min_val, max_val)
            else:
                if param_name == 'lr':  # Log scale for learning rate
                    params[param_name] = trial.suggest_float(param_name, min_val, max_val, log=True)
                else:
                    params[param_name] = trial.suggest_float(param_name, min_val, max_val)
    
    # Create and configure model
    model_params = {'n_feature': X_train.shape[1], **params}
    model = model_class(**model_params)
    
    # Convert data to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    
    # Move data and model to device
    if device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
        X_train_tensor = X_train_tensor.cuda()
        y_train_tensor = y_train_tensor.cuda()
        X_val_tensor = X_val_tensor.cuda()
        y_val_tensor = y_val_tensor.cuda()
    
    # Define loss function and optimizer
    criterion = nn.L1Loss()  # Using L1Loss for MAE optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=params.get('lr', 0.001))
    
    # Train the model
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
    
    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_outputs_np = val_outputs.cpu().numpy()
        y_val_np = y_val_tensor.cpu().numpy()
        val_mae = mean_absolute_error(y_val_np, val_outputs_np)
    
    return val_mae


def run_bayesian_optimization_mae(X_train, y_train, X_val, y_val, model_class, param_space, 
                               epochs=30, n_trials=10, batch_size=256, device='cpu'):
    """
    Perform Bayesian optimization for hyperparameter tuning using MAE as scoring function.
    
    Parameters:
    -----------
    X_train, y_train, X_val, y_val : array-like
        Training and validation data
    model_class : class
        Neural network model class
    param_space : dict
        Dictionary with hyperparameter ranges for Bayesian optimization
    epochs : int
        Number of training epochs
    n_trials : int
        Number of optimization trials
    batch_size : int
        Batch size for training
    device : str
        Device to train on ('cpu' or 'cuda')
        
    Returns:
    --------
    best_params : dict
        Best hyperparameters based on validation MAE
    """
    print(f"Running Bayesian optimization with MAE scoring (trials={n_trials}, epochs={epochs}, device={device})")
    
    # Create a study object and optimize the objective function
    study = optuna.create_study(direction='minimize')
    objective = lambda trial: objective_mae(
        trial, X_train, y_train, X_val, y_val, model_class, param_space, epochs, device
    )
    
    study.optimize(objective, n_trials=n_trials)
    
    # Get best parameters
    best_params = study.best_params
    best_mae = study.best_value
    
    print(f"Bayesian optimization complete. Best MAE: {best_mae:.6f}")
    print(f"Best parameters: {best_params}")
    
    return best_params
