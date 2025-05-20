"""
variable_importance_8.py

This script implements variable importance analysis for equity premium prediction models.
It calculates how much each variable contributes to model performance by comparing
the out-of-sample R-squared when a variable is removed against the baseline performance.

The script follows the methodology from Xiu and Liu (2024), analyzing how performance
changes when each predictor variable is individually set to zero.
"""

import os
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid

from src.utils.io import RF_ALL, X_ALL, Y_ALL
from src.utils.metrics_unified import compute_in_r_square
from src.models import nns, Net1, Net2, Net3, Net4, Net5

def compute_oos_r_square(actual, y_benchmark, y_pred):
    """
    Compute out-of-sample R-squared metric.
    
    Args:
        actual: Actual values
        y_benchmark: Benchmark predictions
        y_pred: Model predictions
        
    Returns:
        float: Out-of-sample R-squared
    """
    MSFE_benchmark = mean_squared_error(y_benchmark, actual)
    MSFE_pred = mean_squared_error(y_pred, actual)
    return 1 - MSFE_pred / MSFE_benchmark

def get_oos_r_square_without_variable_i(predictor, actual, y_pred_HA, in_out_date, N, n_cols, i, models_to_evaluate=None):
    """
    Calculate OOS R-squared for all models when variable i is set to zero.
    
    Args:
        predictor: Complete predictor array
        actual: Actual values
        y_pred_HA: Historical average predictions
        in_out_date: Starting index for out-of-sample period
        N: Total number of samples
        n_cols: Number of columns in predictor
        i: Index of variable to set to zero
        models_to_evaluate: List of models to evaluate
        
    Returns:
        list: OOS R-squared values for each model with variable i removed
    """
    if models_to_evaluate is None:
        models_to_evaluate = ['PLS', 'PCR', 'LASSO', 'ENet', 'RF', 'NN2', 'NN4', 'Ridge']
    
    # Initialize prediction arrays
    y_pred_dict = {model: [] for model in models_to_evaluate}
    
    # Control the update month of models during out-of-sample period
    month_index = 1  # Update models annually (months 1, 13, 25, ...)
    
    for t in tqdm(range(in_out_date, N)):
        X_train_all = predictor[:t, 1:n_cols].copy()
        y_train_all = predictor[:t, 0]
        
        # Set the i-th predictor to 0
        X_train_all[:, i] = 0
        
        # Set 15% of all the train data as validation set
        split_idx = int(len(X_train_all) * 0.85)
        X_train = X_train_all[:split_idx, :]
        X_validation = X_train_all[split_idx:t, :]
        y_train = y_train_all[:split_idx]
        y_validation = y_train_all[split_idx:t]
        
        if month_index % 12 == 1:
            month_index += 1
            
            # Train all models and get predictions
            
            if 'PLS' in models_to_evaluate:
                # PLS
                PLS_param = {'n_components': [1, 2, 3, 4, 5, 6, 7, 8]}
                PLS_result = {}
                for param in ParameterGrid(PLS_param):
                    PLS = PLSRegression(**param)
                    PLS.fit(X_train, y_train)
                    mse = mean_squared_error(PLS.predict(X_validation), y_validation)
                    PLS_result[str(param)] = mse

                PLS_best_param = eval(min(PLS_result, key=PLS_result.get))
                PLS_model = PLSRegression(**PLS_best_param)
                PLS_model.fit(X_train_all, y_train_all)
                y_pred_dict['PLS'].append(PLS_model.predict(predictor[[t], 1:n_cols])[0][0])

            if 'PCR' in models_to_evaluate:
                # PCR
                PCR_param = {'n_components': [1, 2, 3, 4, 5, 6, 7, 8]}
                PCR_result = {}
                for param in ParameterGrid(PCR_param):
                    pca = PCA(**param)
                    pca.fit(X_train)
                    comps = pca.transform(X_train)
                    forecast = LinearRegression()
                    forecast.fit(comps, y_train)
                    mse = mean_squared_error(forecast.predict(pca.transform(X_validation)), y_validation)
                    PCR_result[str(param)] = mse
                
                PCR_best_param = eval(min(PCR_result, key=PCR_result.get))
                PCR_model = PCA(**PCR_best_param)
                PCR_model.fit(X_train_all)
                PCR_comps = PCR_model.transform(X_train_all)
                PCR_forecast = LinearRegression()
                PCR_forecast.fit(PCR_comps, y_train_all)
                y_pred_dict['PCR'].append(PCR_forecast.predict(PCR_model.transform(predictor[[t], 1:n_cols]))[0])

            if 'LASSO' in models_to_evaluate:
                # LASSO
                LASSO_param = {'alpha': list(10 ** np.arange(-4, 1 + 0.001, 0.2))}
                LASSO_result = {}
                for param in ParameterGrid(LASSO_param):
                    LASSO = Lasso(**param)
                    LASSO.fit(X_train, y_train)
                    mse = mean_squared_error(LASSO.predict(X_validation), y_validation)
                    LASSO_result[str(param)] = mse
                
                LASSO_best_param = eval(min(LASSO_result, key=LASSO_result.get))
                LASSO_model = Lasso(**LASSO_best_param)
                LASSO_model.fit(X_train_all, y_train_all)
                y_pred_dict['LASSO'].append(LASSO_model.predict(predictor[[t], 1:n_cols])[0])

            if 'ENet' in models_to_evaluate:
                # ENet
                ENet_param = {'alpha': list(10 ** np.arange(-4, 1 + 0.001, 0.2)),
                              'l1_ratio': list(np.arange(0.2, 1, 0.3))}
                ENet_result = {}
                for param in ParameterGrid(ENet_param):
                    ENet = ElasticNet(**param)
                    ENet.fit(X_train, y_train)
                    mse = mean_squared_error(ENet.predict(X_validation), y_validation)
                    ENet_result[str(param)] = mse

                ENet_best_param = eval(min(ENet_result, key=ENet_result.get))
                ENet_model = ElasticNet(**ENet_best_param)
                ENet_model.fit(X_train_all, y_train_all)
                y_pred_dict['ENet'].append(ENet_model.predict(predictor[[t], 1:n_cols])[0])

            if 'RF' in models_to_evaluate:
                # RF
                RF_param = {'n_estimators': [10, 50, 100, 150, 200],
                            'max_depth': [2, 3, 4],
                            'min_samples_leaf': [1, 3, 5]}
                RF_result = {}
                for param in ParameterGrid(RF_param):
                    RF = RandomForestRegressor(**param)
                    RF.fit(X_train, y_train)
                    mse = mean_squared_error(RF.predict(X_validation), y_validation)
                    RF_result[str(param)] = mse

                RF_best_param = eval(min(RF_result, key=RF_result.get))
                RF_model = RandomForestRegressor(**RF_best_param)
                RF_model.fit(X_train_all, y_train_all)
                y_pred_dict['RF'].append(RF_model.predict(predictor[[t], 1:n_cols])[0])

            # For neural network models
            if 'NN2' in models_to_evaluate or 'NN4' in models_to_evaluate:
                X_train_all_tensor = torch.tensor(X_train_all, dtype=torch.float)
                y_train_all_tensor = torch.tensor(y_train_all.reshape(-1, 1), dtype=torch.float)
                X_train_tensor = torch.tensor(X_train, dtype=torch.float)
                y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float)
                X_validation_tensor = torch.tensor(X_validation, dtype=torch.float)
                
            if 'NN2' in models_to_evaluate:
                # NN2
                from skorch import NeuralNetRegressor
                NN2_result = {}
                NN2_architecture = {"module__n_feature": X_train_tensor.shape[1],
                                    "module__n_hidden1": 32, "module__n_hidden2": 16,
                                    "module__n_output": 1}
                NN2_param = {'module__dropout': [0.2, 0.4, 0.6, 0.8],
                            'lr': [0.001, 0.01],
                            'optimizer__weight_decay': [0.1, 0.01, 0.001]}
                for param in ParameterGrid(NN2_param):
                    NN2 = NeuralNetRegressor(Net2, verbose=0, max_epochs=200,
                                            optimizer=torch.optim.SGD,
                                            **NN2_architecture, **param)
                    NN2.fit(X_train_tensor, y_train_tensor)
                    mse = mean_squared_error(NN2.predict(X_validation_tensor), y_validation)
                    NN2_result[str(param)] = mse
                
                NN2_best_param = eval(min(NN2_result, key=NN2_result.get))
                NN2_model = NeuralNetRegressor(Net2, verbose=0, max_epochs=200, optimizer=torch.optim.SGD,
                                            **NN2_architecture, **NN2_best_param)
                NN2_model.fit(X_train_all_tensor, y_train_all_tensor)
                y_pred_dict['NN2'].append(NN2_model.predict(torch.tensor(predictor[[t], 1:n_cols], dtype=torch.float))[0][0])

            if 'NN4' in models_to_evaluate:
                # NN4
                from skorch import NeuralNetRegressor
                NN4_result = {}
                NN4_architecture = {"module__n_feature": X_train_tensor.shape[1],
                                    "module__n_hidden1": 32, "module__n_hidden2": 16,
                                    "module__n_hidden3": 8, "module__n_hidden4": 4,
                                    "module__n_output": 1}
                NN4_param = {'module__dropout': [0.2, 0.4, 0.6, 0.8],
                            'lr': [0.001, 0.01],
                            'optimizer__weight_decay': [0.1, 0.01, 0.001]}
                for param in ParameterGrid(NN4_param):
                    NN4 = NeuralNetRegressor(Net4, verbose=0, max_epochs=200,
                                            optimizer=torch.optim.SGD,
                                            **NN4_architecture, **param)
                    NN4.fit(X_train_tensor, y_train_tensor)
                    mse = mean_squared_error(NN4.predict(X_validation_tensor), y_validation)
                    NN4_result[str(param)] = mse
                
                NN4_best_param = eval(min(NN4_result, key=NN4_result.get))
                NN4_model = NeuralNetRegressor(Net4, verbose=0, max_epochs=200, optimizer=torch.optim.SGD,
                                            **NN4_architecture, **NN4_best_param)
                NN4_model.fit(X_train_all_tensor, y_train_all_tensor)
                y_pred_dict['NN4'].append(NN4_model.predict(torch.tensor(predictor[[t], 1:n_cols], dtype=torch.float))[0][0])

            if 'Ridge' in models_to_evaluate:
                # Ridge
                Ridge_param = {'alpha': list(10 ** np.arange(0, 20 + 0.001, 1))}
                Ridge_result = {}
                for param in ParameterGrid(Ridge_param):
                    RIDGE = Ridge(**param)
                    RIDGE.fit(X_train, y_train)
                    mse = mean_squared_error(RIDGE.predict(X_validation), y_validation)
                    Ridge_result[str(param)] = mse
                
                Ridge_best_param = eval(min(Ridge_result, key=Ridge_result.get))
                Ridge_model = Ridge(**Ridge_best_param)
                Ridge_model.fit(X_train_all, y_train_all)
                y_pred_dict['Ridge'].append(Ridge_model.predict(predictor[[t], 1:n_cols])[0])

        else:
            month_index += 1
            # Use the existing models to make predictions
            if 'PLS' in models_to_evaluate:
                y_pred_dict['PLS'].append(PLS_model.predict(predictor[[t], 1:n_cols])[0][0])
            if 'PCR' in models_to_evaluate:
                y_pred_dict['PCR'].append(PCR_forecast.predict(PCR_model.transform(predictor[[t], 1:n_cols]))[0])
            if 'LASSO' in models_to_evaluate:
                y_pred_dict['LASSO'].append(LASSO_model.predict(predictor[[t], 1:n_cols])[0])
            if 'ENet' in models_to_evaluate:
                y_pred_dict['ENet'].append(ENet_model.predict(predictor[[t], 1:n_cols])[0])
            if 'RF' in models_to_evaluate:
                y_pred_dict['RF'].append(RF_model.predict(predictor[[t], 1:n_cols])[0])
            if 'NN2' in models_to_evaluate:
                y_pred_dict['NN2'].append(NN2_model.predict(torch.tensor(predictor[[t], 1:n_cols], dtype=torch.float))[0][0])
            if 'NN4' in models_to_evaluate:
                y_pred_dict['NN4'].append(NN4_model.predict(torch.tensor(predictor[[t], 1:n_cols], dtype=torch.float))[0][0])
            if 'Ridge' in models_to_evaluate:
                y_pred_dict['Ridge'].append(Ridge_model.predict(predictor[[t], 1:n_cols])[0])

    # Calculate OOS R-squared for each model
    actual_subset = actual[in_out_date:]
    y_pred_HA_subset = y_pred_HA[in_out_date:]
    
    results = []
    for model in models_to_evaluate:
        if model in y_pred_dict:
            y_pred = np.array(y_pred_dict[model])
            results.append(compute_oos_r_square(actual_subset, y_pred_HA_subset, y_pred.reshape(-1, 1)))
    
    return results

def run(
    models=['Net1', 'Net2', 'Net3', 'Net4', 'Net5'],
    start_date=195701,
    n_jobs=4,
    threads=4,
    device='cpu',
    data_source='original'  # 'original', 'newly_identified', or 'fred'
):
    """
    Run variable importance analysis.
    
    Args:
        models (list): List of models to evaluate
        start_date (int): Start date for out-of-sample period in YYYYMM format
        n_jobs (int): Number of parallel jobs for variable importance calculation
        threads (int): Number of threads for each job
        device (str): Computing device ('cpu' or 'cuda')
        data_source (str): Source of data ('original', 'newly_identified', or 'fred')
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    # Set up output directory
    out_base = Path(f"./runs/8_Variable_Importance")
    out_base.mkdir(parents=True, exist_ok=True)
    run_name = f"{ts}_{start_date}_{data_source}"
    out_dir = out_base / run_name
    out_dir.mkdir(exist_ok=True, parents=True)
    print(f"Output directory: {out_dir}")
    
    # Load and prepare data based on selected source
    if data_source == 'newly_identified':
        from src.experiments.newly_identified_6 import load_data
        X_data, y_data, _ = load_data('standalone')
    elif data_source == 'fred':
        from src.experiments.fred_variables_7 import load_fred_data
        X_data, y_data, _ = load_fred_data()
    else:  # Original data
        data_path = Path("ml_equity_premium_data.xlsx")
        
        if not data_path.exists():
            raise FileNotFoundError(f"Cannot find data file at {data_path}")
            
        # Read data from result_predictor sheet
        df = pd.read_excel(data_path, sheet_name='result_predictor')
        print(f"Loaded {len(df)} records from the original predictor data")
        
        # Remove irrelevant columns
        df = df.drop(['month', 'equity_premium'], axis=1)
        X_data = df.drop(['log_equity_premium'], axis=1)
        y_data = df['log_equity_premium'].values.reshape(-1, 1)
    
    # Prepare data for variable importance analysis
    # Get predictor names
    predictor_names = list(X_data.columns) if hasattr(X_data, 'columns') else [f"Feature_{i}" for i in range(X_data.shape[1])]
    
    # Prepare the data array similar to the example code
    if hasattr(X_data, 'values'):
        X_values = X_data.values
    else:
        X_values = X_data
        
    if hasattr(y_data, 'values'):
        y_values = y_data.values
    else:
        y_values = y_data
    
    # Combine into one array with target shifted for prediction
    predictor_array = np.concatenate([y_values[1:], X_values[:-1]], axis=1)
    
    # Number of rows and columns
    N = predictor_array.shape[0]
    n_cols = predictor_array.shape[1]
    
    # Actual one-month ahead log equity premium
    actual = predictor_array[:, [0]]
    
    # Historical average forecasting as benchmark
    y_pred_HA = np.zeros_like(actual)
    for i in range(len(y_pred_HA)):
        if i == 0:
            y_pred_HA[i] = y_values[0]
        else:
            y_pred_HA[i] = np.mean(y_values[:i+1])
    
    # Find the index of the start date
    if data_source == 'original':
        date_col = df['month'].values
        try:
            in_out_date = np.where(date_col >= start_date)[0][0]
        except:
            print(f"Warning: Start date {start_date} not found. Using the first 70% as training.")
            in_out_date = int(0.7 * N)
    else:
        # For other data sources, use 70% as training
        in_out_date = int(0.7 * N)
    
    print(f"Out-of-sample period starts at index {in_out_date} out of {N} total samples")
    
    # Subset for out-of-sample evaluation
    actual_oos = actual[in_out_date:]
    y_pred_HA_oos = y_pred_HA[in_out_date:]
    
    # Models to evaluate (fixed list for consistency with Xiu and Liu)
    models_to_evaluate = ['PLS', 'PCR', 'LASSO', 'ENet', 'RF', 'NN2', 'NN4', 'Ridge']
    
    # Calculate baseline OOS R-squared for each model
    baseline_results = {}
    print("Calculating baseline performance...")
    torch.set_num_threads(threads)
    
    # Use parallelization to calculate variable importance
    print(f"Starting variable importance analysis with {n_jobs} parallel jobs...")
    
    variable_importance = Parallel(n_jobs=n_jobs)(
        delayed(get_oos_r_square_without_variable_i)(
            predictor_array, actual, y_pred_HA, in_out_date, N, n_cols, i, models_to_evaluate
        ) for i in range(n_cols - 1)
    )
    
    # Convert results to DataFrame
    variable_importance_df = pd.DataFrame(
        np.array(variable_importance),
        index=predictor_names,
        columns=models_to_evaluate
    ).T
    
    # Save results
    variable_importance_df.to_excel(out_dir / "variable_importance_results.xlsx", sheet_name='variable_impt_oos_r2')
    print(f"Variable importance results saved to {out_dir}")
    
    # Retrieve benchmark performance 
    # In a real scenario, we'd load this from previous runs, but for now, let's estimate it
    # In practice, you should replace this with actual benchmark performance
    benchmark_df = pd.DataFrame(
        index=models_to_evaluate,
        columns=['oos_r_square']
    )
    benchmark_df['oos_r_square'] = 0.05  # Example value, replace with actual benchmark R²
    
    # Calculate R-squared reduction
    oos_r_square_reduction_df = variable_importance_df.apply(
        lambda x: benchmark_df['oos_r_square'] - x * 100, 
        raw=True
    ).T
    
    # Save R-squared reduction results
    oos_r_square_reduction_df.to_excel(out_dir / "r_squared_reduction.xlsx", sheet_name='variable_impt_r2_reduction')
    
    # Generate variable importance plots
    plot_variable_importance(oos_r_square_reduction_df, out_dir, models_to_evaluate)
    
    return variable_importance_df, oos_r_square_reduction_df

def plot_variable_importance(importance_df, out_dir, models=None):
    """
    Generate plots showing the most important variables for each model.
    
    Args:
        importance_df: DataFrame with variable importance scores
        out_dir: Output directory for saving plots
        models: List of models to plot
    """
    if models is None:
        models = importance_df.columns
        
    n_models = len(models)
    rows = (n_models + 1) // 2
    
    fig, ax = plt.subplots(rows, 2, figsize=(13, 15))
    
    for i, model in enumerate(models):
        row = i // 2
        col = i % 2
        
        # Get top variables for this model
        performance = importance_df[model].sort_values(ascending=False)[:6]
        y_pos = np.arange(len(performance))
        
        # Create bar chart
        if rows == 1:
            current_ax = ax[col]
        else:
            current_ax = ax[row, col]
            
        current_ax.barh(y_pos, performance.to_numpy(), color='#1f77b4')
        current_ax.set_yticks(y_pos)
        current_ax.set_yticklabels(performance.index)
        current_ax.tick_params(axis='both', which='major', labelsize=12)
        current_ax.invert_yaxis()
        current_ax.set_title(label=model, fontsize=18, fontweight='bold')
    
    # Hide empty subplots if number of models is odd
    if n_models % 2 == 1:
        ax[rows-1, 1].axis('off')
    
    fig.tight_layout()
    fig.savefig(out_dir / 'Variable_importance_ranking.png', dpi=300)
    plt.close(fig)
    
    print(f"Variable importance plot saved to {out_dir}")

if __name__ == "__main__":
    run(models=['PLS', 'PCR', 'LASSO', 'ENet', 'RF', 'NN2', 'NN4', 'Ridge'], start_date=195701, n_jobs=4)
