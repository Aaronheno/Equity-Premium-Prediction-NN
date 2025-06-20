"""
Bayesian Out-of-Sample Neural Network Optimization

This experiment conducts out-of-sample evaluation using Bayesian hyperparameter
optimization (Optuna) for neural network models in equity premium prediction.
Designed for massive parallelization with support for concurrent model training.

Threading Status: PARALLEL_READY (Model-level and HPO-level parallelism)
Hardware Requirements: CPU_REQUIRED, CUDA_PREFERRED, HIGH_MEMORY_BENEFICIAL  
Performance Notes:
    - Model parallelism: 8x speedup opportunity (8 models simultaneously)
    - HPO parallelism: 10-50x speedup (100+ concurrent trials)
    - Memory scaling: Linear with model count and trial count
    - Optimal for 32+ core systems

Experiment Type: Out-of-Sample Evaluation with Annual HPO
Models Supported: Net1, Net2, Net3, Net4, Net5, DNet1, DNet2, DNet3
HPO Method: Bayesian Optimization (Optuna TPE)
Output Directory: runs/1_Bayes_Search_OOS/

Critical Parallelization Opportunities:
    1. Parallel model HPO within each OOS time step
    2. Concurrent model training after HPO completion  
    3. Parallel prediction generation across models
    4. Concurrent metrics computation

Threading Implementation Status:
    ❌ Sequential model processing (MAIN BOTTLENECK)
    ✅ HPO trials can be parallelized via Optuna n_jobs
    ❌ Model training sequential within time steps
    ❌ Prediction generation sequential

Future Parallel Implementation:
    run(models, parallel_models=True, hpo_parallel=True)
    
Expected Performance Gains:
    - Current: 8 hours for 8 models × 200 time steps  
    - With parallelization: 1-2 hours (4-8x speedup)
    - With HPO parallelism: 30-60 minutes (additional 2-4x speedup)
"""

# src/experiments/bayes_oos.py
import sys
from pathlib import Path
import torch

# --- Add project root to sys.path ---
# This ensures Python can find the 'src' package when running the script.
# It assumes this script is located at src/experiments/bayes_oos.py
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
# --- End sys.path modification ---

from src.models import nns
from src.configs.search_spaces import BAYES_OOS # Expecting BAYES_OOS in search_spaces.py
from src.utils.training_optuna import run_study as optuna_hpo_runner_function, OptunaSkorchNet # HPO runner and Skorch wrapper
from src.utils.oos_common import run_oos_experiment, OOS_DEFAULT_START_YEAR_MONTH
from src.utils.load_models import get_model_class_from_name # Helper

# Define a mapping from model names (strings) to their classes and HPO details
# This should align with what's available in BAYES_OOS
ALL_NN_MODEL_CONFIGS_BAYES_OOS = {
    model_name: {
        "model_class": get_model_class_from_name(model_name),
        "hpo_function": optuna_hpo_runner_function, # This is run_study
        "regressor_class": OptunaSkorchNet, # Use the Skorch wrapper for Optuna
        "search_space_config_or_fn": BAYES_OOS.get(model_name, {}).get("hpo_config_fn") # Get the hpo_config_fn
    }
    for model_name in ["Net1", "Net2", "Net3", "Net4", "Net5", "DNet1", "DNet2", "DNet3"]
    if BAYES_OOS.get(model_name, {}).get("hpo_config_fn") is not None # Only include if config fn exists
}

def run(
    model_names, # <<< ADD model_names HERE
    oos_start_date_int,
    hpo_general_config, # Dict: {'hpo_epochs': E, 'hpo_trials': T, 'hpo_device': D, 'hpo_batch_size': B}
    save_annual_models=False,
    parallel_models=False,  # <<< ADD: Enable model-level parallelism for 32-core server
    verbose=False
):
    """
    Runs the Out-of-Sample (OOS) experiment using Bayesian Optimization (Optuna) for HPO.
    """
    experiment_name_suffix = "bayes_opt_oos"
    base_run_folder_name = "1_Bayes_Search_OOS"

    # Filter ALL_NN_MODEL_CONFIGS_BAYES_OOS based on model_names from CLI
    if not model_names:
        print("No models specified to run. Exiting.")
        return

    nn_model_configs_to_run = {
        name: config for name, config in ALL_NN_MODEL_CONFIGS_BAYES_OOS.items()
        if name in model_names
    }
    
    if not nn_model_configs_to_run:
        print(f"None of the specified models ({model_names}) have configurations in BAYES_OOS. Exiting.")
        return

    print(f"--- Running Bayesian OOS for models: {list(nn_model_configs_to_run.keys())} ---")

    run_oos_experiment(
        experiment_name_suffix=experiment_name_suffix,
        base_run_folder_name=base_run_folder_name,
        nn_model_configs=nn_model_configs_to_run, # Pass the filtered configs
        hpo_general_config=hpo_general_config,
        oos_start_date_int=oos_start_date_int,
        save_annual_models=save_annual_models,
        parallel_models=parallel_models  # <<< ADD: Pass through parallel_models flag
    )

if __name__ == '__main__':
    # This section allows direct execution of this script for testing.
    # Ensure src.configs.search_spaces.BAYES_OOS is defined correctly.
    print("--- Running bayes_oos.py directly for testing purposes ---")
    
    # For a quick test, run fewer trials and epochs for a single model
    test_models = ["Net1"] 
    test_trials = 5      # Small number of Optuna trials for a quick test
    test_epochs = 10     # Small number of epochs for a quick test
    test_batch_size = 64

    # Determine device for testing
    test_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Test device: {test_device}")

    # Before running, ensure BAYES_OOS is defined in src/configs/search_spaces.py
    # Example structure for BAYES_OOS["Net1"]["hpo_config_fn"] in search_spaces.py:
    # def net1_optuna_config(trial, n_features):
    #     return {
    #         'module__n_hidden1': trial.suggest_categorical('module__n_hidden1', [16, 32, 64, 128]),
    #         'module__dropout': trial.suggest_float('module__dropout', 0.0, 0.6, step=0.1),
    #         'module__activation_hidden': trial.suggest_categorical('module__activation_hidden', ['relu', 'tanh']),
    #         'optimizer': trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop']),
    #         'lr': trial.suggest_float('lr', 1e-4, 1e-1, log=True),
    #         'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True), # L2
    #         'l1_lambda': trial.suggest_float('l1_lambda', 1e-6, 1e-2, log=True),      # L1
    #         'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256])
    #     }
    # BAYES_OOS = { "Net1": { "hpo_config_fn": net1_optuna_config }, ... }

    # Define hpo_general_config for testing
    test_hpo_general_config = {
        "hpo_epochs": 5,
        "hpo_trials": 3, # Specific to Random/Bayes
        "hpo_device": test_device,
        "hpo_batch_size": test_batch_size
    }
    print(f"Test HPO Config: {test_hpo_general_config}")

    ready_to_test = True
    for model_name_test in test_models:
        if model_name_test not in BAYES_OOS or not BAYES_OOS[model_name_test].get("hpo_config_fn"):
            print(f"Error: BAYES_OOS['{model_name_test}']['hpo_config_fn'] is not defined. Cannot run test.", file=sys.stderr)
            ready_to_test = False
            break
            
    if ready_to_test:
        run(
            model_names=test_models,
            oos_start_date_int=OOS_DEFAULT_START_YEAR_MONTH,
            hpo_general_config=test_hpo_general_config,
            save_annual_models=False
        )
    else:
        print("Please define hpo_config_fn for test models in BAYES_OOS for testing.", file=sys.stderr)
        
    print("--- Test run of bayes_oos.py finished ---")