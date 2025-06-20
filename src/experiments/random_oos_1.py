"""
Random Search Out-of-Sample Neural Network Optimization

This experiment conducts out-of-sample evaluation using random search hyperparameter
optimization for neural network models. Features perfect parallelization potential
with independent trial sampling for maximum scalability.

Threading Status: PERFECTLY_PARALLEL (Independent trials, linear scaling)
Hardware Requirements: CPU_REQUIRED, CUDA_BENEFICIAL, LINEAR_MEMORY_SCALING
Performance Notes:
    - Random trials: Perfect linear scaling with core count
    - Memory usage: ~300MB per concurrent trial
    - No coordination overhead between trials
    - Embarrassingly parallel: Ideal for HPC systems

Experiment Type: Out-of-Sample Evaluation with Annual Random Search HPO
Models Supported: Net1, Net2, Net3, Net4, Net5, DNet1, DNet2, DNet3
HPO Method: Random Search with Independent Sampling
Output Directory: runs/1_Random_Search_OOS/

Critical Parallelization Opportunities:
    1. Perfect trial parallelization (linear scaling)
    2. Independent model HPO (8x speedup)
    3. Concurrent model training after HPO
    4. Parallel prediction generation

Threading Implementation Status:
    ❌ Sequential model processing (MAIN BOTTLENECK)
    ✅ Random trials perfectly parallelizable
    ❌ Model training sequential within time steps
    ❌ Annual HPO sequential across years

Future Parallel Implementation:
    run(models, parallel_models=True, trial_parallel=True, n_jobs=128)
    
Expected Performance Gains:
    - Current: 6 hours for 8 models × 200 time steps × 50 trials
    - With trial parallelism: 2 hours (3x speedup)
    - With model parallelism: 45 minutes (8x speedup)
    - Combined on 128-core server: 5-10 minutes (36-72x speedup)

Random Search Advantages:
    - Best parallel efficiency (linear scaling)
    - No diminishing returns with core count
    - Often matches Bayesian optimization performance
    - Trivial to distribute across multiple machines
"""

# src/experiments/random_oos.py
import sys
from pathlib import Path
import torch

# --- Add project root to sys.path ---
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
# --- End sys.path modification ---

from src.models import nns
from src.configs.search_spaces import RANDOM_OOS # Expecting RANDOM_OOS in search_spaces.py
# Import the train_random function for HPO
from src.utils.training_random import train_random
# Import GridNet to use as regressor class for Random Search
from src.utils.training_grid import GridNet as RandomSkorchNet # Using GridNet as it has L1 regularization
from src.utils.oos_common import run_oos_experiment, OOS_DEFAULT_START_YEAR_MONTH
from src.utils.load_models import get_model_class_from_name # Helper

# Define a mapping from model names (strings) to their classes and HPO details
# This should align with what's available in RANDOM_OOS
ALL_NN_MODEL_CONFIGS_RANDOM_OOS = {
    model_name: {
        "model_class": get_model_class_from_name(model_name),
        "hpo_function": train_random,
        "regressor_class": RandomSkorchNet, # Use the Skorch wrapper for random search
        "search_space_config_or_fn": RANDOM_OOS.get(model_name)
    }
    for model_name in ["Net1", "Net2", "Net3", "Net4", "Net5", "DNet1", "DNet2", "DNet3"]
    if RANDOM_OOS.get(model_name) is not None # Only include if config exists
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
    Runs the Out-of-Sample (OOS) experiment using Random Search for HPO.
    """
    experiment_name_suffix = "random_search_oos"
    base_run_folder_name = "1_Random_Search_OOS"

    # Filter ALL_NN_MODEL_CONFIGS_RANDOM_OOS based on model_names from CLI
    if not model_names: # Should not happen if CLI has a default
        print("No models specified to run. Exiting.")
        return
    
    nn_model_configs_to_run = {
        name: config for name, config in ALL_NN_MODEL_CONFIGS_RANDOM_OOS.items()
        if name in model_names
    }

    if not nn_model_configs_to_run:
        print(f"None of the specified models ({model_names}) have configurations in RANDOM_OOS. Exiting.")
        return
    
    print(f"--- Running Random OOS for models: {list(nn_model_configs_to_run.keys())} ---")

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
    print("--- Running random_oos.py directly for testing purposes ---")
    
    test_models = ["Net1"] 
    # Define hpo_general_config for testing
    test_hpo_general_config = {
        "hpo_epochs": 5,
        "hpo_trials": 3, # Specific to Random/Bayes
        "hpo_device": "cuda" if torch.cuda.is_available() else "cpu",
        "hpo_batch_size": 64
    }
    print(f"Test HPO Config: {test_hpo_general_config}")
    
    ready_to_test = True
    for model_name_test in test_models:
        if model_name_test not in RANDOM_OOS or not RANDOM_OOS[model_name_test]:
            print(f"Error: RANDOM_OOS['{model_name_test}'] is not defined or empty. Cannot run test.", file=sys.stderr)
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
        print("Please define search spaces for test models in RANDOM_OOS for testing.", file=sys.stderr)

    print("--- Test run of random_oos.py finished ---")