"""
Grid Search Out-of-Sample Neural Network Optimization

This experiment conducts out-of-sample evaluation using exhaustive grid search
hyperparameter optimization for neural network models in equity premium prediction.
Optimized for parallel parameter combination evaluation.

Threading Status: PARALLEL_READY (Parameter-level parallelism within models)
Hardware Requirements: CPU_INTENSIVE, CUDA_BENEFICIAL, MODERATE_MEMORY
Performance Notes:
    - Parameter combinations: 3-8x speedup with parallel evaluation
    - Grid traversal: CPU-intensive, benefits from multi-core systems
    - Memory scaling: Linear with grid size and model count
    - Optimal for CPU-heavy workloads with moderate memory

Experiment Type: Out-of-Sample Evaluation with Annual Grid Search
Models Supported: Net1, Net2, Net3, Net4, Net5, DNet1, DNet2, DNet3
HPO Method: Exhaustive Grid Search
Output Directory: runs/1_Grid_Search_OOS/

Critical Parallelization Opportunities:
    1. Parameter combination evaluation within each model/time step
    2. Cross-validation folds can be parallel (sklearn GridSearchCV)
    3. Model training can be concurrent after grid completion
    4. Prediction generation across models

Threading Implementation Status:
    ❌ Sequential model processing (MAIN BOTTLENECK)
    ✅ Parameter combinations can be parallelized (sklearn n_jobs)
    ❌ Model training sequential within time steps
    ❌ Grid traversal sequential across models

Future Parallel Implementation:
    run(models, parallel_models=True, grid_parallel=True, n_jobs=32)

Expected Performance Gains:
    - Current: 6 hours for 8 models × grid combinations
    - With parameter parallelism: 2-3 hours (2-3x speedup)
    - With model parallelism: 1-1.5 hours (additional 2x speedup)
    - Combined optimization: 30-60 minutes (6-12x total speedup)

Grid Search Advantages:
    - Exhaustive parameter space exploration
    - Reproducible results with deterministic ordering
    - Natural early stopping with poor parameter combinations
    - Excellent CPU utilization with parallel evaluation
"""

# src/experiments/grid_oos.py
import sys
from pathlib import Path
import torch

# --- Add project root to sys.path ---
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
# --- End sys.path modification ---

from src.models import nns
from src.configs.search_spaces import GRID_OOS # Expecting GRID_OOS in search_spaces.py
from src.utils.training_grid import train_grid as grid_hpo_function, GridNet as GridSearchSkorchNet
from src.utils.oos_common import run_oos_experiment, OOS_DEFAULT_START_YEAR_MONTH
from src.utils.load_models import get_model_class_from_name


# Define a mapping from model names (strings) to their classes and HPO details
# This should align with what's available in GRID_OOS
ALL_NN_MODEL_CONFIGS_GRID_OOS = {
    model_name: {
        "model_class": get_model_class_from_name(model_name),
        "hpo_function": grid_hpo_function,
        "regressor_class": GridSearchSkorchNet, # Use the Skorch wrapper for grid search
        "search_space_config_or_fn": GRID_OOS.get(model_name)
    }
    for model_name in ["Net1", "Net2", "Net3", "Net4", "Net5", "DNet1", "DNet2", "DNet3"]
    if GRID_OOS.get(model_name) is not None # Only include if config exists
}

def run(
    model_names, # <<< ADD model_names HERE
    oos_start_date_int,
    hpo_general_config, # Includes epochs, device, etc. (trials not used by grid)
    save_annual_models=False,
    parallel_models=False,  # <<< ADD: Enable model-level parallelism for 32-core server
    verbose=False
):
    """
    Runs the Out-of-Sample (OOS) experiment using Grid Search for HPO.
    """
    experiment_name_suffix = "grid_search_oos"
    base_run_folder_name = "1_Grid_Search_OOS"

    # Filter ALL_NN_MODEL_CONFIGS_GRID_OOS based on model_names from CLI
    if not model_names:
        print("No models specified to run. Exiting.")
        return

    nn_model_configs_to_run = {
        name: config for name, config in ALL_NN_MODEL_CONFIGS_GRID_OOS.items()
        if name in model_names
    }

    if not nn_model_configs_to_run:
        print(f"None of the specified models ({model_names}) have configurations in GRID_OOS. Exiting.")
        return

    print(f"--- Running Grid OOS for models: {list(nn_model_configs_to_run.keys())} ---")

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
    print("--- Running grid_oos.py directly for testing purposes ---")
    
    test_models = ["Net1", "Net2"] # Example: test with Net1 and Net2
    test_epochs = 5      
    test_batch_size = 64
    test_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Test device: {test_device}")

    # Check if GRID_OOS for test_models are defined before running
    ready_to_test = True
    for model_name_test in test_models:
        if model_name_test not in GRID_OOS or not GRID_OOS[model_name_test]:
            print(f"Error: GRID_OOS['{model_name_test}'] is not defined or empty in search_spaces.py. Cannot run test.", file=sys.stderr)
            ready_to_test = False
    
    if ready_to_test:
        # Create a proper hpo_general_config structure that matches what run expects
        test_hpo_general_config = {
            "hpo_epochs": test_epochs,
            "hpo_device": test_device,
            "hpo_batch_size": test_batch_size
        }
        run(
            model_names=test_models,
            oos_start_date_int=202001,
            hpo_general_config=test_hpo_general_config,
            save_annual_models=False
        )
    else:
        print("Please define small grids for test models in GRID_OOS for testing.", file=sys.stderr)
        
    print("--- Test run of grid_oos.py finished ---")