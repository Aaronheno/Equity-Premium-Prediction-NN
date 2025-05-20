# src/experiments/grid_oos.py
import sys
from pathlib import Path
import torch
# import argparse # Reverted

# --- Add project root to sys.path ---
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
# --- End sys.path modification ---

from src.models import nns
from src.configs.search_spaces import GRID_OOS # Expecting GRID_OOS in search_spaces.py
from src.utils.training_grid import train_grid as grid_hpo_function, GridNet as GridSearchSkorchNet
from src.utils.oos_common import run_oos_experiment, OOS_DEFAULT_START_YEAR_MONTH
from src.utils.grid_helpers import grid_hpo_runner_function
from src.utils.load_models import get_model_class_from_name # Helper
# from skorch import NeuralNetRegressor # Reverted

# Define a default path for saving runs, can be overridden by CLI
# DEFAULT_BASE_RUN_FOLDER = "grid_search_oos_runs" # This was fine, but let's ensure consistency with original structure if it was different

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
    save_annual_models=False
):
    """
    Runs the Out-of-Sample (OOS) experiment using Grid Search for HPO.
    """
    experiment_name_suffix = "grid_search_oos"
    base_run_folder_name = "grid_oos"

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
        save_annual_models=save_annual_models
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