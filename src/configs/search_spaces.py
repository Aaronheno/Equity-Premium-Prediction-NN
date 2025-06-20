"""
Hyperparameter Search Space Configurations for Neural Network Optimization

This module defines search spaces for Grid Search, Random Search, and Bayesian
Optimization across all neural network architectures. Optimized for parallel
hyperparameter optimization with thread-safe parameter generation.

Threading Status: THREAD_SAFE
Hardware Requirements: CPU_ONLY (configuration), scales with HPO method
Performance Notes:
    - Parameter generation is lightweight and thread-safe
    - Search spaces designed for parallel trial evaluation
    - Grid spaces optimized to prevent combinatorial explosion
    - Bayesian spaces support 100+ parallel trials

Parallelization Features:
    - All distribution objects are thread-safe
    - Parameter sampling supports concurrent access
    - Grid parameter combinations can be evaluated in parallel
    - Bayesian search spaces support high-throughput trial generation

HPO Method Compatibility:
    - Grid Search: Explicit parameter lists, limited combinations
    - Random Search: Distribution-based sampling, unlimited trials
    - Bayesian Search: Continuous/categorical distributions, guided sampling

Threading Notes:
    - Parameter generation: Thread-safe across all methods
    - Memory usage: Minimal, scales with search space size
    - Concurrent access: Safe for multiple HPO processes
"""

# src/configs/search_spaces.py
# Master dictionaries for each search method

import sys 
import numpy as np
import torch.optim
# Ensure this path is correct if your project structure is different
# or if you run scripts from a specific directory.
from src.utils.distributions import CategoricalDistribution, FloatDistribution, IntDistribution

print("--- Executing src.configs.search_spaces.py ---", file=sys.stderr)

try:
    # --- Base Hyperparameter Definitions ---

    # For RANDOM and BAYESIAN searches (using distribution objects)
    _BASE_DISTRIBUTIONS = {
        "optimizer_choice": CategoricalDistribution(["Adam", "RMSprop", "SGD"]),
        "lr": FloatDistribution(1e-5, 1e-2, log=True),
        "weight_decay": FloatDistribution(1e-7, 1e-2, log=True), # L2 regularization
        "l1_lambda": FloatDistribution(1e-7, 1e-2, log=True),      # L1 regularization
        "dropout": FloatDistribution(0.0, 0.6, step=0.05), 
        "batch_size": CategoricalDistribution([64, 128, 256, 512, 1024]),
        # n_feature and n_output are set dynamically in experiment scripts.
        # activation functions will be defined per model type if they differ.
    }
    print("--- _BASE_DISTRIBUTIONS defined for Random/Bayesian Search ---", file=sys.stderr)

    # For GRID search (parameters must be lists of explicit values)
    _BASE_GRID_PARAMS = {
        "optimizer": [torch.optim.Adam, torch.optim.SGD],
        "lr": [1e-4, 5e-4, 1e-3, 2e-3], 
        "optimizer__weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
        "l1_lambda": [0.0, 1e-5, 1e-4, 1e-3],    
        "module__dropout": [0.0, 0.1, 0.2, 0.3, 0.4],
        "batch_size": [128, 256, 512], 
    }
    print("--- _BASE_GRID_PARAMS defined for Grid Search ---", file=sys.stderr)

    # Initialize dictionaries for each search method (In-Sample)
    BAYES = {}
    GRID = {}
    RANDOM = {}
    print("--- In-Sample BAYES, GRID, RANDOM dictionaries initialized ---", file=sys.stderr)

    # --- Helper function for creating Optuna HPO config functions (for BAYES spaces) ---
    def _create_optuna_hpo_config_fn(model_name_suffix, hidden_layer_configs, is_dnn_model=False, base_params_dist=_BASE_DISTRIBUTIONS):
        """
        Creates a configuration function for an Optuna trial.
        Args:
            model_name_suffix (str): Unique suffix for this model (e.g., "Net1", "DNN1").
            hidden_layer_configs (dict): Keys are 'n_hiddenX', values are IntDistribution.
            is_dnn_model (bool): True if it's a DNN model (uses 'activation_fn'), False for Net (uses 'activation_hidden').
            base_params_dist (dict): Base distribution parameters.
        """
        def hpo_config_fn(trial, n_features): # n_features is passed by the objective function
            params = {}
            # Model architecture parameters (hidden layers)
            for layer_name, dist_obj in hidden_layer_configs.items():
                # Ensure unique name for Optuna's trial.suggest_* methods
                param_suggest_name = f"module__{layer_name}_{model_name_suffix}"
                params[f"module__{layer_name}"] = dist_obj.sample(trial=trial, name=param_suggest_name)
            
            # Activation function
            if is_dnn_model:
                params["module__activation_fn"] = trial.suggest_categorical(
                    f"module__activation_fn_{model_name_suffix}", ["relu"]
                )
            else: # For Net1-Net5
                params["module__activation_hidden"] = trial.suggest_categorical(
                    f"module__activation_hidden_{model_name_suffix}", ["relu"]
                )
            # Note: ReLU is used as the only activation function for hidden layers based on
            # sensitivity testing that showed it consistently outperformed other activations.
            # The output layer uses a linear activation (default in the model implementation)
            # which is appropriate for this regression task.
            
            # Training/Skorch parameters from base distributions
            params["optimizer"] = base_params_dist["optimizer_choice"].sample(trial=trial, name=f"optimizer_{model_name_suffix}")
            params["lr"] = base_params_dist["lr"].sample(trial=trial, name=f"lr_{model_name_suffix}")
            params["optimizer__weight_decay"] = base_params_dist["weight_decay"].sample(trial=trial, name=f"weight_decay_{model_name_suffix}")
            params["l1_lambda"] = base_params_dist["l1_lambda"].sample(trial=trial, name=f"l1_lambda_{model_name_suffix}")
            params["module__dropout"] = base_params_dist["dropout"].sample(trial=trial, name=f"module__dropout_{model_name_suffix}") # Skorch needs module__
            params["batch_size"] = base_params_dist["batch_size"].sample(trial=trial, name=f"batch_size_{model_name_suffix}")
            return params
        return hpo_config_fn

    # --- IN-SAMPLE SEARCH SPACES ---

    # --- BAYES (In-Sample) ---
    # Defines hpo_config_fn for each model, used by training_optuna.py
    BAYES["Net1"] = {"hpo_config_fn": _create_optuna_hpo_config_fn("Net1", {"n_hidden1": IntDistribution(16, 256)})}
    BAYES["Net2"] = {"hpo_config_fn": _create_optuna_hpo_config_fn("Net2", {"n_hidden1": IntDistribution(16, 192), "n_hidden2": IntDistribution(8, 128)})}
    BAYES["Net3"] = {"hpo_config_fn": _create_optuna_hpo_config_fn("Net3", {"n_hidden1": IntDistribution(16, 128), "n_hidden2": IntDistribution(8, 96), "n_hidden3": IntDistribution(4, 64)})}
    BAYES["Net4"] = {"hpo_config_fn": _create_optuna_hpo_config_fn("Net4", {"n_hidden1": IntDistribution(32, 192), "n_hidden2": IntDistribution(16, 128), "n_hidden3": IntDistribution(8, 96), "n_hidden4": IntDistribution(4, 64)})}
    BAYES["Net5"] = {"hpo_config_fn": _create_optuna_hpo_config_fn("Net5", {"n_hidden1": IntDistribution(32, 256), "n_hidden2": IntDistribution(16, 192), "n_hidden3": IntDistribution(8, 128), "n_hidden4": IntDistribution(8, 96), "n_hidden5": IntDistribution(4, 64)})}
    
    BAYES["DNet1"] = {"hpo_config_fn": _create_optuna_hpo_config_fn("DNet1", {"n_hidden1": IntDistribution(64, 384), "n_hidden2": IntDistribution(32, 256), "n_hidden3": IntDistribution(16, 192), "n_hidden4": IntDistribution(16, 128)}, is_dnn_model=True)}
    # DNet2 requires 5 hidden layers - added n_hidden5
    BAYES["DNet2"] = {"hpo_config_fn": _create_optuna_hpo_config_fn("DNet2", {"n_hidden1": IntDistribution(64, 512), "n_hidden2": IntDistribution(32, 384), "n_hidden3": IntDistribution(16, 256), "n_hidden4": IntDistribution(16, 192), "n_hidden5": IntDistribution(8, 128)}, is_dnn_model=True)}
    BAYES["DNet3"] = {"hpo_config_fn": _create_optuna_hpo_config_fn("DNet3", {"n_hidden1": IntDistribution(64, 512), "n_hidden2": IntDistribution(32, 384), "n_hidden3": IntDistribution(16, 256), "n_hidden4": IntDistribution(16, 192), "n_hidden5": IntDistribution(8, 128)}, is_dnn_model=True)}
    print("--- BAYES (In-Sample) search spaces defined ---", file=sys.stderr)

    # --- RANDOM (In-Sample) ---
    # Defines parameter distributions directly.
    RANDOM["Net1"] = {**_BASE_DISTRIBUTIONS, "module__n_hidden1": IntDistribution(16, 256), "module__activation_hidden": CategoricalDistribution(['relu'])}
    RANDOM["Net2"] = {**_BASE_DISTRIBUTIONS, "module__n_hidden1": IntDistribution(16, 192), "module__n_hidden2": IntDistribution(8, 128), "module__activation_hidden": CategoricalDistribution(['relu'])}
    RANDOM["Net3"] = {**_BASE_DISTRIBUTIONS, "module__n_hidden1": IntDistribution(16, 128), "module__n_hidden2": IntDistribution(8, 96), "module__n_hidden3": IntDistribution(4, 64), "module__activation_hidden": CategoricalDistribution(['relu'])}
    RANDOM["Net4"] = {**_BASE_DISTRIBUTIONS, "module__n_hidden1": IntDistribution(32, 192), "module__n_hidden2": IntDistribution(16, 128), "module__n_hidden3": IntDistribution(8, 96), "module__n_hidden4": IntDistribution(4, 64), "module__activation_hidden": CategoricalDistribution(['relu'])}
    RANDOM["Net5"] = {**_BASE_DISTRIBUTIONS, "module__n_hidden1": IntDistribution(32, 256), "module__n_hidden2": IntDistribution(16, 192), "module__n_hidden3": IntDistribution(8, 128), "module__n_hidden4": IntDistribution(8, 96), "module__n_hidden5": IntDistribution(4, 64), "module__activation_hidden": CategoricalDistribution(['relu'])}

    RANDOM["DNet1"] = {**_BASE_DISTRIBUTIONS, "module__n_hidden1": IntDistribution(64, 384), "module__n_hidden2": IntDistribution(32, 256), "module__n_hidden3": IntDistribution(16, 192), "module__n_hidden4": IntDistribution(16, 128), "module__activation_fn": CategoricalDistribution(['relu'])}
    # DNet2 requires 5 hidden layers - added n_hidden5
    RANDOM["DNet2"] = {**_BASE_DISTRIBUTIONS, "module__n_hidden1": IntDistribution(64, 512), "module__n_hidden2": IntDistribution(32, 384), "module__n_hidden3": IntDistribution(16, 256), "module__n_hidden4": IntDistribution(16, 192), "module__n_hidden5": IntDistribution(8, 128), "module__activation_fn": CategoricalDistribution(['relu'])}
    # DNet3 now has 5 hidden layers (removed n_hidden6)
    RANDOM["DNet3"] = {**_BASE_DISTRIBUTIONS, "module__n_hidden1": IntDistribution(64, 512), "module__n_hidden2": IntDistribution(32, 384), "module__n_hidden3": IntDistribution(16, 256), "module__n_hidden4": IntDistribution(16, 192), "module__n_hidden5": IntDistribution(8, 128), "module__activation_fn": CategoricalDistribution(['relu'])}
    print("--- RANDOM (In-Sample) search spaces defined ---", file=sys.stderr)

    # --- GRID (In-Sample) ---
    # Defines lists of explicit values for each parameter.
    GRID["Net1"] = {**_BASE_GRID_PARAMS, "module__n_hidden1": [32, 64, 128, 256], "module__activation_hidden": ["relu"]}
    GRID["Net2"] = {**_BASE_GRID_PARAMS, "module__n_hidden1": [64, 128, 192], "module__n_hidden2": [32, 64, 96], "module__activation_hidden": ["relu"]}
    GRID["Net3"] = {**_BASE_GRID_PARAMS, "module__n_hidden1": [64, 128], "module__n_hidden2": [32, 64], "module__n_hidden3": [16, 32, 48], "module__activation_hidden": ["relu"]}
    GRID["Net4"] = {**_BASE_GRID_PARAMS, "module__n_hidden1": [64, 128], "module__n_hidden2": [48, 96], "module__n_hidden3": [32, 64], "module__n_hidden4": [16, 32], "module__activation_hidden": ["relu"]}
    GRID["Net5"] = {**_BASE_GRID_PARAMS, "module__n_hidden1": [96, 128, 192], "module__n_hidden2": [64, 96, 128], "module__n_hidden3": [48, 64, 96], "module__n_hidden4": [32, 48, 64], "module__n_hidden5": [16, 24, 32], "module__activation_hidden": ["relu"]}
    
    GRID["DNet1"] = {**_BASE_GRID_PARAMS, "module__n_hidden1": [128, 256], "module__n_hidden2": [64, 128], "module__n_hidden3": [32, 64], "module__n_hidden4": [16, 32], "module__activation_fn": ["relu"]}
    # DNet2 requires 5 hidden layers - added n_hidden5
    GRID["DNet2"] = {**_BASE_GRID_PARAMS, "module__n_hidden1": [128, 256, 384], "module__n_hidden2": [96, 192, 256], "module__n_hidden3": [64, 128, 192], "module__n_hidden4": [32, 64, 96], "module__n_hidden5": [16, 32, 48], "module__activation_fn": ["relu"]}
    GRID["DNet3"] = {**_BASE_GRID_PARAMS, "module__n_hidden1": [192, 256, 384], "module__n_hidden2": [128, 192, 256], "module__n_hidden3": [96, 128, 192], "module__n_hidden4": [64, 96, 128], "module__n_hidden5": [32, 48, 64], "module__activation_fn": ["relu"]}
    print("--- GRID (In-Sample) search spaces defined ---", file=sys.stderr)

    # --- OUT-OF-SAMPLE (OOS) Search Spaces ---
    # These are used by the *_oos.py experiment scripts.
    # For OOS, especially with annual HPO, grids should be much smaller.
    # BAYES_OOS and RANDOM_OOS can often reuse the in-sample distribution definitions.
    print("\n--- Defining OOS Search Spaces ---", file=sys.stderr)
    
    BAYES_OOS = {}
    
    # Ensure all 8 models are properly configured for OOS Bayesian optimization
    # First, initialize from the in-sample configurations if available
    if 'BAYES' in locals() and isinstance(BAYES, dict):
        for model_name_key in ["Net1", "Net2", "Net3", "Net4", "Net5", "DNet1", "DNet2", "DNet3"]:
            if model_name_key in BAYES and "hpo_config_fn" in BAYES[model_name_key]:
                BAYES_OOS[model_name_key] = {"hpo_config_fn": BAYES[model_name_key]["hpo_config_fn"]}
            else:
                print(f"Warning: Config for {model_name_key} not found or 'hpo_config_fn' missing in BAYES. Will create a default one.", file=sys.stderr)
        print("--- BAYES_OOS initialized from in-sample BAYES structure ---", file=sys.stderr)
    else:
        print("Warning: In-sample 'BAYES' search space not found. Creating default configurations.", file=sys.stderr)
    
    # Add any missing model configurations to ensure all 8 models are available
    all_models = ["Net1", "Net2", "Net3", "Net4", "Net5", "DNet1", "DNet2", "DNet3"]
    
    # Add default configurations for any missing models
    for model_name in all_models:
        if model_name not in BAYES_OOS or "hpo_config_fn" not in BAYES_OOS[model_name]:
            # Create default configurations based on model type
            if model_name == "Net1":
                BAYES_OOS[model_name] = {"hpo_config_fn": _create_optuna_hpo_config_fn(model_name, {"n_hidden1": IntDistribution(16, 256)})}
            elif model_name == "Net2":
                BAYES_OOS[model_name] = {"hpo_config_fn": _create_optuna_hpo_config_fn(model_name, {"n_hidden1": IntDistribution(16, 192), "n_hidden2": IntDistribution(8, 128)})}
            elif model_name == "Net3":
                BAYES_OOS[model_name] = {"hpo_config_fn": _create_optuna_hpo_config_fn(model_name, {"n_hidden1": IntDistribution(16, 128), "n_hidden2": IntDistribution(8, 96), "n_hidden3": IntDistribution(4, 64)})}
            elif model_name == "Net4":
                BAYES_OOS[model_name] = {"hpo_config_fn": _create_optuna_hpo_config_fn(model_name, {"n_hidden1": IntDistribution(32, 192), "n_hidden2": IntDistribution(16, 128), "n_hidden3": IntDistribution(8, 96), "n_hidden4": IntDistribution(4, 64)})}
            elif model_name == "Net5":
                BAYES_OOS[model_name] = {"hpo_config_fn": _create_optuna_hpo_config_fn(model_name, {"n_hidden1": IntDistribution(32, 256), "n_hidden2": IntDistribution(16, 192), "n_hidden3": IntDistribution(8, 128), "n_hidden4": IntDistribution(8, 96), "n_hidden5": IntDistribution(4, 64)})}
            elif model_name == "DNet1":
                BAYES_OOS[model_name] = {"hpo_config_fn": _create_optuna_hpo_config_fn(model_name, {"n_hidden1": IntDistribution(64, 384), "n_hidden2": IntDistribution(32, 256), "n_hidden3": IntDistribution(16, 192), "n_hidden4": IntDistribution(16, 128)}, is_dnn_model=True)}
            elif model_name == "DNet2":
                BAYES_OOS[model_name] = {"hpo_config_fn": _create_optuna_hpo_config_fn(model_name, {"n_hidden1": IntDistribution(64, 384), "n_hidden2": IntDistribution(48, 256), "n_hidden3": IntDistribution(32, 192), "n_hidden4": IntDistribution(24, 128), "n_hidden5": IntDistribution(12, 64)}, is_dnn_model=True)}
            elif model_name == "DNet3":
                BAYES_OOS[model_name] = {"hpo_config_fn": _create_optuna_hpo_config_fn(model_name, {"n_hidden1": IntDistribution(128, 512), "n_hidden2": IntDistribution(64, 384), "n_hidden3": IntDistribution(48, 256), "n_hidden4": IntDistribution(32, 192), "n_hidden5": IntDistribution(16, 128)}, is_dnn_model=True)}

            print(f"Created default configuration for {model_name} in BAYES_OOS", file=sys.stderr)
            
    print(f"--- BAYES_OOS now contains all 8 models: {sorted(list(BAYES_OOS.keys()))} ---", file=sys.stderr)

    RANDOM_OOS = {}
    if 'RANDOM' in locals() and isinstance(RANDOM, dict):
        RANDOM_OOS = {k: v.copy() for k, v in RANDOM.items()} 
        print("--- RANDOM_OOS initialized (copied from in-sample RANDOM) ---", file=sys.stderr)
    else:
        print("Error: In-sample 'RANDOM' search space not found. RANDOM_OOS cannot be derived.", file=sys.stderr)

    # --- ADD/OVERRIDE DNet configurations for RANDOM_OOS to be "less deep" ---
    # These definitions will override any DNetX copied from RANDOM for RANDOM_OOS
    RANDOM_OOS["DNet1"] = {**_BASE_DISTRIBUTIONS,
                          "module__n_hidden1": IntDistribution(64, 96, step=32),
                          "module__n_hidden2": IntDistribution(32, 64, step=16),
                          "module__n_hidden3": IntDistribution(16, 32, step=16),
                          "module__n_hidden4": CategoricalDistribution([16]),
                          "module__activation_fn": CategoricalDistribution(["relu"])}

    RANDOM_OOS["DNet2"] = {**_BASE_DISTRIBUTIONS,
                          "module__n_hidden1": IntDistribution(96, 128, step=32),
                          "module__n_hidden2": IntDistribution(48, 96, step=16),
                          "module__n_hidden3": IntDistribution(32, 48, step=16),
                          "module__n_hidden4": CategoricalDistribution([24]),
                          "module__n_hidden5": CategoricalDistribution([12]),
                          "module__activation_fn": CategoricalDistribution(["relu"])}

    RANDOM_OOS["DNet3"] = {**_BASE_DISTRIBUTIONS,
                          "module__n_hidden1": IntDistribution(128, 192, step=32),
                          "module__n_hidden2": IntDistribution(64, 128, step=32),
                          "module__n_hidden3": IntDistribution(32, 64, step=16),
                          "module__n_hidden4": CategoricalDistribution([32]),
                          "module__n_hidden5": CategoricalDistribution([16]),
                          "module__n_hidden6": CategoricalDistribution([8]),
                          "module__activation_fn": CategoricalDistribution(["relu"])}
    print("--- DNet configurations for RANDOM_OOS have been overridden to be simpler. ---", file=sys.stderr)

    # --- OOS Grid Search Spaces (Significantly Reduced for Feasibility) ---
    # WARNING: Annual HPO with Grid Search is VERY computationally expensive.
    # These grids are intentionally kept extremely small (1-2 options per param).
    # Expand these grids very cautiously based on available computational resources.
    GRID_OOS = {}
    _BASE_GRID_PARAMS_OOS_REDUCED = { 
        "optimizer": [torch.optim.Adam, torch.optim.RMSprop],  # Added RMSprop
        "lr": [5e-4],  # Single value for efficiency
        "optimizer__weight_decay": [1e-5, 1e-4],  # Two L2 regularization options
        "l1_lambda": [0, 1e-4],  # Both no L1 and moderate L1 sparsity
        "module__dropout": [0.1, 0.3],  # Low and moderate dropout
        "batch_size": [256],  # Standard batch size   
    }

    GRID_OOS["Net1"] = {**_BASE_GRID_PARAMS_OOS_REDUCED, "module__n_hidden1": [64],  "module__activation_hidden": ["relu"]} # Single neuron count value "module__activation_hidden": ["relu"]
    GRID_OOS["Net2"] = {**_BASE_GRID_PARAMS_OOS_REDUCED, "module__n_hidden1": [64], "module__n_hidden2": [32], "module__activation_hidden": ["relu"]}
    GRID_OOS["Net3"] = {**_BASE_GRID_PARAMS_OOS_REDUCED, "module__n_hidden1": [96], "module__n_hidden2": [48], "module__n_hidden3": [24], "module__activation_hidden": ["relu"]}
    GRID_OOS["Net4"] = {**_BASE_GRID_PARAMS_OOS_REDUCED, "module__n_hidden1": [96], "module__n_hidden2": [64], "module__n_hidden3": [32], "module__n_hidden4": [16], "module__activation_hidden": ["relu"]}
    GRID_OOS["Net5"] = {**_BASE_GRID_PARAMS_OOS_REDUCED, "module__n_hidden1": [96], "module__n_hidden2": [64], "module__n_hidden3": [48], "module__n_hidden4": [32], "module__n_hidden5": [16], "module__activation_hidden": ["relu"]}
    
    GRID_OOS["DNet1"] = {**_BASE_GRID_PARAMS_OOS_REDUCED, "module__n_hidden1": [256], "module__n_hidden2": [128], "module__n_hidden3": [64], "module__n_hidden4": [32], "module__activation_fn": ["relu"]}
    # DNet2 requires 5 hidden layers - added n_hidden5
    GRID_OOS["DNet2"] = {**_BASE_GRID_PARAMS_OOS_REDUCED, "module__n_hidden1": [256], "module__n_hidden2": [192], "module__n_hidden3": [128], "module__n_hidden4": [64], "module__n_hidden5": [32], "module__activation_fn": ["relu"]}
    # DNet3 now has 5 hidden layers - higher neuron counts than DNet2
    GRID_OOS["DNet3"] = {**_BASE_GRID_PARAMS_OOS_REDUCED, 
                         "module__n_hidden1": [384],  # Higher than DNet2
                         "module__n_hidden2": [256],  # Higher than DNet2
                         "module__n_hidden3": [192],  # Higher than DNet2
                         "module__n_hidden4": [128],  # Higher than DNet2
                         "module__n_hidden5": [64],   # Higher than DNet2
                         "module__activation_fn": ["relu"]}
    
    print("--- GRID_OOS initialized with REDUCED parameter sets for OOS. Expand cautiously. ---", file=sys.stderr)
    for model_key in ["Net1", "Net2", "Net3", "Net4", "Net5", "DNet1", "DNet2", "DNet3"]:
        if model_key not in GRID_OOS: # Check if the key was actually added (e.g. if GRID was empty)
             print(f"Warning: GRID_OOS for {model_key} was not defined. This might happen if the base GRID was empty or model not included.", file=sys.stderr)
        else:
            print(f"GRID_OOS for {model_key}: {GRID_OOS.get(model_key)}", file=sys.stderr)

except Exception as e:
    print(f"--- ERROR during execution of src.configs.search_spaces.py: {e} ---", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    raise

print("--- Finished executing src.configs.search_spaces.py ---", file=sys.stderr)
