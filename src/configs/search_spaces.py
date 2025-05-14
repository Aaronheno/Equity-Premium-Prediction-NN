# master dictionaries for each search method
import sys # Add sys import for stderr
import numpy as np
from src.utils.distributions import CategoricalDistribution, FloatDistribution, IntDistribution

print("--- Executing src.configs.search_spaces.py ---", file=sys.stderr)

try:
    # Base parameters common to most searches
    _BASE = {
        "optimizer": ["Adam", "RMSprop"], # For GRID
        "optimizer_choice": CategoricalDistribution(["Adam", "RMSprop"]), # For RANDOM/BAYES
        "lr": FloatDistribution(1e-5, 1e-2, log=True),
        "lr_grid": [1e-4, 5e-4, 1e-3, 5e-3],
        "weight_decay": FloatDistribution(1e-6, 1e-3, log=True), # For L2
        "weight_decay_grid": [0.0, 1e-5, 1e-4],
        "l1_lambda": FloatDistribution(1e-6, 1e-2, log=True), # For L1
        "l1_lambda_grid": [0.0, 1e-5, 1e-4, 1e-3],
        "dropout": FloatDistribution(0.0, 0.5),
        "dropout_grid": [0.0, 0.1, 0.25, 0.4],
        "batch_size": CategoricalDistribution([64, 128, 256]),
        "batch_size_grid": [64, 128, 256],
        # n_feature and n_output are usually set dynamically in the experiment script
    }
    print("--- _BASE defined ---", file=sys.stderr)

    # --- Create a base dictionary specifically for GRID parameters ---
    _BASE_GRID_PARAMS = {
        "optimizer": _BASE["optimizer"], # Already a list: ["Adam", "RMSprop"]
        "lr": _BASE["lr_grid"],
        "weight_decay": _BASE["weight_decay_grid"],
        "l1_lambda": _BASE["l1_lambda_grid"],
        "dropout": _BASE["dropout_grid"],
        "batch_size": _BASE["batch_size_grid"],
    }
    print("--- _BASE_GRID_PARAMS defined ---", file=sys.stderr)

    # --- Break the chained assignment into separate lines ---
    BAYES = {}
    GRID = {}
    RANDOM = {}
    # --- End modification ---
    print("--- BAYES, GRID, RANDOM initialized ---", file=sys.stderr)

    # This loop seems redundant as H isn't used inside, but shouldn't cause an error
    for H in [16,32,64]:
        pass
    print("--- Loop 'for H...' completed ---", file=sys.stderr)

    # --- Bayesian Optimization Search Spaces ---
    # Net1: 1 Hidden Layer
    BAYES["Net1"] = _BASE.copy()
    BAYES["Net1"].update({
        "n_hidden1": IntDistribution(16, 256), # Define directly for Net1
        # "dropout" is already inherited from _BASE, no need to update unless overriding
    })

    # Net2: 2 Hidden Layers
    BAYES["Net2"] = _BASE.copy()
    BAYES["Net2"].update({
        "n_hidden1": IntDistribution(16, 128), # Define directly for Net2
        "n_hidden2": IntDistribution(8, 64),   # Define directly for Net2
    })

    # Net3: 3 Hidden Layers
    BAYES["Net3"] = _BASE.copy()
    BAYES["Net3"].update({
        "n_hidden1": IntDistribution(16, 128), # Define directly for Net3
        "n_hidden2": IntDistribution(8, 64),   # Define directly for Net3
        "n_hidden3": IntDistribution(4, 32),   # Define directly for Net3
    })

    # Net4: 4 Hidden Layers
    BAYES["Net4"] = _BASE.copy()
    BAYES["Net4"].update({
        "n_hidden1": IntDistribution(32, 192),
        "n_hidden2": IntDistribution(16, 128),
        "n_hidden3": IntDistribution(8, 96),
        "n_hidden4": IntDistribution(4, 64),
    })

    # Net5: 5 Hidden Layers
    BAYES["Net5"] = _BASE.copy()
    BAYES["Net5"].update({
        "n_hidden1": IntDistribution(32, 256),
        "n_hidden2": IntDistribution(16, 192),
        "n_hidden3": IntDistribution(8, 128),
        "n_hidden4": IntDistribution(8, 96),
        "n_hidden5": IntDistribution(4, 64),
    })

    # DNN1: Example 3 Hidden Layers (Deep Network)
    BAYES["DNet1"] = _BASE.copy()
    BAYES["DNet1"].update({
        "n_hidden1": IntDistribution(64, 256), # Wider/deeper ranges for DNet
        "n_hidden2": IntDistribution(32, 192),
        "n_hidden3": IntDistribution(16, 128),
        "n_hidden4": IntDistribution(16, 64),
    })

    # DNN2: Example 4 Hidden Layers (Deep Network)
    BAYES["DNet2"] = _BASE.copy()
    BAYES["DNet2"].update({
        "n_hidden1": IntDistribution(64, 300),
        "n_hidden2": IntDistribution(32, 256),
        "n_hidden3": IntDistribution(16, 192),
        "n_hidden4": IntDistribution(16, 128),
    })

    # DNN3: Example 5 Hidden Layers (Deep Network)
    BAYES["DNet3"] = _BASE.copy()
    BAYES["DNet3"].update({
        "n_hidden1": IntDistribution(64, 384),
        "n_hidden2": IntDistribution(32, 300),
        "n_hidden3": IntDistribution(16, 256),
        "n_hidden4": IntDistribution(16, 192),
        "n_hidden5": IntDistribution(8, 128),
    })
    print("--- BAYES search spaces defined ---", file=sys.stderr)

    # --- RANDOM SEARCH SPACES ---
    # Typically similar to BAYES, using the same Distribution objects
    RANDOM = {}
    for model_name, params in BAYES.items(): # Leverage BAYES definitions for RANDOM
        RANDOM[model_name] = params.copy() # Creates a shallow copy
    print("--- RANDOM search spaces defined (copied from BAYES) ---", file=sys.stderr)


    # --- GRID SEARCH SPACES ---
    # For GRID, all parameter values must be lists.
    GRID = {}
    # Net1: 1 Hidden Layer
    GRID["Net1"] = _BASE_GRID_PARAMS.copy() # Use the grid-specific base
    GRID["Net1"]["optimizer"] = ["Adam"] # Example: fix optimizer for this model
    GRID["Net1"]["lr"] = _BASE_GRID_PARAMS["lr"][:2] # Example: use a subset of learning rates
    GRID["Net1"]["n_hidden1"] = [32, 64, 128]

    # Net2: 2 Hidden Layers
    GRID["Net2"] = _BASE_GRID_PARAMS.copy()
    GRID["Net2"]["optimizer"] = ["Adam"]
    GRID["Net2"]["lr"] = _BASE_GRID_PARAMS["lr"][:2]
    GRID["Net2"]["n_hidden1"] = [64, 128]
    GRID["Net2"]["n_hidden2"] = [32, 64]

    # Net3: 3 Hidden Layers
    GRID["Net3"] = _BASE_GRID_PARAMS.copy()
    GRID["Net3"]["optimizer"] = ["Adam"]
    GRID["Net3"]["lr"] = _BASE_GRID_PARAMS["lr"][:2]
    GRID["Net3"]["n_hidden1"] = [64, 128, 192]
    GRID["Net3"]["n_hidden2"] = [32, 64, 96]
    GRID["Net3"]["n_hidden3"] = [16, 32, 48]

    # Net4: 4 Hidden Layers
    GRID["Net4"] = _BASE_GRID_PARAMS.copy()
    GRID["Net4"]["optimizer"] = ["Adam"]
    GRID["Net4"]["lr"] = _BASE_GRID_PARAMS["lr"][:2]
    GRID["Net4"]["n_hidden1"] = [64, 128]
    GRID["Net4"]["n_hidden2"] = [48, 96]
    GRID["Net4"]["n_hidden3"] = [32, 64]
    GRID["Net4"]["n_hidden4"] = [16, 32]

    # Net5: 5 Hidden Layers
    GRID["Net5"] = _BASE_GRID_PARAMS.copy()
    GRID["Net5"]["optimizer"] = ["Adam"]
    GRID["Net5"]["lr"] = _BASE_GRID_PARAMS["lr"][:2]
    GRID["Net5"]["n_hidden1"] = [96, 128]
    GRID["Net5"]["n_hidden2"] = [64, 96]
    GRID["Net5"]["n_hidden3"] = [48, 64]
    GRID["Net5"]["n_hidden4"] = [32, 48]
    GRID["Net5"]["n_hidden5"] = [16, 24]

    # Example for DNet1:
    GRID["DNet1"] = _BASE_GRID_PARAMS.copy()
    GRID["DNet1"]["optimizer"] = ["Adam"]
    GRID["DNet1"]["lr"] = _BASE_GRID_PARAMS["lr"][:2]
    GRID["DNet1"]["n_hidden1"] = [128, 192]
    GRID["DNet1"]["n_hidden2"] = [64, 128]
    GRID["DNet1"]["n_hidden3"] = [32, 64]
    GRID["DNet1"]["n_hidden4"] = [16, 32]

    # Example for DNet2:
    GRID["DNet2"] = _BASE_GRID_PARAMS.copy()
    GRID["DNet2"]["optimizer"] = ["Adam"]
    GRID["DNet2"]["lr"] = _BASE_GRID_PARAMS["lr"][:2]
    GRID["DNet2"]["n_hidden1"] = [128, 192, 256]
    GRID["DNet2"]["n_hidden2"] = [96, 128, 192]
    GRID["DNet2"]["n_hidden3"] = [64, 96, 128]
    GRID["DNet2"]["n_hidden4"] = [32, 48, 64]
    
    # Example for DNet3:
    GRID["DNet3"] = _BASE_GRID_PARAMS.copy()
    GRID["DNet3"]["optimizer"] = ["Adam"]
    GRID["DNet3"]["lr"] = _BASE_GRID_PARAMS["lr"][:2]
    GRID["DNet3"]["n_hidden1"] = [192, 256]
    GRID["DNet3"]["n_hidden2"] = [128, 192]
    GRID["DNet3"]["n_hidden3"] = [96, 128]
    GRID["DNet3"]["n_hidden4"] = [64, 96]
    GRID["DNet3"]["n_hidden5"] = [32, 48]
    print("--- GRID search spaces defined ---", file=sys.stderr)

except Exception as e:
    print(f"--- ERROR during execution of src.configs.search_spaces.py: {e} ---", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    # Re-raise the exception to ensure the import fails clearly
    raise

print("--- Finished executing src.configs.search_spaces.py ---", file=sys.stderr)