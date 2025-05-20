import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time # Import time
from sklearn.metrics import mean_squared_error
from skorch.regressor import NeuralNetRegressor
from skorch.callbacks import EarlyStopping
import skorch
import traceback # Import traceback
from functools import partial
import sys

# --- L1 Regularization Mixin ---
class L1Mixin:
    def __init__(self, l1_lambda=0.0, **kwargs):
        self.l1_lambda = l1_lambda
        # Ensure super().__init__ is called correctly by the class inheriting this
        super().__init__(**kwargs) # Pass remaining kwargs up

    def get_loss(self, y_pred, y_true, X=None, training=False):
        # Get base loss from the parent class (e.g., NeuralNetRegressor's MSE)
        base_loss = super().get_loss(y_pred, y_true, X=X, training=training)

        # Add L1 penalty during training if lambda > 0
        if self.l1_lambda > 0 and training:
            l1_penalty = 0.0
            # Iterate over parameters of the underlying torch module
            for param in self.module_.parameters():
                if param.requires_grad: # Only penalize trainable parameters
                    l1_penalty += torch.norm(param, 1)
            base_loss = base_loss + self.l1_lambda * l1_penalty
        return base_loss

# --- Custom Skorch Regressor with L1 ---
# Inherits from L1Mixin first, then NeuralNetRegressor
class L1Net(L1Mixin, NeuralNetRegressor):
     # __init__ is inherited from L1Mixin, which calls NeuralNetRegressor's __init__
     # get_loss is inherited from L1Mixin, which calls NeuralNetRegressor's get_loss
     pass

# Export L1Net as OptunaSkorchNet for external use in bayes_oos.py and other modules
OptunaSkorchNet = L1Net

# --- Objective Function for Optuna ---
def _objective(
    trial,
    model_module_class,         # PyTorch model class (e.g., Net1)
    skorch_net_class_to_use,    # Skorch wrapper (e.g., L1Net)
    hpo_config_fn_for_trial,    # Function that creates hyperparameters
    X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor,
    n_features_val, epochs_val, device_val, batch_size_default_val
):
    """Objective function for Optuna study."""
    # 1. Get hyperparameters from the config function
    params_from_optuna = hpo_config_fn_for_trial(trial, n_features_val)
    
    # 2. Extract non-module parameters (those not prefixed with "module__")
    module_specific_params = {k: v for k, v in params_from_optuna.items() if k.startswith("module__")}
    non_module_params = {k: v for k, v in params_from_optuna.items() if not k.startswith("module__")}
    
    # 3. Prepare optimizer and other configurations
    optimizer_name = non_module_params.pop("optimizer", "Adam")
    lr = non_module_params.pop("lr", 0.001)
    batch_size = non_module_params.pop("batch_size", batch_size_default_val or 128)
    weight_decay = non_module_params.pop("weight_decay", 0.0)
    l1_lambda = non_module_params.pop("l1_lambda", 0.0)
    
    # 4. Create Skorch neural network with proper parameters
    # IMPORTANT CHANGE: Pass n_feature and n_output to the module initialization
    net = skorch_net_class_to_use(
        module=model_module_class,
        module__n_feature=n_features_val,  # Add this line to fix the error
        module__n_output=1,                # Add this line to fix the error
        **module_specific_params,          # Pass module__n_hidden1, module__activation_hidden, etc.
        optimizer=getattr(torch.optim, optimizer_name),
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        l1_lambda=l1_lambda,
        batch_size=batch_size,
        max_epochs=epochs_val,
        device=device_val,
        # Add other necessary Skorch parameters
    )
    
    try:
        net.fit(X_train_tensor, y_train_tensor)
        validation_loss = net.history[-1, 'valid_loss'] # Optuna minimizes this by default
        return validation_loss
    except Exception as e:
        print(f"Error in Optuna objective for trial {trial.number}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return float('inf') # Return a large value if an error occurs, so Optuna prunes/ignores

def run_study(
    model_module,               # <<< ADDED: The PyTorch model class (e.g., nns.Net1)
    skorch_net_class,           # <<< ADDED: The Skorch wrapper (e.g., training_grid.GridNet)
    hpo_config_fn,              # Function from search_spaces.py (trial, n_features) -> params
    X_hpo_train, y_hpo_train,   # Training data for HPO
    X_hpo_val, y_hpo_val,       # Validation data for HPO
    trials,                     # Number of Optuna trials
    epochs,                     # Max epochs per trial
    device,
    batch_size_default,         # Default batch size if not in hpo_config_fn's output
    study_name_prefix="optuna_study", # For naming the study
    # metric_to_optimize="val_loss", # Or "cer_val" etc.
    # y_hpo_val_unscaled=None,    # For CER calculation
    # rf_hpo_val_unscaled=None,   # For CER calculation
    # gamma_cer=3.0               # For CER calculation
):
    """
    Runs an Optuna hyperparameter optimization study.
    """
    n_features = X_hpo_train.shape[1] # Infer n_features from training data

    # Use functools.partial to pass fixed arguments to the objective function
    objective_with_args = partial(
        _objective,
        model_module_class=model_module,
        skorch_net_class_to_use=skorch_net_class,
        hpo_config_fn_for_trial=hpo_config_fn,
        X_train_tensor=X_hpo_train, y_train_tensor=y_hpo_train,
        X_val_tensor=X_hpo_val, y_val_tensor=y_hpo_val,
        n_features_val=n_features, epochs_val=epochs, device_val=device,
        batch_size_default_val=batch_size_default
        # Pass y_val_unscaled etc. if optimizing CER
    )
    
    # direction = "minimize" if metric_to_optimize == "val_loss" else "maximize"
    direction = "minimize" # Defaulting to minimizing validation loss

    study_name_full = f"{study_name_prefix}_{model_module.__name__}"
    study = optuna.create_study(study_name=study_name_full, direction=direction, load_if_exists=False) # Set load_if_exists=True if using persistent storage
    
    try:
        study.optimize(objective_with_args, n_trials=trials, timeout=None) # Add timeout if needed
    except optuna.exceptions.TrialPruned:
        print("A trial was pruned.", file=sys.stderr)
    except Exception as e:
        print(f"Exception during Optuna study.optimize: {e}", file=sys.stderr)
        # Fallback: return empty dict or raise, depending on desired behavior
        return {}, None


    if not study.trials or study.best_trial is None: # Handle case where all trials fail or no trials run
        print(f"Warning: Optuna study for {study_name_full} completed with no successful trials or no best trial.", file=sys.stderr)
        return {}, study # Return empty params and the study object

    best_params_from_trial = study.best_trial.params
    
    # The hpo_config_fn in search_spaces.py (via _create_optuna_hpo_config_fn)
    # returns a flat dictionary of parameters as suggested by Optuna.
    # This flat dictionary is what `oos_common.py` expects for retraining.
    # It will extract 'optimizer', 'lr', 'batch_size' and pass the rest (module__*) to Skorch.
    
    print(f"Optuna study {study_name_full} best trial:")
    print(f"  Value (valid_loss): {study.best_trial.value}")
    print(f"  Params: {best_params_from_trial}")

    return best_params_from_trial, study # Return best params and the study object


def create_objective_function(model_class, regressor_class, hpo_config_fn_from_search_space, 
                           X_train, y_train, X_val, y_val, epochs, device, batch_size=128,
                           scoring='neg_mean_squared_error', use_early_stopping=True, 
                           patience=10, early_stopping_delta=0.001):
    """
    Creates an Optuna objective function for use in OOS experiments.
    This is a wrapper around _objective to match the interface expected in oos_common.py.
    
    Returns a callable objective function that Optuna can use.
    """
    def objective_fn(trial):
        # Use the existing _objective function with the parameters from this closure
        return _objective(
            trial=trial,
            model_module_class=model_class,
            skorch_net_class_to_use=regressor_class,
            hpo_config_fn_for_trial=hpo_config_fn_from_search_space,
            X_train_tensor=X_train, 
            y_train_tensor=y_train,
            X_val_tensor=X_val, 
            y_val_tensor=y_val,
            n_features_val=X_train.shape[1], 
            epochs_val=epochs, 
            device_val=device,
            batch_size_default_val=batch_size
        )
    
    return objective_fn
