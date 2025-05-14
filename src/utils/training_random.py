import random
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
# Use GridNet from training_grid for consistency with L1 regularization
from src.utils.training_grid import GridNet # Assuming GridNet is appropriate
from skorch.callbacks import EarlyStopping
from src.utils.distributions import CategoricalDistribution, FloatDistribution, IntDistribution # Ensure this import is present

def _sample_param(config):
    """Helper to sample a single parameter value based on its config."""
    if isinstance(config, (CategoricalDistribution, FloatDistribution, IntDistribution)):
        return config.sample() # Use the sample() method of our custom distribution classes
    elif isinstance(config, list):
        # If the list itself contains a single tuple (min, max), sample from that range
        if len(config) == 1 and isinstance(config[0], tuple) and len(config[0]) == 2:
            val_range = config[0]
            if all(isinstance(v, int) for v in val_range):
                return random.randint(val_range[0], val_range[1])
            elif any(isinstance(v, float) for v in val_range):
                return random.uniform(val_range[0], val_range[1])
            else: # Fallback for mixed types or other tuple contents
                return random.choice(val_range) # Or raise error
        else: # Standard list of discrete choices
            return random.choice(config)
    elif isinstance(config, tuple) and len(config) == 2:
        # Direct (min, max) tuple
        if all(isinstance(v, int) for v in config):
            return random.randint(config[0], config[1])
        elif any(isinstance(v, float) for v in config):
            return random.uniform(config[0], config[1])
        else: # Fallback
            raise ValueError(f"Unsupported tuple parameter configuration: {config}")
    # If it's a single fixed value (not typical for search space but good to handle)
    elif isinstance(config, (int, float, str, bool)):
        return config
    else:
        raise ValueError(f"Unsupported parameter configuration: {config}")

def train_random(model_class, X_tr, y_tr, X_val, y_val, space, epochs, trials, device="cpu"):
    """
    Performs random search for hyperparameter optimization.

    Args:
        model_class: The neural network class (e.g., nns.Net1).
        X_tr, y_tr: Training data (numpy arrays).
        X_val, y_val: Validation data (numpy arrays).
        space (dict): Dictionary defining the hyperparameter search space.
                      Values can be lists of choices or tuples (min, max) for ranges.
        epochs (int): Maximum number of epochs for training each trial.
        trials (int): Number of random hyperparameter combinations to try.
        device (str): Device to use for training ('cpu' or 'cuda').

    Returns:
        tuple: (best_net, best_hyperparams, best_validation_mse)
    """
    best_net = None
    best_hyperparams = None
    best_validation_mse = float("inf")

    print(f"--- Starting Random Search: {trials} trials, {epochs} epochs each ---")

    for i in range(trials):
        current_hyperparams = {}
        print(f"\nTrial {i+1}/{trials}: Sampling hyperparameters...")
        for key, config in space.items():
            current_hyperparams[key] = _sample_param(config)
        
        print(f"  Hyperparameters: {current_hyperparams}")

        # Extract skorch-specific and module-specific params
        # Similar to how it's done in bayes.py or training_grid.py
        optimizer_name = current_hyperparams.pop("optimizer", "Adam")
        lr = current_hyperparams.pop("lr", 1e-3)
        weight_decay = current_hyperparams.pop("weight_decay", 0.0)
        l1_lambda = current_hyperparams.pop("l1_lambda", 0.0)
        batch_size = current_hyperparams.pop("batch_size", 256) # Default if not in space

        # Module parameters (n_hidden, dropout, etc.) remain in current_hyperparams
        module_params = {f"module__{k}": v for k, v in current_hyperparams.items()}

        try:
            # Using GridNet as it handles L1 and is similar to what train_grid uses
            # If L1 is not always desired, could switch to NeuralNetRegressor
            # and conditionally add L1Mixin or handle l1_lambda inside.
            net = GridNet(
                module=model_class,
                module__n_feature=X_tr.shape[1],
                module__n_output=1, # Assuming regression with 1 output
                **module_params,     # Pass sampled module params like module__n_hidden1, module__dropout
                max_epochs=epochs,
                batch_size=batch_size,
                optimizer=getattr(torch.optim, optimizer_name),
                lr=lr,
                optimizer__weight_decay=weight_decay,
                l1_lambda=l1_lambda, # GridNet specific
                iterator_train__shuffle=True, # Shuffle training data
                callbacks=[EarlyStopping(patience=10, monitor="valid_loss", lower_is_better=True)],
                device=device,
                verbose=0 # Suppress skorch's own epoch printing for cleaner trial logs
            )

            # Skorch expects numpy arrays for X and y (or DataFrames)
            # y needs to be shape (n_samples, 1) or (n_samples,) for regressor
            y_tr_fit = y_tr.reshape(-1, 1) if y_tr.ndim == 1 else y_tr
            
            net.fit(X_tr, y_tr_fit) # y_val is used by EarlyStopping via valid_ds

            # Predict on validation set
            y_val_pred_scaled = net.predict(X_val)
            current_mse = mean_squared_error(y_val, y_val_pred_scaled)
            print(f"  Trial {i+1} Validation MSE: {current_mse:.6f}")

            if current_mse < best_validation_mse:
                best_validation_mse = current_mse
                best_hyperparams = current_hyperparams # Store the original sampled params
                best_hyperparams.update({ # Add back optimizer params for record
                    "optimizer": optimizer_name, "lr": lr, 
                    "weight_decay": weight_decay, "l1_lambda": l1_lambda,
                    "batch_size": batch_size
                })
                best_net = net # Store the trained skorch net
                print(f"  New best MSE found: {best_validation_mse:.6f}")

        except Exception as e:
            print(f"  Error in Trial {i+1} with params {current_hyperparams}: {e}")
            import traceback
            traceback.print_exc()
            # Optionally, continue to the next trial or handle error

    if best_net:
        print(f"\n--- Random Search Complete ---")
        print(f"Best Validation MSE: {best_validation_mse:.6f}")
        print(f"Best Hyperparameters: {best_hyperparams}")
    else:
        print("\n--- Random Search Complete: No successful trials ---")

    return best_net, best_hyperparams, best_validation_mse
