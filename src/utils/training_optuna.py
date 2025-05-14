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

# --- Objective Function for Optuna ---
def objective(trial, model_name, X_tr, y_tr, X_val, y_val, model_class, space, epochs, batch, device):
    """Objective function for Optuna study."""
    # --- Hyperparameter Sampling ---
    lr = trial.suggest_float("lr", *space["lr"])
    dropout = trial.suggest_float("dropout", *space["dropout"])

    model_params = {
        "lr": lr, # Keep lr for optimizer later
        "dropout": dropout,
    }

    # Determine required hidden args explicitly per model type
    # This needs to be precise based on your nns.py definitions
    num_hidden_layers = 0
    if model_name == "Net1": num_hidden_layers = 1
    elif model_name == "Net2": num_hidden_layers = 2
    elif model_name == "Net3": num_hidden_layers = 3
    elif model_name == "Net4": num_hidden_layers = 4
    elif model_name == "Net5": num_hidden_layers = 5
    elif model_name == "DNet1": num_hidden_layers = 4 # DNet1 expects n_hidden1 to n_hidden4
    elif model_name == "DNet2": num_hidden_layers = 4 # DNet2 expects n_hidden1 to n_hidden4
    elif model_name == "DNet3": num_hidden_layers = 5 # DNet3 expects n_hidden1 to n_hidden5
    else:
        print(f"Warning: Unknown model name '{model_name}' for determining hidden layers.")
        num_hidden_layers = 0 # Default, likely causing init error

    # print(f"[{model_name} Trial {trial.number}] Determined num_hidden_layers: {num_hidden_layers}") # DEBUG
    # print(f"[{model_name} Trial {trial.number}] Space keys: {list(space.keys())}") # DEBUG

    # Sample each required hidden layer dimension
    for i in range(1, num_hidden_layers + 1):
        key = f"n_hidden{i}"
        # print(f"[{model_name} Trial {trial.number}] Checking for key: '{key}'") # DEBUG
        if key in space:
            # print(f"[{model_name} Trial {trial.number}] Found key '{key}'. Sampling...") # DEBUG
            model_params[key] = trial.suggest_int(key, *space[key])
        else:
            # This is where the error likely happens for DNet if keys aren't found
            print(f"Error: Search space for {model_name} missing required key '{key}'")
            raise optuna.TrialPruned(f"Missing search key: {key}")

    # --- Data Preparation ---
    try:
        X_tr_tensor = torch.tensor(X_tr, dtype=torch.float32).to(device)
        y_tr_tensor = torch.tensor(y_tr, dtype=torch.float32).to(device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

        train_dataset = TensorDataset(X_tr_tensor, y_tr_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    except Exception as e:
        print(f"Error creating tensors/dataloaders: {e}")
        return float('inf')

    # --- Model Initialization ---
    model_params["n_feature"] = X_tr.shape[1]
    model_params["n_output"] = 1
    init_params = {k: v for k, v in model_params.items() if k != 'lr'}

    # print(f"[{model_name} Trial {trial.number}] Initializing model with params: {init_params}") # DEBUG

    try:
        net = model_class(**init_params).to(device)
    except TypeError as e:
        print(f"!!! TypeError during model initialization for {model_name}: {e}")
        print(f"    Parameters passed: {init_params}") # Print what was actually passed
        raise optuna.TrialPruned(f"Model init TypeError: {e}")
    except Exception as e:
        print(f"!!! Unexpected error during model initialization for {model_name}: {e}")
        traceback.print_exc()
        raise optuna.TrialPruned(f"Model init Error: {e}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=model_params["lr"]) # Use the sampled LR

    # --- Training Loop ---
    net.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = net(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        # Optional: Add pruning based on intermediate validation loss
        # net.eval()
        # with torch.no_grad():
        #     val_outputs = net(X_val_tensor)
        #     intermediate_val_loss = criterion(val_outputs, y_val_tensor).item()
        # net.train()
        # trial.report(intermediate_val_loss, epoch)
        # if trial.should_prune():
        #     raise optuna.TrialPruned()

    # --- Validation ---
    net.eval()
    with torch.no_grad():
        val_outputs = net(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor).item()

    return val_loss # Return validation loss for Optuna

# --- Main Study Function ---
def run_study(model_name, X_tr, y_tr, X_val, y_val, X_ALL, y_ALL, model_class, space, trials, epochs, batch, device):
    """Runs the Optuna study and fits the final model."""
    study_name = f"optuna-study-{model_name}-{device}"
    # study = optuna.create_study(direction="minimize", study_name=study_name) # Original
    # Use storage=None for default in-memory storage, avoids potential conflicts if study_name reused quickly
    study = optuna.create_study(direction="minimize", study_name=study_name, storage=None)

    try:
        study.optimize(
            lambda trial: objective(trial, model_name, X_tr, y_tr, X_val, y_val, model_class, space, epochs, batch, device),
            n_trials=trials,
            # n_jobs=threads # Be careful with n_jobs and CUDA
        )
    except optuna.exceptions.TrialPruned as e:
        # This might catch pruning from within objective, but optimize usually handles it
        print(f"A trial was pruned during optimization: {e}")
    except Exception as e:
         print(f"Error during Optuna optimization: {e}")
         traceback.print_exc()
         print(f"Study {study_name} failed for model {model_name}.")
         # Ensure we return the failure signature
         return None, None, float('inf')

    print(f"\nStudy {study_name} finished for model {model_name}.")

    # --- Robust check for best trial ---
    best_val_mse = float('inf')
    best_hp = None
    try:
        # Check if there are any trials AT ALL first
        if not study.trials:
            print("Error: No trials were even started.")
        else:
            # Accessing best_trial might raise ValueError if no trials completed
            best_trial = study.best_trial
            best_hp = best_trial.params
            best_val_mse = best_trial.value if best_trial.value is not None else float('inf')
            print(f"Best trial for {model_name} (State: {best_trial.state}):")
            print(f"  MSE: {best_val_mse:.6f}")
            print(f"  Params: {best_hp}")

            # Explicitly check if the best trial actually completed successfully
            if best_trial.state != optuna.trial.TrialState.COMPLETE:
                print(f"Warning: Best trial (Trial {best_trial.number}) did not complete successfully (State: {best_trial.state}). Final model may not be optimal.")
                # Decide if you still want to proceed with fitting based on a failed/pruned trial's params
                # For now, we will proceed but the warning is important.

    except ValueError as e:
        # This specifically catches the "No trials are completed yet" error
        print(f"Error retrieving best trial: {e}. Cannot proceed with final model fitting.")
        # Attempt to find the best value among *all* trials (even failed/pruned)
        best_value_so_far = float('inf')
        best_params_so_far = None
        for t in study.trials:
            if t.value is not None and t.value < best_value_so_far:
                best_value_so_far = t.value
                best_params_so_far = t.params
        if best_params_so_far:
             print(f"  Best value found among all trials (including pruned/failed): {best_value_so_far:.6f}")
             print(f"  Params for best value: {best_params_so_far}")
             return None, best_params_so_far, best_value_so_far # Return HPs and value, but no model
        else:
             return None, None, float('inf') # No usable trial found at all
    except Exception as e:
        print(f"Unexpected error retrieving best trial: {e}")
        traceback.print_exc()
        return None, None, float('inf') # General failure

    # If best_hp is None here, it means we couldn't get parameters to fit a final model
    if best_hp is None:
        print("Could not determine best hyperparameters. Skipping final model fit.")
        return None, None, best_val_mse # Return best MSE found (maybe inf), but no model/hp

    # --- Final Model Fit on ALL Scaled Data ---
    print(f"Fitting final {model_name} model on the entire scaled dataset...")
    # Prepare full dataset (already NumPy arrays)
    try:
        X_ALL_tensor = torch.tensor(X_ALL, dtype=torch.float32).to(device)
        y_ALL_tensor = torch.tensor(y_ALL, dtype=torch.float32).to(device)
        full_dataset = TensorDataset(X_ALL_tensor, y_ALL_tensor)
        # Use a larger batch size for final fit? Or same as trial? Using same for now.
        full_loader = DataLoader(full_dataset, batch_size=batch, shuffle=True)
    except Exception as e:
        print(f"Error creating final tensors/dataloaders: {e}")
        return None, best_hp, best_val_mse # Return HPs/MSE, but no model

    # Construct final model params (ensure this logic is robust)
    final_model_params = {
        "n_feature": X_ALL.shape[1],
        "n_output": 1,
        "dropout": best_hp.get("dropout", 0.0),
    }
    num_hidden_layers_final = 0
    for i in range(1, 6): # Check up to 5 layers based on keys in best_hp
        key = f"n_hidden{i}"
        if key in best_hp:
            final_model_params[key] = best_hp[key]
            num_hidden_layers_final = i
        else:
            # If the key isn't in best_hp, we assume the model doesn't need more layers
            break

    # Add a check: Does the number of layers found match expectation?
    expected_layers = 0
    if model_name == "Net1": expected_layers = 1
    elif model_name == "Net2": expected_layers = 2
    # ... add all model types ...
    elif model_name == "DNet1": expected_layers = 4
    elif model_name == "DNet2": expected_layers = 4
    elif model_name == "DNet3": expected_layers = 5

    if num_hidden_layers_final != expected_layers:
        print(f"!!! WARNING: Final model parameter mismatch for {model_name}. Expected {expected_layers} hidden layers based on name, but found {num_hidden_layers_final} in best_hp keys: {list(best_hp.keys())}")
        # This might indicate an issue with the objective function logic or search space
        # Decide how to handle: proceed with caution, or return failure?
        # Let's proceed for now, but this warning is critical.

    try:
        final_net = model_class(**final_model_params).to(device)
    except TypeError as e:
        print(f"!!! TypeError during FINAL model initialization for {model_name}: {e}")
        print(f"    Parameters passed: {final_model_params}")
        return None, best_hp, best_val_mse
    except Exception as e:
        print(f"!!! Unexpected error during FINAL model initialization for {model_name}: {e}")
        traceback.print_exc()
        return None, best_hp, best_val_mse

    criterion = nn.MSELoss()
    # Use the best learning rate found
    final_lr = best_hp.get("lr", 0.001) # Default LR if not found? Should be there.
    optimizer = optim.Adam(final_net.parameters(), lr=final_lr)

    # Final training loop
    final_net.train()
    start_final_fit = time.time()
    # Use same number of epochs as in trials? Or potentially more/fewer? Using same for now.
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in full_loader:
            optimizer.zero_grad()
            outputs = final_net(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        # Simple progress print
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
             print(f"  Final Fit Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(full_loader):.6f}")

    end_final_fit = time.time()
    print(f"Final model fitting complete. Time: {end_final_fit - start_final_fit:.2f}s")

    final_net.eval() # Set to evaluation mode before returning

    return final_net, best_hp, best_val_mse
