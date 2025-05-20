from itertools import product
from sklearn.metrics import mean_squared_error
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping
import numpy as np, torch
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import sys # For stderr

class GridNet(NeuralNetRegressor):
    def __init__(self,*a,l1_lambda=0.0,**kw):super().__init__(*a,**kw);self.l1_lambda=l1_lambda
    def get_loss(self,y_pred,y_true,*_,**__):
        loss=super().get_loss(y_pred,y_true);l1=sum(p.abs().sum() for p in self.module_.parameters());return loss+self.l1_lambda*l1/len(y_true)

def train_grid(
    model_module,      # The PyTorch model class
    regressor_class,   # The regressor class
    search_space_config,  # Grid parameters
    X_train, y_train, X_val, y_val,
    n_features, epochs, device,
    batch_size_default=128
):
    """
    Performs grid search for hyperparameter optimization.
    
    Args:
        model_module: The neural network class (e.g., nns.Net1).
        regressor_class: The regressor class (added for compatibility with other HPO methods).
        search_space_config: Dictionary defining the hyperparameter grid.
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        n_features: Number of input features.
        epochs: Maximum epochs for training.
        device: Device to use ('cpu' or 'cuda').
        batch_size_default: Default batch size if not specified in grid.
        
    Returns:
        Tuple of (best_params, best_estimator) - matches format of other HPO methods.
    """
    best_hp_for_return = None
    best_net_object = None # Or 'best = None' if that's your variable name

    # Skorch estimator
    # Ensure regressor_class is correctly instantiated.
    # It might be GridNet or a similar Skorch wrapper.
    net = regressor_class(
        module=model_module,
        module__n_feature=n_features,
        module__n_output=1,
        max_epochs=epochs,
        # Other necessary params for Skorch like default lr, optimizer if not in grid,
        # or ensure they are always in search_space.
        # Example:
        # optimizer=torch.optim.Adam, # Default optimizer if not in grid
        # lr=0.01, # Default lr if not in grid
        device=device,
        train_split=None, # We are providing a manual CV split
        verbose=0 # Set to 1 or higher for more GridSearchCV output
    )

    # Ensure search_space keys match what Skorch and your module expect.
    # e.g., 'lr', 'optimizer__weight_decay', 'module__dropout', 'module__n_hidden1', 'batch_size'
    
    # The CV split: one split using (X_train, y_train) for training and (X_val, y_val) for validation
    # Indices for X_train
    train_indices = np.arange(X_train.shape[0])
    # Indices for X_val, offset by the length of X_train because we'll vstack them
    val_indices = np.arange(X_train.shape[0], X_train.shape[0] + X_val.shape[0])
    
    custom_cv = [(train_indices, val_indices)]
    
    # Combine X_train and X_val for GridSearchCV, as it expects full X, y
    X_combined = np.vstack((X_train, X_val))
    y_combined = np.vstack((y_train, y_val))


    gs = GridSearchCV(
        estimator=net,
        param_grid=search_space_config,
        scoring=make_scorer(mean_squared_error, greater_is_better=False), # Negative MSE
        cv=custom_cv,
        refit=True, # Refits the best estimator on the whole training part of the custom_cv split
        verbose=2, # <<< INCREASED VERBOSITY (try 2, then 3 if needed)
        error_score='raise'
    )

    try:
        gs.fit(X_combined, y_combined)
        
        best_hp_for_return = gs.best_params_
        best_net_object = gs.best_estimator_ # This is the refitted best Skorch net
        print(f"Grid Search for {model_module.__name__} best score (MSE): {-gs.best_score_ if gs.best_score_ is not None else 'N/A'}", file=sys.stderr)
        print(f"Grid Search for {model_module.__name__} best HPs: {best_hp_for_return}", file=sys.stderr)

    except Exception as e:
        print(f"Error during GridSearchCV for {model_module.__name__}: {e}", file=sys.stderr)
        # best_hp_for_return and best_net_object will remain None, which is handled by oos_common.py
    
    return best_hp_for_return, best_net_object # Ensure the second variable matches what oos_common expects