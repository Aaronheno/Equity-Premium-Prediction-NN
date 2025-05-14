import argparse, importlib, sys, pathlib, torch
from src.models import nns

# ensure project root is on sys.path when invoked via "python -m src.cli"
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Define available models for each method (adjust as needed)
# MODEL_LISTS = { # <<< REMOVE THIS OLD DICTIONARY
#     "bayes": ["Net1", "Net2", "Net3", "Net4", "Net5", "DNet1", "DNet2", "DNet3"],
#     "grid": ["Net1", "Net2"],
#     "random": ["Net1", "Net2", "Net3"],
# }

DEFAULT_MODELS_ALL = ["Net1", "Net2", "Net3", "Net4", "Net5", "DNet1", "DNet2", "DNet3"]

DEFAULT_MODELS_RANDOM = DEFAULT_MODELS_ALL[:]
DEFAULT_MODELS_GRID = DEFAULT_MODELS_ALL[:]
DEFAULT_MODELS_BAYES = DEFAULT_MODELS_ALL[:] # Ensure this uses DNN consistently

def main():
    parser = argparse.ArgumentParser(description="Launch equity‑premium NN experiments")
    parser.add_argument("--method", choices=["bayes", "grid", "random"], default="bayes",
                        help="training strategy (Bayesian / Grid / Random)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="model names, e.g. NN1 NN2 DNet1. Overrides default list for the method.")
    parser.add_argument("--trials", type=int, default=50,
                        help="hyper‑parameter trials (for Bayes/Random) or grid cells")
    parser.add_argument("--epochs", type=int, default=100,
                        help="max epochs per trial (early‑stopping cuts earlier)")
    parser.add_argument("--threads", type=int, default=1,
                        help="PyTorch intra‑op CPU threads")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--batch", type=int, default=256,
                        help="Batch size for training.")
    parser.add_argument("--gamma", type=float, default=3.0,
                        help="Risk aversion coefficient for CER calculation.")
    args = parser.parse_args()

    # Determine models to run
    models_to_run = args.models
    if not models_to_run: # If --models flag was not used
        if args.method == "random":
            models_to_run = DEFAULT_MODELS_RANDOM
        elif args.method == "grid":
            models_to_run = DEFAULT_MODELS_GRID
        elif args.method == "bayes":
            models_to_run = DEFAULT_MODELS_BAYES
        else:
            models_to_run = [] # Should not happen if choices are enforced by argparse

    if not models_to_run:
        print(f"Error: No models defined for method '{args.method}' and --models not specified.", file=sys.stderr)
        return 1

    # Determine device
    if args.device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        selected_device = args.device
    print(f"--- Using device: {selected_device} ---")

    # Dynamically import the module based on the method
    mod_name = f"src.experiments.{args.method}"
    try:
        module = importlib.import_module(mod_name)
    except ModuleNotFoundError:
        print(f"Error: Experiment module '{mod_name}' not found.", file=sys.stderr)
        return 1

    # Prepare arguments for the run function
    run_kwargs = {
        "models": models_to_run,
        "trials": args.trials,
        "epochs": args.epochs,
        "threads": args.threads,
        "batch": args.batch,
        "device": selected_device,
    }
    if args.method == "bayes":
        run_kwargs["gamma"] = args.gamma
    elif args.method == "random":
        run_kwargs["gamma_cer"] = args.gamma
    elif args.method == "grid":
        run_kwargs["gamma_cer"] = args.gamma

    # Run the experiment
    module.run(**run_kwargs)
    return 0

if __name__ == "__main__":
    sys.exit(main())