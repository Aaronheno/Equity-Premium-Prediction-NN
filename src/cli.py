"""
CLI Interface for Equity Premium Prediction Neural Networks

This module provides a comprehensive command-line interface for running neural network
experiments with equity premium prediction. Supports massive parallelization and
hardware optimization for systems ranging from 4-core laptops to 128+ core HPC servers.

Threading Status: THREAD_COORDINATING_SAFE
Hardware Requirements: CPU_REQUIRED, CUDA_PREFERRED, MULTI_GPU_SUPPORTED
Performance Notes: 
    - Automatic hardware detection and optimization
    - Supports 5-100x speedup with proper parallelization
    - Server mode for HPC clusters (64+ cores)
    - Memory management for high-RAM systems (256GB+)

Multithreading Features:
    - Model-level parallelism (8 models simultaneously)
    - HPO trial parallelism (100+ concurrent trials)
    - Nested parallelism (models × trials × windows)
    - Automatic resource allocation and scaling

Usage Examples:
    # Standard workstation (8-16 cores)
    python -m src.cli run --method bayes_oos --models Net1 Net2 Net3
    
    # High-performance server (128+ cores)
    python -m src.cli run --method bayes_oos --server-mode --nested-parallelism
    
    # Cloud deployment with resource limits
    python -m src.cli run --method bayes_oos --max-cores 64 --memory-gb 128

Threading Safety:
    - All experiment orchestration is thread-safe
    - Resource allocation prevents race conditions
    - Graceful degradation on resource constraints
    - Backward compatibility with single-threaded mode
"""

# src/cli.py    
import argparse
import importlib
import sys
import pathlib 
import torch
import os

# Fix OpenMP conflict issue common with Anaconda + PyTorch
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Enable parallel capabilities for window experiments (safe injection)
try:
    from src.utils.parallel_injection import inject_parallel_capabilities
    # This is already called on import, but we can ensure it's done
except ImportError:
    # If parallel injection not available, system works normally
    pass

# --- Ensure project root is on sys.path ---
# This must be done BEFORE any 'from src...' imports.
# If cli.py is in 'src/', then project_root is one level up (parents[1]).
_project_root = pathlib.Path(__file__).resolve().parents[1] 
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
# --- End sys.path modification ---

# Default model lists for convenience
DEFAULT_MODELS_ALL = ["Net1", "Net2", "Net3", "Net4", "Net5", "DNet1", "DNet2", "DNet3"]

DEFAULT_MODELS_IN_SAMPLE_RANDOM = DEFAULT_MODELS_ALL[:]
DEFAULT_MODELS_IN_SAMPLE_GRID = ["Net1", "Net2"] 
DEFAULT_MODELS_IN_SAMPLE_BAYES = DEFAULT_MODELS_ALL[:]

DEFAULT_MODELS_OOS_RANDOM = DEFAULT_MODELS_ALL[:]
DEFAULT_MODELS_OOS_GRID = ["Net1", "Net2"] 
DEFAULT_MODELS_OOS_BAYES = DEFAULT_MODELS_ALL[:]

def main():
    parser = argparse.ArgumentParser(description="Launch Equity Premium NN Experiments and Analysis")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Main experiment command (existing functionality)
    run_parser = subparsers.add_parser("run", help="Run equity premium prediction experiments")
    method_choices = [
        "bayes", "grid", "random",
        "bayes_oos", "grid_oos", "random_oos",
        "rolling_grid", "rolling_random", "rolling_bayes",
        "expanding_grid", "expanding_random", "expanding_bayes",
        "grid_mae", "random_mae", "bayes_mae",
        "grid_mae_oos", "random_mae_oos", "bayes_mae_oos",
        "newly_identified", "grid_oos_6", "random_oos_6", "bayes_oos_6",
        "fred_variables", "grid_oos_7", "random_oos_7", "bayes_oos_7",
        "variable_importance_8", "gamma_sensitivity_9", "profit_optimization_10", "profit_oos_10"
    ]
    run_parser.add_argument(
        "--method", 
        choices=method_choices, 
        default="bayes_oos", 
        help="Training and evaluation strategy."
    )
    run_parser.add_argument(
        "--models", 
        nargs="+", 
        default=None, 
        help=(
            "List of models to run (e.g., Net1 Net2 DNN1). "
            "If not specified, a default list for the chosen method is used."
        )
    )
    run_parser.add_argument(
        "--trials", 
        type=int, 
        default=50, 
        help="Number of trials for Random Search or Bayesian Optimization (per HPO cycle for OOS)."
    )
    run_parser.add_argument(
        "--epochs", 
        type=int, 
        default=75, 
        help="Number of epochs for training each model configuration (per HPO trial and for final annual OOS model)."
    )
    run_parser.add_argument(
        "--batch", 
        type=int, 
        default=128, 
        help="Batch size for training."
    )
    run_parser.add_argument(
        "--device", 
        choices=["cpu", "cuda", "auto"], 
        default="auto", 
        help="Device to use for training ('auto' uses CUDA if available)."
    )
    
    run_parser.add_argument(
        "--oos-start-date",
        type=int,
        default=200001, 
        help="Start date for OOS evaluation in YYYYMM format (e.g., 195701)."
    )
    run_parser.add_argument(
        "--oos-end-date",
        type=int,
        default=None, 
        help="End date for OOS evaluation in YYYYMM format (e.g., 202112). If not specified, uses all available data."
    )
    run_parser.add_argument(
        "--window-sizes",
        type=str,
        default="5,10,20",
        help="Comma-separated list of window sizes in years for rolling window analysis."
    )
    run_parser.add_argument(
        "--threads", 
        type=int, 
        default=4, 
        help="Number of threads (primarily for older grid/random in-sample scripts if they use it)."
    )
    run_parser.add_argument(
        "--gamma", 
        type=float, 
        default=3.0, 
        help="Risk aversion coefficient for CER calculation (primarily for in-sample bayes.py)."
    )
    run_parser.add_argument(
        "--integration-mode",
        choices=["standalone", "integrated"],
        default="standalone",
        help="Integration mode for newly identified variables: standalone (only new variables) or integrated (combined with existing)"
    )
    run_parser.add_argument(
        "--optimization-method",
        choices=["grid", "random", "bayes"],
        default="grid",
        help="Optimization method to use with newly identified or FRED variables"
    )
    run_parser.add_argument(
        "--data-source",
        choices=["original", "newly_identified", "fred"],
        default="original",
        help="Data source to use for variable importance analysis"
    )
    run_parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Number of parallel jobs for variable importance calculation or gamma sensitivity analysis"
    )
    run_parser.add_argument(
        "--gamma-range", 
        type=str, 
        default=None,
        help="Custom range of gamma values for sensitivity analysis in format 'start,end,num_points' (e.g., '0.5,10,20')"
    )
    run_parser.add_argument(
        "--rf-rate-default", 
        type=float, 
        default=0.03,
        help="Default annual risk-free rate (used only if actual rates unavailable)"
    )
    run_parser.add_argument(
        "--run-dir", 
        type=str, 
        default=None,
        help="Specific run directory to analyze for gamma sensitivity or profit evaluation"
    )
    
    run_parser.add_argument(
        "--max-leverage", 
        type=float, 
        default=1.5,
        help="Maximum leverage allowed for profit optimization (1.5 = 150%)"
    )
    
    run_parser.add_argument(
        "--transaction-cost", 
        type=float, 
        default=0.0007,
        help="Transaction cost per trade for profit optimization (0.0007 = 0.07%)"
    )
    
    run_parser.add_argument(
        "--position-sizing", 
        choices=['binary', 'proportional'], 
        default='binary',
        help="Position sizing method for profit optimization"
    )
    
    run_parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Print detailed progress information during execution"
    )
    
    # Performance optimization arguments (safe defaults, opt-in parallelization)
    run_parser.add_argument(
        "--hpo-jobs",
        type=int,
        default=None,
        help="Number of parallel HPO jobs (default: 1 for backward compatibility, 'auto' with --parallel-trials)"
    )
    run_parser.add_argument(
        "--parallel-trials",
        action="store_true",
        help="Enable parallel HPO trials (automatically sets safe hpo-jobs if not specified)"
    )
    run_parser.add_argument(
        "--parallel-models",
        action="store_true",
        help="Enable parallel model processing within time steps (experimental)"
    )
    run_parser.add_argument(
        "--max-cores",
        type=int,
        default=None,
        help="Maximum CPU cores to use (default: auto-detect safe limit)"
    )
    run_parser.add_argument(
        "--server-mode",
        action="store_true",
        help="Enable aggressive optimizations for high-core server environments"
    )
    run_parser.add_argument(
        "--resource-info",
        action="store_true",
        help="Display system resource information and recommended settings"
    )
    run_parser.add_argument(
        "--parallel-windows",
        action="store_true",
        help="Enable parallel window size processing (experimental, for rolling/expanding window experiments)"
    )
    run_parser.add_argument(
        "--nested-parallelism",
        action="store_true", 
        help="Enable nested parallelism (models + windows + HPO) for HPC systems (advanced)"
    )
    
    # Economic value analysis command (new functionality)
    econ_parser = subparsers.add_parser("economic-value", help="Run economic value analysis on model predictions")
    econ_parser.add_argument(
        "--runs-dir",
        type=str,
        default="./runs",
        help="Directory containing OOS experiment results"
    )
    econ_parser.add_argument(
        "--data-path", 
        type=str,
        default="./data",
        help="Path to directory with raw data files"
    )
    econ_parser.add_argument(
        "--output-path", 
        type=str,
        default="./runs/2_Economic_Value_Analysis",
        help="Path to save analysis results"
    )
    econ_parser.add_argument(
        "--window-sizes", 
        type=str, 
        default="1,3",
        help="Comma-separated list of expanding window sizes (in years)"
    )
    econ_parser.add_argument(
        "--oos-start-date",
        type=int,
        default=199001, 
        help="Start date for out-of-sample period in YYYYMM format (e.g., 199001)"
    )
    econ_parser.add_argument(
        "--force", 
        action="store_true",
        help="Force re-analysis of previously analyzed results"
    )
    econ_parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Print detailed progress information"
    )
    
    # Collect OOS results command (new functionality)
    collect_parser = subparsers.add_parser("collect-oos-results", help="Collect and format OOS results for economic analysis")
    collect_parser.add_argument(
        "--base-dir",
        type=str,
        default="./runs",
        help="Base directory to search for OOS results"
    )
    collect_parser.add_argument(
        "--output-file", 
        type=str,
        default="./combined_predictions.csv",
        help="Path to save the combined predictions"
    )
    collect_parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Print detailed progress information"
    )

    args = parser.parse_args()
    
    # Default to 'run' command if no command specified
    if args.command is None:
        args.command = "run"

    # Handle new commands for economic analysis
    if args.command == "collect-oos-results":
        from src.experiments.economic_value_2 import convert_oos_results_to_economic_format
        
        print(f"Collecting OOS results from {args.base_dir}")
        combined_df = convert_oos_results_to_economic_format(
            oos_results_dir=args.base_dir,
            output_file=args.output_file,
            verbose=args.verbose
        )
        
        if not combined_df.empty:
            print(f"Successfully collected OOS results from {len(combined_df.columns)} models")
            print(f"Results saved to {args.output_file}")
            return 0
        else:
            print("Error: Failed to collect OOS results")
            return 1
            
    elif args.command == "economic-value":
        from src.experiments.economic_value_2 import run_economic_value_analysis
        
        # Process window sizes
        window_sizes = [int(w) for w in args.window_sizes.split(',')]
        
        # Check if data directory exists
        if not os.path.exists(args.data_path):
            print(f"Error: Data directory {args.data_path} not found")
            return 1
            
        # Check if collected predictions exist
        prediction_file = args.runs_dir
        if os.path.isdir(prediction_file):
            # If it's a directory, we need to collect results first
            from src.experiments.economic_value_2 import convert_oos_results_to_economic_format
            print(f"First collecting OOS results from {args.runs_dir}")
            temp_output_file = os.path.join(args.output_path, "combined_predictions.csv")
            os.makedirs(os.path.dirname(temp_output_file), exist_ok=True)
            combined_df = convert_oos_results_to_economic_format(
                oos_results_dir=args.runs_dir,
                output_file=temp_output_file,
                verbose=args.verbose
            )
            if combined_df.empty:
                print("Error: Failed to collect OOS results")
                return 1
            prediction_file = temp_output_file
            
        print(f"Running economic value analysis with window sizes: {window_sizes}")
        print(f"OOS start date: {args.oos_start_date}")
        
        results = run_economic_value_analysis(
            predictions_file=prediction_file,
            raw_data_path=args.data_path,
            output_path=args.output_path,
            window_sizes=window_sizes,
            oos_start_date=args.oos_start_date,
            force=args.force,
            verbose=args.verbose,
            oos_results_dir=args.runs_dir
        )
        
        if results:
            print(f"Economic value analysis complete. Results saved to {args.output_path}")
            return 0
        else:
            print("Error: Economic value analysis failed")
            return 1
    
    # Handle original run command (existing functionality)
    elif args.command == "run":
        # Handle resource info request first
        if args.resource_info:
            from src.utils.resource_manager import ResourceManager
            rm = ResourceManager(verbose=True)
            print("\nRecommended settings for your system:")
            print(f"  --hpo-jobs {rm.get_safe_worker_count('hpo')}")
            print(f"  --parallel-trials (for Bayesian/Random experiments)")
            if rm.system_type in ["WORKSTATION", "HPC_SERVER"]:
                print(f"  --parallel-models (for additional speedup)")
            print(f"\nExample command:")
            print(f"  python -m src.cli run --method bayes_oos --models Net1 Net2 --parallel-trials --hpo-jobs {rm.get_safe_worker_count('hpo')}")
            return 0
        
        models_to_run = args.models
        if models_to_run is None:
            if args.method == "bayes": models_to_run = DEFAULT_MODELS_IN_SAMPLE_BAYES
            elif args.method == "grid": models_to_run = DEFAULT_MODELS_IN_SAMPLE_GRID
            elif args.method == "random": models_to_run = DEFAULT_MODELS_IN_SAMPLE_RANDOM
            elif args.method == "bayes_oos": models_to_run = DEFAULT_MODELS_OOS_BAYES
            elif args.method == "grid_oos": models_to_run = DEFAULT_MODELS_OOS_GRID
            elif args.method == "random_oos": models_to_run = DEFAULT_MODELS_OOS_RANDOM
            else: models_to_run = DEFAULT_MODELS_ALL 
            print(f"No models specified, using default for {args.method}: {models_to_run}")

        if not models_to_run:
            print(f"Error: No models defined for method '{args.method}' and --models not specified.", file=sys.stderr)
            return 1

        # Original run command logic continues
        if args.device == "auto":
            selected_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            selected_device = args.device
        print(f"--- Primary device: {selected_device} ---")
        
        if selected_device == "cuda":
            try:
                # Test if CUDA tensors are actually available
                test_tensor = torch.tensor([1.0], device='cuda')
                torch.set_default_device('cuda')
                torch.set_default_dtype(torch.float32)
                print("--- Default device set to CUDA ---")
            except (RuntimeError, TypeError) as e:
                print(f"Warning: CUDA reported as available but tensors not accessible: {e}")
                print("--- Falling back to CPU ---")
                selected_device = "cpu"

        experiment_module_name = args.method
        
        # Map method names to the new file naming convention with suffixes
        method_to_module = {
            "bayes": "bayes_is_0", 
            "grid": "grid_is_0", 
            "random": "random_is_0",
            "bayes_oos": "bayes_oos_1", 
            "grid_oos": "grid_oos_1", 
            "random_oos": "random_oos_1",
            "rolling_grid": "rolling_window_3",
            "rolling_random": "rolling_window_3",
            "rolling_bayes": "rolling_window_3",
            "expanding_grid": "expanding_window_4",
            "expanding_random": "expanding_window_4",
            "expanding_bayes": "expanding_window_4",
            "grid_mae": "grid_mae_is_5",
            "random_mae": "random_mae_is_5",
            "bayes_mae": "bayes_mae_is_5",
            "grid_mae_oos": "grid_mae_oos_5",
            "random_mae_oos": "random_mae_oos_5",
            "bayes_mae_oos": "bayes_mae_oos_5"
        }
        
        # Get module name with suffix
        if experiment_module_name in method_to_module:
            module_with_suffix = method_to_module[experiment_module_name]
        else:
            module_with_suffix = experiment_module_name
            
        mod_path = f"src.experiments.{module_with_suffix}"
        try:
            experiment_module = importlib.import_module(mod_path)
        except ModuleNotFoundError:
            print(f"Error: Experiment module '{mod_path}' not found.", file=sys.stderr)
            print("Ensure you have files like 'bayes_oos.py', 'grid_oos.py', etc. in src/experiments/", file=sys.stderr)
            print(f"Current sys.path: {sys.path}", file=sys.stderr) 
            return 1

        # Configure parallelization settings
        from src.utils.resource_manager import ResourceManager
        rm = ResourceManager(verbose=False)
        
        # Determine HPO jobs setting
        if args.parallel_trials and args.hpo_jobs is None:
            # Auto-set safe number of jobs
            hpo_jobs = rm.get_safe_worker_count("hpo")
            print(f"Auto-setting hpo-jobs to {hpo_jobs} based on system resources")
        elif args.hpo_jobs is not None:
            # Use user-specified value (validated by ResourceManager)
            hpo_jobs = rm.get_safe_worker_count("hpo", requested_workers=args.hpo_jobs)
        else:
            # Default: single-threaded for backward compatibility
            hpo_jobs = 1
        
        # Different experiment types need different parameters
        if experiment_module_name.endswith('_oos'):
            # OOS experiments take differently formatted parameters
            hpo_general_config = {
                'hpo_epochs': args.epochs,
                'hpo_trials': args.trials,  # For random/bayes (ignored by grid)
                'hpo_device': selected_device,
                'hpo_batch_size': args.batch,
                'hpo_jobs': hpo_jobs  # Add parallelization parameter
            }
            run_kwargs = {
                'model_names': models_to_run,
                'oos_start_date_int': args.oos_start_date,
                'hpo_general_config': hpo_general_config,
                'save_annual_models': False,  # Default to false for CLI
                'parallel_models': args.parallel_models,  # Add model-level parallelism
                'verbose': args.verbose  # Pass verbose flag
            }
        elif experiment_module_name.startswith('random') or experiment_module_name.startswith('bayes'):
            # In-sample trial-based methods
            run_kwargs = {
                'models': models_to_run,
                'trials': args.trials,
                'epochs': args.epochs,
                'batch': args.batch,
                'threads': args.threads,
                'device': selected_device,
                'gamma': args.gamma,
                'verbose': args.verbose  # Pass verbose flag
            }
        elif experiment_module_name.startswith('grid'):
            # In-sample grid search doesn't use trials
            run_kwargs = {
                'models': models_to_run,
                'trials': args.trials,  # Keep for signature compatibility
                'epochs': args.epochs,
                'threads': args.threads,
                'batch': args.batch,
                'device': selected_device,
                'gamma_cer': args.gamma,
                'verbose': args.verbose  # Pass verbose flag
            }
        elif experiment_module_name.startswith('rolling_'):
            # Extract optimization method from experiment_module_name
            opt_method = experiment_module_name.split('_')[1]  # grid, random, or bayes
            
            # Parse window sizes
            window_sizes = [int(size) for size in args.window_sizes.split(',')]
            
            # Configure HPO settings
            hpo_general_config = {
                'hpo_epochs': args.epochs,
                'hpo_trials': args.trials,  # For random/bayes (ignored by grid)
                'hpo_device': selected_device,
                'hpo_batch_size': args.batch,
                'hpo_jobs': hpo_jobs  # Add parallelization parameter
            }
            
            # Configure parallelization for window experiments
            enable_model_parallel = args.parallel_models or args.nested_parallelism
            enable_window_parallel = args.parallel_windows or args.nested_parallelism
            
            # Set up run kwargs for rolling window analysis
            run_kwargs = {
                'model_names': models_to_run,
                'window_sizes': window_sizes,
                'oos_start_date_int': args.oos_start_date,
                'optimization_method': opt_method,
                'hpo_general_config': hpo_general_config,
                'parallel_models': enable_model_parallel,  # Add model-level parallelism
                'parallel_windows': enable_window_parallel,  # Add window-level parallelism
                'verbose': args.verbose  # Pass verbose flag
            }
        elif experiment_module_name.startswith('expanding_'):
            # Extract optimization method from experiment_module_name
            opt_method = experiment_module_name.split('_')[1]  # grid, random, or bayes
            
            # Parse window sizes
            window_sizes = [int(size) for size in args.window_sizes.split(',')]
            
            # Configure HPO settings
            hpo_general_config = {
                'hpo_epochs': args.epochs,
                'hpo_trials': args.trials,  # For random/bayes (ignored by grid)
                'hpo_device': selected_device,
                'hpo_batch_size': args.batch,
                'hpo_jobs': hpo_jobs  # Add parallelization parameter
            }
            
            # Configure parallelization for window experiments  
            enable_model_parallel = args.parallel_models or args.nested_parallelism
            enable_window_parallel = args.parallel_windows or args.nested_parallelism
            
            # Set up run kwargs for expanding window analysis
            run_kwargs = {
                'model_names': models_to_run,
                'window_sizes': window_sizes,
                'oos_start_date_int': args.oos_start_date,
                'optimization_method': opt_method,
                'hpo_general_config': hpo_general_config,
                'parallel_models': enable_model_parallel,  # Add model-level parallelism
                'parallel_windows': enable_window_parallel,  # Add window-level parallelism
                'verbose': args.verbose  # Pass verbose flag
            }
        elif experiment_module_name == 'newly_identified':
            # Setup for newly identified variables
            run_kwargs = {
                'models': models_to_run,
                'method': args.method.split('_')[-1] if '_' in args.method else 'grid',
                'integration_mode': args.integration_mode,
                'trials': args.trials,
                'epochs': args.epochs,
                'batch': args.batch,
                'threads': args.threads,
                'device': selected_device,
                'gamma_cer': args.gamma,
                'verbose': args.verbose  # Pass verbose flag
            }
        elif experiment_module_name in ['grid_oos_6', 'random_oos_6', 'bayes_oos_6']:
            # Setup for out-of-sample evaluation with newly identified variables
            run_kwargs = {
                'models': models_to_run,
                'trials': args.trials,  # Add trials parameter for flexibility
                'epochs': args.epochs,   # Add epochs parameter for flexibility
                'oos_start_date': args.oos_start_date,  # Add OOS start date parameter
                'integration_mode': args.integration_mode,
                'threads': args.threads,
                'batch': args.batch,     # Add batch size parameter
                'device': selected_device,
                'gamma_cer': args.gamma,  # This parameter needs to match the function signature
                'verbose': args.verbose  # Pass verbose flag
            }
            
            # Add OOS end date if specified
            if hasattr(args, 'oos_end_date') and args.oos_end_date is not None:
                run_kwargs['oos_end_date'] = args.oos_end_date
        elif experiment_module_name == 'fred_variables':
            # Setup for FRED variables
            # Check if optimization_method is explicitly specified
            if hasattr(args, 'optimization_method') and args.optimization_method:
                method_param = args.optimization_method
            else:
                # Default to 'grid' if not specified
                method_param = 'grid'
            
            run_kwargs = {
                'models': models_to_run,
                'method': method_param,
                'trials': args.trials,
                'epochs': args.epochs,
                'batch': args.batch,
                'threads': args.threads,
                'device': selected_device,
                'gamma_cer': args.gamma
            }
        elif experiment_module_name in ['grid_oos_7', 'random_oos_7', 'bayes_oos_7']:
            # Setup for out-of-sample evaluation with FRED variables
            run_kwargs = {
                'models': models_to_run,
                'threads': args.threads,
                'device': selected_device,
                'gamma_cer': args.gamma
            }
        elif experiment_module_name == 'variable_importance_8':
            # Setup for variable importance analysis
            run_kwargs = {
                'models': models_to_run,
                'start_date': args.oos_start_date,
                'n_jobs': args.n_jobs,
                'threads': args.threads,
                'device': selected_device,
                'data_source': args.data_source
            }
        elif experiment_module_name == 'gamma_sensitivity_9':
            # Setup for gamma sensitivity analysis
            # For methods, use the base method if specified, otherwise default to bayes_oos
            base_method = args.method.split('_')[0] if '_' in args.method else args.method
            if base_method in ['grid', 'random', 'bayes']:
                methods = [f"{base_method}_oos"] 
            else:
                methods = ['bayes_oos']
                
            run_kwargs = {
                'models': models_to_run,
                'methods': methods,
                'gamma_range': args.gamma_range,
                'rf_rate': args.rf_rate_default,
                'run_dir': args.run_dir,
                'start_date': args.oos_start_date,
                'data_source': args.data_source,
                'n_jobs': args.n_jobs
            }
        elif experiment_module_name == 'profit_optimization_10':
            # Setup for profit optimization
            base_method = args.method.split('_')[-1] if '_' in args.method else args.method
            if base_method in ['grid', 'random', 'bayes']:
                method_param = base_method
            else:
                method_param = 'grid'
                
            run_kwargs = {
                'models': models_to_run,
                'method': method_param,
                'trials': args.trials,
                'epochs': args.epochs,
                'rf_rate_default': args.rf_rate_default,
                'max_leverage': args.max_leverage,
                'transaction_cost': args.transaction_cost,
                'position_sizing': args.position_sizing,
                'batch': args.batch,
                'threads': args.threads,
                'device': selected_device,
                'data_source': args.data_source
            }
        elif experiment_module_name == 'profit_oos_10':
            # Setup for OOS evaluation of profit-optimized models
            base_method = args.method.split('_')[-1] if '_' in args.method else args.method
            if base_method in ['grid', 'random', 'bayes']:
                method_param = base_method
            else:
                method_param = 'grid'
                
            run_kwargs = {
                'models': models_to_run,
                'method': method_param,
                'rf_rate_default': args.rf_rate_default,
                'max_leverage': args.max_leverage,
                'transaction_cost': args.transaction_cost,
                'position_sizing': args.position_sizing,
                'oos_start_date': args.oos_start_date,
                'threads': args.threads,
                'device': selected_device,
                'data_source': args.data_source
            }
        else:
            # Other experiment types
            run_kwargs = {
                # Set appropriate parameters based on experiment type
            }
    
        print(f"--- Launching Experiment: {experiment_module_name} ---")
        print(f"   Models: {models_to_run}")
        if experiment_module_name in ["bayes_oos", "random_oos"]:
            print(f"   OOS Start Date: {args.oos_start_date}")
            print(f"   Epochs per HPO/Retrain: {args.epochs}")
            print(f"   Trials per HPO: {args.trials}")
            print(f"   Batch Size for HPO: {args.batch}")
        else:
            print(f"   Epochs: {args.epochs}")
            print(f"   Batch Size: {args.batch}")
            if experiment_module_name in ["bayes", "random"]:
                 print(f"   Trials: {args.trials}")
            if experiment_module_name == "bayes":
                print(f"   Gamma (CER): {args.gamma}")

        try:
            experiment_module.run(**run_kwargs)
            print(f"--- Experiment {experiment_module_name} completed successfully. ---")
        except Exception as e:
            print(f"Error during experiment {experiment_module_name}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return 1
    
    else:
        # Unknown command
        print(f"Error: Unknown command '{args.command}'")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())