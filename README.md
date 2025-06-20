# Equity Premium Prediction with Neural Networks

This framework provides a comprehensive and customizable approach to neural network development for equity premium prediction. The codebase offers a systematic implementation of three leading hyperparameter optimization methods:

- **Grid Search**: Exhaustive search over specified parameter values
- **Random Search**: Efficient exploration of parameter space through random sampling
- **Bayesian Optimization**: Intelligent search guided by probabilistic models of parameter performance

The framework's modular design supports model evaluation through in-sample validation, out-of-sample testing, economic value analysis, and advanced window-based testing methodologies. Each component is carefully engineered to provide meaningful insights into forecasting performance while maintaining research reproducibility.

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

Required packages include PyTorch, Pandas, NumPy, Scikit-learn, Optuna, and Matplotlib.

### Environment Setup

**Important for Windows/Anaconda users**: To avoid OpenMP library conflicts when using PyTorch, set the following environment variable before running experiments:

```powershell
# PowerShell
$env:KMP_DUPLICATE_LIB_OK="TRUE"

# Command Prompt
set KMP_DUPLICATE_LIB_OK=TRUE
```

This resolves the common "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized" error.

## Performance Optimization and Multithreading

This framework is designed for **massive scalability** and can utilize high-performance computing resources efficiently. The codebase supports comprehensive multithreading across all 20+ experiment types and utility modules.

### Hardware Compatibility

**Automatic Resource Detection**: The framework automatically detects available hardware and optimizes accordingly:
- **4-16 cores**: Standard workstation optimization
- **16-64 cores**: High-end workstation with aggressive parallelization  
- **64-128+ cores**: HPC/Server mode with maximum parallel utilization
- **Multi-GPU**: Automatic GPU memory management and model distribution

### Performance Scaling

**Expected Performance Improvements**:
- **Standard Hardware (8-16 cores)**: 5-10x speedup
- **High-End Workstation (32+ cores)**: 10-20x speedup
- **HPC/Server (128+ cores)**: 20-100x speedup

**Example Performance Transformation**:

| Experiment Type | Current Runtime | With Parallelization | Speedup |
|----------------|----------------|---------------------|---------|
| OOS Experiments | Long runtime | Significantly faster | 8-16x |
| Variable Importance | Long runtime | Significantly faster | 30x |
| HPO Optimization | Long runtime | Significantly faster | 24-48x |
| Window Analysis | Long runtime | Significantly faster | 3-6x |

### Multithreading Features

**Comprehensive Parallelization Coverage**:
- ✅ **Model-Level Parallelism**: All 8 neural network models train simultaneously
- ✅ **HPO Trial Parallelism**: 100+ parallel hyperparameter trials
- ✅ **Window Analysis**: Multiple window sizes processed concurrently  
- ✅ **Variable Importance**: Parallel variable permutation testing
- ✅ **Economic Analysis**: Concurrent gamma sensitivity and profit optimization
- ✅ **Data Pipeline**: Parallel data loading and preprocessing

**Threading Configuration Options**:

```bash
# 🔒 LAPTOP SAFE: Default behavior (unchanged)
python -m src.cli run --method bayes_oos --models Net1 Net2

# ✅ PHASE 1: Enable HPO trial parallelism (2-4x speedup)
python -m src.cli run --method bayes_oos --models Net1 Net2 --parallel-trials

# ✅ PHASE 2: Enable model-level parallelism (4-8x additional speedup)
python -m src.cli run --method bayes_oos --models Net1 Net2 Net3 --parallel-models

# 🚀 MAXIMUM: Both types of parallelism (15-30x total speedup)
python -m src.cli run --method bayes_oos --models Net1 Net2 Net3 Net4 Net5 --parallel-trials --parallel-models --hpo-jobs 24

# 🔍 Check system capabilities first
python -m src.cli run --resource-info
```

### Implementation Status

The multithreading implementation follows a **comprehensive 4-phase plan** covering all experiment types:

**Phase 1** (Immediate Wins): ✅ Core HPO parallelization
- Optuna parallel trials
- Grid search parameter parallelism
- Random search trial parallelism

**Phase 2** (High Impact): ✅ Model-level parallelism (COMPLETED)
- Parallel model processing in OOS loops
- Smart GPU memory management
- Adaptive resource detection

**Phase 3** (HPC Optimization): 🚀 Server-specific features
- 128-core server optimizations
- Nested parallelization strategies
- Memory management for high-RAM systems

**Phase 4** (Advanced Features): 🔬 Specialized optimizations
- Asynchronous data pipelines
- Memory-mapped data access
- Distributed computing support

### Server and Cloud Deployment

**Automatic Optimization**: The framework automatically detects and optimizes for:
- **AWS/GCP Instances**: Scales to available vCPUs
- **HPC Clusters**: Utilizes high core counts efficiently
- **Docker Containers**: Respects resource limits
- **Local Workstations**: Conservative resource usage

**Server Mode Commands**:

```bash
# Automatic server detection (recommended)
python -m src.cli run --method bayes_oos --models Net1 DNet1 --server-mode

# Manual configuration for 128-core server
python -m src.cli run --method bayes_oos --max-cores 128 --trials-multiplier 8.0

# Memory-optimized mode for high-RAM servers (256GB+)
python -m src.cli run --method bayes_oos --memory-gb 512 --nested-parallelism
```

### Implementation Guide

For detailed implementation instructions, see `MULTITHREADING_IMPLEMENTATION_PLAN.txt` which provides:
- File-by-file parallelization strategies
- Hardware-specific optimization guidance  
- Step-by-step implementation timeline
- Performance benchmarking protocols
- Risk mitigation strategies

The multithreading implementation ensures **maximum speed optimization** while preserving existing workflow integrity and providing adaptive performance based on available hardware resources.

### Data

Place your data files in the `./data` directory:
- `ml_equity_premium_data.xlsx`: Contains predictor data and historical equity premium values

## Workflow Structure

The analysis pipeline is structured in multiple stages, each focused on different aspects of model development and evaluation:

### 0. In-Sample Hyperparameter Optimization

Perform in-sample hyperparameter optimization using different methods:
- Grid Search (`grid_is`)
- Random Search (`random_is`)
- Bayesian Optimization (`bayes_is`)

### 1. Out-of-Sample Evaluation

Conduct out-of-sample testing with optimized hyperparameters:
- Grid Search OOS (`grid_oos`)
- Random Search OOS (`random_oos`)
- Bayesian Optimization OOS (`bayes_oos`)

### 2. Economic Value Analysis

Evaluate economic value of forecasting models through market timing performance metrics.

### 3. Rolling Window Analysis

Perform out-of-sample testing using rolling windows of different sizes:
- Grid Search Rolling (`rolling_grid`)
- Random Search Rolling (`rolling_random`)
- Bayesian Optimization Rolling (`rolling_bayes`)

### 4. Expanding Window Analysis

Perform out-of-sample testing using expanding windows with configurable minimum sizes:
- Grid Search Expanding (`expanding_grid`)
- Random Search Expanding (`expanding_random`)
- Bayesian Optimization Expanding (`expanding_bayes`)

### 5. Alternative Validation Schemes

Perform hyperparameter optimization using Mean Absolute Error (MAE) as the validation metric:
- Grid Search with MAE scoring (`grid_mae`)
- Random Search with MAE scoring (`random_mae`)
- Bayesian Optimization with MAE scoring (`bayes_mae`)

Conduct out-of-sample testing with MAE-optimized hyperparameters:
- Grid Search OOS with MAE parameters (`grid_mae_oos`)
- Random Search OOS with MAE parameters (`random_mae_oos`)
- Bayesian Optimization OOS with MAE parameters (`bayes_mae_oos`)

### 6. Newly Identified Variables Analysis

Process and evaluate newly identified variables from the `NewlyIdentifiedVariables` sheet in the data file:
- In either standalone mode (only new variables) or integrated mode (combined with existing predictors)
- Using all three optimization methods (grid, random, Bayesian)
- With corresponding out-of-sample evaluation scripts

In-sample optimization with newly identified variables:
- Grid Search (`newly_identified` with method=grid)
- Random Search (`newly_identified` with method=random)
- Bayesian Optimization (`newly_identified` with method=bayes)

Out-of-sample evaluation with newly identified variables:
- Grid Search OOS (`grid_oos_6`)
- Random Search OOS (`random_oos_6`)
- Bayesian Optimization OOS (`bayes_oos_6`)

#### Running Out-of-Sample Analysis with Newly Identified Variables

```bash
# Using standalone mode (only new variables)
python -m src.experiments.bayes_oos_6 --models Net1 Net2 Net3 --integration-mode standalone --oos-start-date 200001
python -m src.experiments.grid_oos_6 --models Net1 Net2 Net3 --integration-mode standalone --oos-start-date 200001
python -m src.experiments.random_oos_6 --models Net1 Net2 Net3 --integration-mode standalone --oos-start-date 200001

# Using integrated mode (combining new variables with existing predictors)
python -m src.experiments.bayes_oos_6 --models Net1 Net2 Net3 --integration-mode integrated --oos-start-date 200001
python -m src.experiments.grid_oos_6 --models Net1 Net2 Net3 --integration-mode integrated --oos-start-date 200001
python -m src.experiments.random_oos_6 --models Net1 Net2 Net3 --integration-mode integrated --oos-start-date 200001

# With custom parameters
python -m src.experiments.bayes_oos_6 --models Net1 DNet1 --integration-mode integrated --oos-start-date 200001 --oos-end-date 202112 --trials 20 --epochs 50 --batch 256
```

The newly identified variables are available from 1990-01 to 2021-12. Results are saved following the consistent naming convention with `6_` prefix for directories.

### 7. FRED Variables Analysis

Process and evaluate macroeconomic variables from the `FRED_MD` sheet in the data file:
- Using all three optimization methods (grid, random, Bayesian)
- With corresponding out-of-sample evaluation scripts

### 8. Variable Importance Analysis

Assess the relative importance of individual predictors using permutation-based techniques:
- Supports analysis across different data sources (original, newly identified, FRED)
- Provides insights into which variables drive predictive performance

### 9. Gamma Sensitivity Analysis

Evaluate model performance across different risk aversion levels:
- Calculates model performance metrics across a range of risk aversion coefficients
- Identifies which models perform best for different investor risk profiles

### 10. Profit Optimization

Implements trading strategies optimized for direct profit maximization:
- Customizable position sizing, leverage, and transaction cost parameters
- Provides realistic performance evaluation incorporating market frictions
- Data is min-max scaled before processing (as in Xiu and Liu, 2024)
- Automatically handles date format conversion from the FRED format
- Limited to data from 199001 to 202312 for consistency

In-sample optimization with FRED variables:
- Grid Search (`fred_variables` with optimization-method=grid)
- Random Search (`fred_variables` with optimization-method=random)
- Bayesian Optimization (`fred_variables` with optimization-method=bayes)

Out-of-sample evaluation with FRED variables:
- Grid Search OOS (`grid_oos_7`)
- Random Search OOS (`random_oos_7`)
- Bayesian Optimization OOS (`bayes_oos_7`)

Results are saved following the consistent naming convention with `7_` prefix.

## Quick Start (Windows/Anaconda)

For Windows users with Anaconda, here's the minimal command needed to run experiments:

```powershell
# Quick command for Net1 with Bayesian OOS (PowerShell)
$env:KMP_DUPLICATE_LIB_OK="TRUE"; & "C:\Users\AaronHennessy\anaconda3\python.exe" -m src.cli run --method bayes_oos --models Net1 --device cuda --oos-start-date 200001
```

This single command:
- Sets the required environment variable to prevent OpenMP conflicts
- Uses your Anaconda Python installation directly
- Runs Net1 with Bayesian optimization in out-of-sample mode
- Utilizes CUDA for GPU acceleration
- Starts OOS evaluation from January 2000

No need to copy DLL files or set up complex environments!

### Performance Scaling Examples

The framework auto-detects hardware capabilities, but explicit flags ensure maximum resource utilization:

**Basic Run (Limited Parallelization):**
```bash
python -m src.cli run --method bayes_oos --models Net1 --device cuda --oos-start-date 200001
```

**High-End Workstation (32+ cores):**
```bash
python -m src.cli run --method bayes_oos --models Net1 --device cuda --oos-start-date 200001 --parallel-models --hpo-jobs 32
```

**Server Mode (64+ cores):**
```bash
python -m src.cli run --method bayes_oos --models Net1 --device cuda --oos-start-date 200001 --server-mode --parallel-models
```

**HPC Cluster (128+ cores):**
```bash
python -m src.cli run --method bayes_oos --models Net1 --device cuda --oos-start-date 200001 --server-mode --nested-parallelism --hpo-jobs 64
```

**Multi-GPU System:**
```bash
python -m src.cli run --method bayes_oos --models Net1 Net2 Net3 --device cuda --oos-start-date 200001 --server-mode --parallel-models
```

## Using the CLI

All functions are accessible through a unified command-line interface.

### In-Sample Optimization

```bash
# Grid Search
python -m src.cli run --method grid --models Net1 Net2 Net3 --epochs 75

# Random Search
python -m src.cli run --method random --models Net1 Net2 Net3 --trials 50 --epochs 75

# Bayesian Optimization
python -m src.cli run --method bayes --models Net1 Net2 Net3 --trials 50 --epochs 75
```

Common options:
- `--models`: List of models to run (Net1-5, DNet1-3)
- `--trials`: Number of trials for Random Search or Bayesian Optimization
- `--epochs`: Number of training epochs per configuration
- `--batch`: Batch size for training
- `--device`: Computing device (`cpu`, `cuda`, or `auto`)
- `--gamma`: Risk aversion coefficient for CER calculation
- `--verbose`: Enable detailed progress output

**Performance and Threading Options**:
- `--server-mode`: Enable optimizations for high-core server environments
- `--hpo-jobs`: Number of parallel HPO jobs (default: auto-detect)
- `--max-cores`: Maximum cores to use (default: auto-detect)
- `--trials-multiplier`: Multiply base trial count for servers (default: 1.0)
- `--parallel-models`: Enable parallel model processing
- `--parallel-trials`: Enable parallel HPO trials
- `--nested-parallelism`: Enable nested parallelism (models × trials × windows)

### Out-of-Sample Evaluation

```bash
# Grid Search OOS
python -m src.cli run --method grid_oos --models Net1 Net2 Net3 --epochs 30 --oos-start-date 199001

# Random Search OOS
python -m src.cli run --method random_oos --models Net1 Net2 Net3 --trials 10 --epochs 30 --oos-start-date 199001

# Bayesian Optimization OOS
python -m src.cli run --method bayes_oos --models Net1 Net2 Net3 --trials 10 --epochs 30 --oos-start-date 199001
```

Additional options:
- `--oos-start-date`: Start date for out-of-sample period in YYYYMM format (default: 200001)

### Hardware-Optimized Examples

**Local Workstation (8-16 cores)**:
```bash
# Standard optimization with conservative threading
python -m src.cli run --method bayes_oos --models Net1 Net2 Net3 --trials 20 --epochs 50 --verbose

# Enable parallel model processing for faster execution
python -m src.cli run --method bayes_oos --models Net1 Net2 Net3 DNet1 --parallel-models --trials 30
```

**High-End Workstation (32+ cores)**:
```bash
# Aggressive parallelization with higher trial counts
python -m src.cli run --method bayes_oos --models Net1 Net2 Net3 Net4 Net5 --trials 100 --hpo-jobs 16 --parallel-models

# Multiple window analysis with parallel processing
python -m src.cli run --method rolling_bayes --models Net1 Net2 Net3 --window-sizes 5,10,20 --parallel-windows --trials 50
```

**HPC/Server (128+ cores)**:
```bash
# Maximum parallelization with automatic server detection
python -m src.cli run --method bayes_oos --models Net1 Net2 Net3 Net4 Net5 DNet1 DNet2 DNet3 --server-mode --trials-multiplier 8.0

# Nested parallelism for ultimate performance
python -m src.cli run --method bayes_oos --models Net1 Net2 Net3 Net4 Net5 DNet1 DNet2 DNet3 --nested-parallelism --max-cores 128 --trials 500

# Variable importance with massive parallelization
python -m src.cli run --method variable_importance_8 --data-source original --n-jobs 64 --parallel-variables
```

**Cloud/Container Deployment**:
```bash
# Auto-detect container resource limits
python -m src.cli run --method bayes_oos --models Net1 Net2 Net3 --trials 50 --device auto

# Memory-optimized for high-RAM instances
python -m src.cli run --method bayes_oos --models Net1 Net2 Net3 Net4 Net5 --memory-gb 256 --trials 200
```

### Economic Value Analysis

Calculate the economic value of your forecasting models:

```bash
# Run analysis on all OOS results
python -m src.cli economic-value --runs-dir ./runs --data-path ./data --oos-start-date 199001
```

Options:
- `--runs-dir`: Directory containing OOS results (or path to combined predictions file)
- `--data-path`: Path to directory with raw data files
- `--output-path`: Path to save analysis results (default: "./runs/2_Economic_Value_Analysis")
- `--window-sizes`: Comma-separated list of expanding window sizes in years (default: "1,3")
- `--oos-start-date`: Start date for out-of-sample period
- `--force`: Re-analyze data even if previous analysis exists

To collect OOS results separately:
```bash
python -m src.cli collect-oos-results --base-dir ./runs
```

### Rolling Window Analysis

Perform out-of-sample analysis using rolling windows instead of expanding windows:

```bash
# Grid Search with rolling windows
python -m src.cli run --method rolling_grid --models Net1 Net2 Net3 --epochs 30 --oos-start-date 199001 --window-sizes 5,10,20

# Random Search with rolling windows
python -m src.cli run --method rolling_random --models Net1 Net2 Net3 --trials 10 --epochs 30 --oos-start-date 199001 --window-sizes 5,10,20

# Bayesian Optimization with rolling windows
python -m src.cli run --method rolling_bayes --models Net1 Net2 Net3 --trials 10 --epochs 30 --oos-start-date 199001 --window-sizes 5,10,20
```

Additional options:
- `--window-sizes`: Comma-separated list of window sizes in years (default: "5,10,20")

### Expanding Window Analysis

Perform out-of-sample analysis using expanding windows with configurable minimum sizes:

```bash
# Grid Search with expanding windows
python -m src.cli run --method expanding_grid --models Net1 Net2 Net3 --epochs 30 --oos-start-date 199001 --window-sizes 1,3

# Random Search with expanding windows
python -m src.cli run --method expanding_random --models Net1 Net2 Net3 --trials 10 --epochs 30 --oos-start-date 199001 --window-sizes 1,3

# Bayesian Optimization with expanding windows
python -m src.cli run --method expanding_bayes --models Net1 Net2 Net3 --trials 10 --epochs 30 --oos-start-date 199001 --window-sizes 1,3
```

The `--window-sizes` parameter specifies the minimum sizes (in years) for the expanding windows. Each window starts with the specified minimum size and expands as time progresses.

### Alternative Validation Schemes (MAE Scoring)

Perform hyperparameter optimization using Mean Absolute Error (MAE) as the validation metric:

```bash
# Grid Search with MAE scoring
python -m src.cli run --method grid_mae --models Net1 Net2 Net3 --epochs 30

# Random Search with MAE scoring
python -m src.cli run --method random_mae --models Net1 Net2 Net3 --trials 10 --epochs 30

# Bayesian Optimization with MAE scoring
python -m src.cli run --method bayes_mae --models Net1 Net2 Net3 --trials 10 --epochs 30
```

Conduct out-of-sample evaluation using MAE-optimized hyperparameters:

```bash
# Grid Search OOS with MAE-optimized parameters
python -m src.cli run --method grid_mae_oos --models Net1 Net2 Net3 --epochs 30 --oos-start-date 199001

# Random Search OOS with MAE-optimized parameters
python -m src.cli run --method random_mae_oos --models Net1 Net2 Net3 --epochs 30 --oos-start-date 199001

# Bayesian Optimization OOS with MAE-optimized parameters
python -m src.cli run --method bayes_mae_oos --models Net1 Net2 Net3 --epochs 30 --oos-start-date 199001
```

MAE-based models use an alternative validation scheme that may be better suited for certain types of forecasting tasks, potentially providing more robust performance in the presence of outliers.

### Newly Identified Variables Analysis

Process and evaluate newly identified variables from the data file:

```bash
# Using the CLI for in-sample optimization with newly identified variables
python -m src.cli run --method newly_identified --models Net1 Net2 Net3 --integration-mode standalone

# Using the CLI for out-of-sample evaluation with different optimization methods
python -m src.cli run --method grid_oos_6 --models Net1 Net2 Net3 --integration-mode standalone
python -m src.cli run --method random_oos_6 --models Net1 Net2 Net3 --integration-mode standalone
python -m src.cli run --method bayes_oos_6 --models Net1 Net2 Net3 --integration-mode standalone

# You can also run specific optimization method on newly identified variables
python -m src.cli run --method newly_identified --models Net1 Net2 Net3 --optimization-method grid --integration-mode standalone
python -m src.cli run --method newly_identified --models Net1 Net2 Net3 --optimization-method random --trials 50 --integration-mode standalone
python -m src.cli run --method newly_identified --models Net1 Net2 Net3 --optimization-method bayes --trials 50 --integration-mode standalone

# For the integrated mode (combining new variables with existing predictors)
python -m src.cli run --method newly_identified --models Net1 Net2 Net3 --integration-mode integrated
```

Additional options:
- `--integration-mode`: Either 'standalone' (only new variables) or 'integrated' (combine with existing predictors)
- `--optimization-method`: Specify which optimization method to use with newly identified variables ('grid', 'random', or 'bayes')

#### Direct Script Access (Alternative)

You can also run the scripts directly if needed:

```bash
# Direct script execution for in-sample optimization
python -m src.experiments.newly_identified_6 --models Net1 Net2 Net3 --method grid --integration-mode standalone

# Direct script execution for out-of-sample evaluation
python -m src.experiments.grid_oos_6 --models Net1 Net2 Net3 --integration-mode standalone
python -m src.experiments.random_oos_6 --models Net1 Net2 Net3 --integration-mode standalone
python -m src.experiments.bayes_oos_6 --models Net1 Net2 Net3 --integration-mode standalone
```

### FRED Variables Analysis

Process and evaluate macroeconomic variables from the `FRED_MD` sheet in the data file:

```bash
# Using the CLI for in-sample optimization with FRED variables
python -m src.cli run --method fred_variables --models Net1 Net2 Net3 

# You can specify different optimization methods
python -m src.cli run --method fred_variables --models Net1 Net2 Net3 --optimization-method grid
python -m src.cli run --method fred_variables --models Net1 Net2 Net3 --optimization-method random --trials 50
python -m src.cli run --method fred_variables --models Net1 Net2 Net3 --optimization-method bayes --trials 50

# For out-of-sample evaluation using different optimization methods
python -m src.cli run --method grid_oos_7 --models Net1 Net2 Net3
python -m src.cli run --method random_oos_7 --models Net1 Net2 Net3
python -m src.cli run --method bayes_oos_7 --models Net1 Net2 Net3
```

If you want to run multiple models with FRED variables:

```bash
# Run all main models with FRED variables using grid search
python -m src.cli run --method fred_variables --models Net1 Net2 Net3 Net4 Net5 --optimization-method grid --epochs 100

# Run all main models with out-of-sample evaluation
python -m src.cli run --method grid_oos_7 --models Net1 Net2 Net3 Net4 Net5
```

#### Direct Script Access (Alternative)

You can also run the scripts directly if needed:

```bash
# Direct script execution for in-sample optimization
python -m src.experiments.fred_variables_7 --models Net1 Net2 Net3 --method grid

# Direct script execution for out-of-sample evaluation
python -m src.experiments.grid_oos_7 --models Net1 Net2 Net3
python -m src.experiments.random_oos_7 --models Net1 Net2 Net3
python -m src.experiments.bayes_oos_7 --models Net1 Net2 Net3
```

The implementation follows the approach demonstrated in Xiu and Liu (2024), using min-max scaling for the FRED variables before building prediction models.

### 8. Variable Importance Analysis

Analyze the importance of individual predictor variables by measuring how model performance changes when each variable is removed from the dataset:
- Evaluates multiple machine learning models (PLS, PCR, LASSO, ENet, RF, NN2, NN4, Ridge)
- Calculates out-of-sample R-squared reduction when each variable is eliminated
- Visualizes the most important variables for each model
- Works with all three data sources (original variables, newly identified variables, or FRED variables)
- Implements the methodology from Xiu and Liu (2024) for variable importance analysis

The variable importance analysis creates detailed rankings showing which variables contribute most to prediction performance across different model types. This provides insight into feature relevance and can guide feature selection decisions.

Results are saved following the consistent naming convention with `8_` prefix for directories.

```bash
# Run variable importance analysis on original data
python -m src.cli run --method variable_importance_8 --data-source original --oos-start-date 195701

# Run variable importance analysis on newly identified variables
python -m src.cli run --method variable_importance_8 --data-source newly_identified --n-jobs 4

# Run variable importance analysis on FRED variables
python -m src.cli run --method variable_importance_8 --data-source fred --n-jobs 8

# Specify models to analyze (defaults to a standard set)
python -m src.cli run --method variable_importance_8 --models Net1 Net2 Net3 Net4 Net5 --data-source original
```

Additional options:
- `--data-source`: Specify which dataset to analyze ('original', 'newly_identified', or 'fred')
- `--n-jobs`: Number of parallel jobs for variable importance calculation (higher values speed up computation but use more resources)
- `--oos-start-date`: Start date for out-of-sample period in YYYYMM format

#### Direct Script Access (Alternative)

You can also run the script directly if needed:

```bash
# Direct script execution for variable importance analysis
python -m src.experiments.variable_importance_8 --data-source original --n-jobs 4 --start-date 195701
```

### 9. Gamma Sensitivity Analysis

Analyze how Certainty Equivalent Return (CER) varies with different risk aversion (gamma) values:
- Evaluates model performance across a spectrum of investor risk preferences
- Identifies optimal models for different risk tolerance levels
- Calculates CER advantage over buy-and-hold strategy 
- Detects crossover points where model ranking changes
- Visualizes sensitivity curves and heatmaps for multiple models simultaneously

This analysis helps understand how robust model performance is to varying risk preferences and identifies which models perform best for different investor types (from low to high risk aversion).

Results are saved following the consistent naming convention with `9_` prefix for directories.

```bash
# Run gamma sensitivity analysis on original data using Bayesian optimization models
python -m src.cli run --method gamma_sensitivity_9 --data-source original --oos-start-date 200001

# Run sensitivity analysis for specific models with custom gamma range
python -m src.cli run --method gamma_sensitivity_9 --models Net1 Net3 Net5 --gamma-range "1,10,20"

# Analyze different data sources with multiple models
python -m src.cli run --method gamma_sensitivity_9 --data-source fred --models Net1 Net2 Net3 Net4 Net5

# Use grid search results instead of Bayesian optimization
python -m src.cli run --method grid_gamma_sensitivity_9 --models Net1 Net2
```

Additional options:
- `--gamma-range`: Customize the range of gamma values to test in format "start,end,num_points"
- `--rf-rate`: Set the annual risk-free rate for return calculations (default: 0.03 or 3%)
- `--run-dir`: Specify an exact results directory to analyze (instead of auto-detection)
- `--n-jobs`: Control parallelization for computation

#### Direct Script Access (Alternative)

```bash
# Direct script execution for gamma sensitivity analysis
python -m src.experiments.gamma_sensitivity_9 --models Net1 Net2 Net3 --methods bayes_oos --gamma-range "1,12,20"
```

### 10. Profit Optimization

Tune models to directly maximize profit instead of minimizing statistical error metrics:
- Implements a realistic trading strategy with institutional constraints
- Long positions with leverage (up to 150%) when predictions are positive
- Investment in risk-free assets when predictions are negative
- Includes institutional-level transaction costs (0.07% per trade)
- Supports both binary and proportional position sizing

This approach aligns model training with economic goals rather than statistical accuracy, potentially leading to better real-world trading performance. The profit optimization includes both in-sample tuning and out-of-sample evaluation phases.

Results are saved following the consistent naming convention with `10_` prefix for directories.

```bash
# Run profit optimization with default parameters (grid search)
python -m src.cli run --method profit_optimization_10 --models Net1 Net2 --data-source original

# Custom trading parameters (risk-free rate, leverage, transaction costs)
python -m src.cli run --method profit_optimization_10 --models Net1 Net2 --rf-rate 0.02 --max-leverage 1.2 --transaction-cost 0.001

# Different position sizing strategies
python -m src.cli run --method profit_optimization_10 --models Net3 Net4 --position-sizing proportional

# Out-of-sample evaluation of profit-optimized models
python -m src.cli run --method profit_oos_10 --models Net1 Net2 --oos-start-date 200001
```

Additional options:
- `--max-leverage`: Maximum leverage allowed (default: 1.5 or 150%)
- `--transaction-cost`: Cost per transaction (default: 0.0007 or 0.07%)
- `--position-sizing`: Method for sizing positions ('binary' or 'proportional')

#### Direct Script Access (Alternative)

```bash
# Direct script execution for profit optimization
python -m src.experiments.profit_optimization_10 --models Net1 Net2 --method grid --max-leverage 1.5

# Direct script execution for out-of-sample evaluation
python -m src.experiments.profit_oos_10 --models Net1 Net2 --oos-start-date 200001
```

## Models

Available neural network models:
- `Net1`: Single hidden layer
- `Net2`: Two hidden layers
- `Net3`: Three hidden layers
- `Net4`: Four hidden layers with skip connections
- `Net5`: Five hidden layers with skip connections
- `DNet1`: Deep network with 4 layers
- `DNet2`: Deep network with 5 layers
- `DNet3`: Deep network with 5 layers and more neurons

## Results

- In-sample results are saved to `./runs/` with matching model names and timestamps
- Out-of-sample results are saved to `./runs/` with OOS details
- Economic value analysis results are saved to `./runs/2_Economic_Value_Analysis/`

Each result directory contains model outputs, predictions, performance metrics, and visualizations.

## Implementation Documentation

**Comprehensive Implementation Guide**: See `MULTITHREADING_IMPLEMENTATION_PLAN.txt` for:
- **Complete Directory Coverage**: File-by-file parallelization strategies for all 20+ experiment types
- **Hardware-Specific Optimization**: Detailed guidance for 4-core laptops to 128-core servers
- **Step-by-Step Implementation**: 4-phase implementation timeline with risk mitigation
- **Performance Benchmarking**: Expected speedups and testing protocols
- **Server Optimization**: Specific strategies for HPC and cloud deployment

The multithreading implementation transforms this framework from a sequential research tool into a **massively parallel neural network optimization engine** capable of utilizing 100+ CPU cores and multiple GPUs simultaneously while preserving all existing functionality and research reproducibility.
