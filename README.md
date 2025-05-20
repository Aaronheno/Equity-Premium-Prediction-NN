# Equity Premium Prediction with Neural Networks

This repository contains tools for predicting equity premiums using neural network models with a focus on hyperparameter optimization and economic value analysis.

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

Required packages include PyTorch, Pandas, NumPy, Scikit-learn, Optuna, and Matplotlib.

### Data

Place your data files in the `./data` directory:
- `ml_equity_premium_data.xlsx`: Contains predictor data and historical equity premium values

## Workflow Structure

The analysis pipeline is structured in four stages:

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
- `--rf-rate`: Annual risk-free rate (default: 0.03 or 3%)
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

## Testing

The framework includes a comprehensive test suite that validates all components. Tests are organized into stages matching the framework's architecture (stages 0-10).

### Running Tests

All tests can be executed using the `run_tests.py` script in the `tests` directory:

```bash
# Run quick tests for rapid validation
python tests/run_tests.py --quick

# Run thorough tests for comprehensive validation
python tests/run_tests.py --thorough

# Test specific stages (e.g., stages 1, 5, and 10)
python tests/run_tests.py --quick --stage 1 5 10
```

### Test Modes

- **Quick Tests**: Fast tests that validate core functionality without extensive computational overhead. Perfect for quick verification after code changes.
  - Automatically skips resource-intensive stage 0 (in-sample tests)
  - Uses minimal datasets and iteration counts
  - Typically completes in under 2 minutes

- **Thorough Tests**: Comprehensive tests that validate all aspects of the framework, including edge cases and complex scenarios.
  - Tests all components including hyperparameter optimization
  - Uses larger datasets and more iterations
  - May take 5-10 minutes to complete

### Test Structure

The tests are organized by stages matching the framework components:

| Test File | Description |
|-----------|-------------|
| `test_stage0_in_sample.py` | In-sample hyperparameter optimization |
| `test_stage1_oos.py` | Out-of-sample evaluation |
| `test_stage2_economic_value.py` | Economic value analysis |
| `test_stage3_4_window_analysis.py` | Rolling and expanding window analyses |
| `test_stage5_mae.py` | MAE-based validation schemes |
| `test_stage6_7_data_sources.py` | Alternative data sources |
| `test_stage8_variable_importance.py` | Variable importance analysis |
| `test_stage9_gamma_sensitivity.py` | Gamma sensitivity analysis |
| `test_stage10_profit_optimization.py` | Profit optimization methods |

## Models

Available neural network models:
- `Net1`: Single hidden layer
- `Net2`: Two hidden layers
- `Net3`: Three hidden layers
- `Net4`: Three hidden layers with skip connections
- `Net5`: Five hidden layers with skip connections
- `DNet1`: Deep network with 4 layers
- `DNet2`: Deep network with 5 layers
- `DNet3`: Deep network with 5 layers and more neurons

## Results

- In-sample results are saved to `./runs/` with matching model names and timestamps
- Out-of-sample results are saved to `./runs/` with OOS details
- Economic value analysis results are saved to `./runs/2_Economic_Value_Analysis/`

Each result directory contains model outputs, predictions, performance metrics, and visualizations.
