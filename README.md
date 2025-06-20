# Equity Premium Prediction with Neural Networks

A comprehensive neural network framework for equity premium prediction featuring 8 neural network architectures, advanced hyperparameter optimization, and multi-core parallel processing.

## Quick Start

### Requirements
```bash
pip install -r requirements.txt
```

### Environment Setup (Windows/Anaconda)
```powershell
# PowerShell - Prevent OpenMP conflicts
$env:KMP_DUPLICATE_LIB_OK="TRUE"

# Command Prompt
set KMP_DUPLICATE_LIB_OK=TRUE
```

### Basic Usage
```bash
# Simple Bayesian optimization with GPU
python -m src.cli run --method bayes_oos --models Net1 --device cuda --oos-start-date 200001
```

## Codebase Structure

### Core Implementation

#### `/src/models/nns.py`
- **8 Neural Network Architectures**: Net1-Net5 (progressive complexity), DNet1-DNet3 (deep networks)
- **Thread-safe implementation** for parallel training
- **Memory requirements**: Net1 (~1GB) to DNet3 (~4GB GPU memory)

#### `/src/experiments/`
- **In-sample optimization**: `bayes_is_0.py`, `grid_is_0.py`, `random_is_0.py`
- **Out-of-sample evaluation**: `bayes_oos_1.py`, `grid_oos_1.py`, `random_oos_1.py`
- **Advanced analysis**: `variable_importance_8.py`, `gamma_sensitivity_9.py`, `profit_optimization_10.py`

#### `/src/utils/`
- **Training modules**: `training_optuna.py`, `training_grid.py`, `training_random.py`
- **Metrics calculation**: `metrics_unified.py`
- **Data handling**: `io.py`, `oos_common.py`
- **Performance optimization**: `parallel_helpers.py`, `resource_manager.py`

#### `/src/configs/search_spaces.py`
- Hyperparameter search spaces for all optimization methods
- Model-specific configurations

### Data Structure

#### `/data/ml_equity_premium_data.xlsx`
- **Main sheet**: 31 predictor variables + log equity premium target
- **FRED_MD sheet**: Macroeconomic variables (FRED database)
- **NewlyIdentifiedVariables sheet**: Additional predictors

## Usage Guide

### 1. In-Sample Hyperparameter Optimization

Find optimal hyperparameters using your preferred method:

```bash
# Grid Search
python -m src.cli run --method grid --models Net1 Net2 Net3 --epochs 75

# Random Search  
python -m src.cli run --method random --models Net1 Net2 Net3 --trials 50 --epochs 75

# Bayesian Optimization
python -m src.cli run --method bayes --models Net1 Net2 Net3 --trials 50 --epochs 75
```

### 2. Out-of-Sample Evaluation

Test models on unseen data:

```bash
# Basic OOS evaluation
python -m src.cli run --method bayes_oos --models Net1 Net2 Net3 --trials 10 --epochs 30 --oos-start-date 199001

# With performance optimization
python -m src.cli run --method bayes_oos --models Net1 Net2 Net3 --parallel-trials --hpo-jobs 8
```

### 3. Advanced Experiments

#### Variable Importance Analysis
```bash
python -m src.cli run --method variable_importance_8 --data-source original --n-jobs 4
```

#### Economic Value Analysis
```bash
python -m src.cli economic-value --runs-dir ./runs --data-path ./data --oos-start-date 199001
```

#### Profit Optimization
```bash
python -m src.cli run --method profit_optimization_10 --models Net1 Net2 --data-source original
```

### 4. Window-Based Analysis

#### Rolling Windows
```bash
python -m src.cli run --method rolling_bayes --models Net1 Net2 --window-sizes 5,10,20 --oos-start-date 199001
```

#### Expanding Windows
```bash
python -m src.cli run --method expanding_bayes --models Net1 Net2 --window-sizes 1,3 --oos-start-date 199001
```

## CLI Reference

### Common Parameters
- `--models`: Model selection (Net1-5, DNet1-3)
- `--trials`: Number of HPO trials for Random/Bayesian methods
- `--epochs`: Training epochs per configuration
- `--device`: Computing device (`cpu`, `cuda`, `auto`)
- `--oos-start-date`: Out-of-sample start date (YYYYMM format)

### Performance Options
- `--parallel-trials`: Enable parallel HPO execution
- `--parallel-models`: Enable parallel model processing  
- `--hpo-jobs N`: Number of parallel HPO workers
- `--server-mode`: Optimize for high-core server systems

## Data Sources

### Standard Variables (31 predictors)
- **Valuation ratios**: DP, DY, EP, BM
- **Interest rates**: TBL, LTR, LTY, TMS  
- **Credit spreads**: DFY, DFR
- **Technical indicators**: MA_*, MOM_*, VOL_*

### Extended Datasets
```bash
# FRED macroeconomic variables
python -m src.cli run --method fred_variables --models Net1 Net2 --optimization-method bayes

# Newly identified variables
python -m src.cli run --method newly_identified --models Net1 Net2 --integration-mode standalone
```

## Output Structure

Results are saved to `/runs/` with organized subdirectories:

- **0_[Method]_In_Sample/**: In-sample optimization results
- **1_[Method]_OOS/**: Out-of-sample evaluation results  
- **2_Economic_Value_Analysis/**: Market timing performance
- **[N]_[Experiment_Name]/**: Specialized experiment results

Each run contains:
- Model parameters (`*_best_params.pkl`)
- Performance metrics (`final_metrics.csv`)
- Predictions (`*_predictions.csv`)
- Scalers (`scaler_x.pkl`, `scaler_y.pkl`)

## Neural Network Architectures

| Model | Layers | Complexity | Use Case |
|-------|--------|------------|----------|
| Net1 | 1 hidden | Lowest | Quick experiments, baseline |
| Net2 | 2 hidden | Low | Standard comparisons |
| Net3 | 3 hidden | Medium | Balanced performance |
| Net4 | 4 hidden + skip | Medium-High | Complex patterns |
| Net5 | 5 hidden + skip | High | Maximum capacity |
| DNet1 | 4 deep + BatchNorm | High | Deep learning approach |
| DNet2 | 5 deep + BatchNorm | High | Advanced deep learning |
| DNet3 | 5 deep + more neurons | Highest | Maximum complexity |

## Performance Optimization

The framework includes comprehensive performance optimization features. See `OPTIMIZATION_GUIDE.md` for detailed hardware-specific configuration and safety guidelines.

## Dependencies

- **PyTorch**: Neural network implementation
- **Optuna**: Bayesian optimization
- **Scikit-learn**: Preprocessing and metrics
- **Pandas/NumPy**: Data manipulation
- **Matplotlib**: Visualization

## Research Documentation

- **NEURAL_NETWORKS_EXPLAINED.md**: Comprehensive guide to neural network concepts and implementation
- **OPTIMIZATION_GUIDE.md**: Performance optimization and hardware configuration guide

---

*For theoretical background and detailed neural network explanations, see NEURAL_NETWORKS_EXPLAINED.md*