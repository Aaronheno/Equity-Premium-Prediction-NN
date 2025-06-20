# Performance Optimization and Hardware Configuration Guide

This guide covers performance optimization strategies, hardware-specific configurations, and safety measures for running neural network experiments across different computing environments.

## Safety Guarantees

### Laptop Safety Features

**Complete backward compatibility is guaranteed** - your existing workflow will work exactly as before:

```bash
# This command works identically to the original implementation
python -m src.cli run --method bayes_oos --models Net1 Net2
```

#### Automatic Protection Systems

1. **ResourceManager Laptop Detection**
   - Automatically detects laptop vs desktop vs server environments
   - Conservative limits: Laptops limited to 1-2 workers maximum
   - Always reserves 2+ CPU cores for system stability
   - Memory-aware processing to prevent overload

2. **Default Safety Settings**
   ```python
   # All parallelization defaults to OFF
   parallel_models=False          # Model parallelism OFF by default
   parallel_trials=False          # HPO parallelism OFF by default
   hpo_jobs=1                     # Single-threaded by default
   ```

3. **Opt-In Performance Features**
   ```bash
   --parallel-trials              # Must be explicitly enabled
   --parallel-models              # Must be explicitly enabled  
   --hpo-jobs N                   # Defaults to 1 unless specified
   ```

#### Laptop Resource Limits

When ResourceManager detects a laptop system:

```python
if self.system_type == "LAPTOP":
    # Very conservative for laptops
    if task_type == "hpo":
        return max(1, min(2, self.cpu_count - 2))  # Max 2 workers for HPO
    else:
        return 1  # Single-threaded by default on laptops
```

**Safety Layers:**
- **Layer 1**: All new features default to OFF
- **Layer 2**: Automatic laptop detection with conservative limits
- **Layer 3**: Runtime validation with fallback to sequential processing
- **Layer 4**: Graceful error handling and automatic recovery

## Hardware-Specific Optimization

### High-End Workstations (16-32 cores)

**Recommended Configuration:**
```bash
# Enable both trial and model parallelism
python -m src.cli run --method bayes_oos --models Net1 Net2 Net3 Net4 --parallel-trials --parallel-models --hpo-jobs 20

# Variable importance with parallel processing
python -m src.cli run --method variable_importance_8 --data-source original --n-jobs 16
```

**Expected Performance:**
- HPO trial parallelism: 4-8x speedup
- Model-level parallelism: 3-6x additional speedup
- Combined: 10-20x total improvement

### Server Systems (32+ cores)

**Auto-Detection Benefits:**
Your server will be classified as "HPC_SERVER" and automatically receive:

```python
{
    'hpo_jobs': 28,                    # Use most cores for HPO trials
    'model_parallel_workers': 8,       # All 8 models simultaneously  
    'data_loader_workers': 8,          # Fast data loading
    'use_gpu_parallel': True,          # Multi-GPU if available
    'memory_conserving_mode': False,   # Use full RAM capacity
    'max_trials_per_model': 1000       # High trial counts feasible
}
```

**Server-Optimized Commands:**
```bash
# Automatic server detection (recommended)
python -m src.cli run --method bayes_oos --models Net1 DNet1 --server-mode

# Manual high-performance configuration
python -m src.cli run --method bayes_oos --models Net1 Net2 Net3 Net4 Net5 --parallel-trials --parallel-models --hpo-jobs 24

# Memory-optimized for high-RAM servers (256GB+)
python -m src.cli run --method bayes_oos --memory-gb 512 --nested-parallelism
```

## Performance Scaling Guide

### Standard Hardware (4-16 cores)
```bash
# Safe optimization with conservative threading
python -m src.cli run --method bayes_oos --models Net1 Net2 Net3 --trials 20 --epochs 50

# Enable parallel model processing for faster execution
python -m src.cli run --method bayes_oos --models Net1 Net2 Net3 DNet1 --parallel-models --trials 30
```

**Expected improvements:** 2-5x speedup

### High-End Systems (32+ cores, 64GB+ RAM)
```bash
# Aggressive parallelization with higher trial counts
python -m src.cli run --method bayes_oos --models Net1 Net2 Net3 Net4 Net5 --trials 100 --hpo-jobs 16 --parallel-models

# Multiple window analysis with parallel processing
python -m src.cli run --method rolling_bayes --models Net1 Net2 Net3 --window-sizes 5,10,20 --parallel-windows --trials 50
```

**Expected improvements:** 5-20x speedup under optimal conditions

### HPC/Server Systems (64+ cores, 128GB+ RAM)
```bash
# Maximum parallelization with automatic server detection
python -m src.cli run --method bayes_oos --models Net1 Net2 Net3 Net4 Net5 DNet1 DNet2 DNet3 --server-mode --trials-multiplier 4.0

# Nested parallelism for ultimate performance
python -m src.cli run --method bayes_oos --models Net1 Net2 Net3 Net4 Net5 DNet1 DNet2 DNet3 --nested-parallelism --max-cores 64 --trials 500
```

**Expected improvements:** 10-50x speedup under optimal conditions

## System Requirements by Experiment Type

### Grid Search Experiments
- **Memory**: 4-16GB depending on parameter space size
- **CPU**: Highly parallelizable, benefits from many cores
- **Optimal**: 16+ cores for maximum efficiency

### Bayesian Optimization  
- **Memory**: 2-8GB for trial history and surrogate models
- **CPU**: Moderately parallelizable, 4-16 cores optimal
- **Database**: SQLite sufficient, PostgreSQL for distributed setups

### Variable Importance Analysis
- **Memory**: 8-32GB for permutation testing across variables
- **CPU**: Extremely parallelizable, scales well to 64+ cores
- **Storage**: Temporary space for intermediate results

### Window-Based Analysis
- **Memory**: 16-64GB for multiple time windows simultaneously  
- **CPU**: Benefits from both model and window parallelism
- **Storage**: High I/O for rolling data windows

## Memory Management

### Conservative Mode (Default)
- Single model in memory at a time
- Batch sizes: 32-256
- Memory usage: 2-8GB typical

### Aggressive Mode (High-end systems)
- Multiple models cached simultaneously
- Batch sizes: 512-2048  
- Memory usage: 16-64GB

### Memory Optimization Commands
```bash
# Conservative memory usage
python -m src.cli run --method bayes_oos --models Net1 --batch 128 --conservative-memory

# Aggressive memory usage for high-RAM systems
python -m src.cli run --method bayes_oos --models Net1 Net2 Net3 --batch 1024 --memory-gb 128
```

## GPU Optimization

### Single GPU Systems
```bash
# Automatic GPU detection and optimization
python -m src.cli run --method bayes_oos --models Net1 Net2 --device cuda
```

### Multi-GPU Systems
```bash
# Automatic multi-GPU utilization
python -m src.cli run --method bayes_oos --models Net1 Net2 Net3 Net4 --device cuda --parallel-models
```

**GPU Memory Requirements:**
- Net1: ~1GB
- Net2: ~1.5GB  
- Net3: ~2GB
- Net4: ~2.5GB
- Net5: ~4GB
- DNet1-3: ~3-4GB each

## Testing and Validation

### System Capability Check
```bash
# Check what your system supports
python -m src.cli run --resource-info
```

### Safe Testing Approach
```bash
# 1. Start with basic functionality
python -m src.cli run --method bayes_oos --models Net1 --trials 10

# 2. Try HPO parallelism (if recommended by resource check)
python -m src.cli run --method bayes_oos --models Net1 --parallel-trials --trials 20

# 3. Scale up gradually
python -m src.cli run --method bayes_oos --models Net1 Net2 --parallel-models --parallel-trials
```

## Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce batch size or use fewer parallel processes
2. **System Slowdown**: Reduce `--hpo-jobs` or disable `--parallel-models`
3. **Process Hangs**: Check for deadlocks in nested parallelism

### Emergency Fallbacks
All parallel operations automatically fall back to sequential processing if:
- Insufficient system resources detected
- Parallel execution fails
- User interrupts with Ctrl+C

### Performance Monitoring
```bash
# Monitor system resources during execution
# The framework automatically logs resource usage and warnings
```

## Cloud and Container Deployment

### AWS/GCP Instances
```bash
# Auto-detect cloud instance capabilities
python -m src.cli run --method bayes_oos --models Net1 Net2 Net3 --trials 50 --device auto
```

### Docker Containers
```bash
# Respect container resource limits
python -m src.cli run --method bayes_oos --models Net1 Net2 --memory-gb 8 --max-cores 4
```

## Performance Disclaimers

**Important**: Performance improvements depend heavily on:
- **Hardware specifications** (CPU cores, RAM, GPU capabilities)
- **Dataset size and complexity**
- **System configuration and available resources**
- **Concurrent system load**

Actual speedups may vary significantly from estimates. The framework is designed to provide substantial improvements under optimal conditions while maintaining safety and reliability across all hardware configurations.

---

*This optimization guide ensures maximum performance while maintaining complete safety and backward compatibility across all computing environments.*