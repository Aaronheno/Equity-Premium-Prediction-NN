"""
Resource Manager for Adaptive Performance Optimization

This module provides intelligent resource detection and allocation for the neural network
experiments. It ensures safe operation on resource-constrained systems while enabling
massive parallelization on HPC systems.

Key Features:
    - Automatic hardware detection with conservative defaults
    - Laptop-safe operation (never uses all cores)
    - Gradual scaling based on available resources
    - Memory-aware batch size recommendations
    - Fallback mechanisms for resource constraints

Threading Safety: THREAD_SAFE
Hardware Compatibility: ALL_SYSTEMS (4-core laptop to 128-core HPC)
Default Behavior: CONSERVATIVE (preserves existing single-threaded operation)
"""

import os
import sys
import psutil
import torch
import multiprocessing as mp
from typing import Dict, Optional, Tuple
import warnings


class ResourceManager:
    """
    Intelligent resource manager that adapts to available hardware.
    
    Design Principles:
        1. Safety first - never overwhelm the system
        2. Explicit opt-in - parallelization requires user consent
        3. Laptop-friendly - conservative defaults for small systems
        4. Progressive enhancement - scale up on better hardware
    """
    
    def __init__(self, verbose: bool = False):
        """Initialize resource detection with safety checks."""
        self.verbose = verbose
        
        # Hardware detection
        self.cpu_count = mp.cpu_count()
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        self.available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # GPU detection with safety
        try:
            self.gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
            self.gpu_memory_gb = self._get_gpu_memory() if self.gpu_count > 0 else []
        except Exception as e:
            if self.verbose:
                print(f"Warning: GPU detection failed: {e}", file=sys.stderr)
            self.gpu_count = 0
            self.gpu_memory_gb = []
        
        # System classification
        self.system_type = self._classify_system()
        
        if self.verbose:
            self._print_system_info()
    
    def _classify_system(self) -> str:
        """Classify system type based on available resources."""
        if self.cpu_count >= 64 and self.memory_gb >= 128:
            return "HPC_SERVER"
        elif self.cpu_count >= 16 and self.memory_gb >= 32:
            return "WORKSTATION"
        elif self.cpu_count >= 8:
            return "DESKTOP"
        else:
            return "LAPTOP"
    
    def _get_gpu_memory(self) -> list:
        """Safely get GPU memory information."""
        gpu_memory = []
        for i in range(self.gpu_count):
            try:
                props = torch.cuda.get_device_properties(i)
                gpu_memory.append(props.total_memory / (1024**3))
            except Exception:
                gpu_memory.append(0)
        return gpu_memory
    
    def _print_system_info(self):
        """Print detected system information."""
        print(f"\n{'='*60}")
        print(f"System Resource Detection")
        print(f"{'='*60}")
        print(f"CPU Cores: {self.cpu_count}")
        print(f"Total Memory: {self.memory_gb:.1f} GB")
        print(f"Available Memory: {self.available_memory_gb:.1f} GB")
        print(f"GPU Count: {self.gpu_count}")
        if self.gpu_count > 0:
            print(f"GPU Memory: {[f'{m:.1f} GB' for m in self.gpu_memory_gb]}")
        print(f"System Type: {self.system_type}")
        print(f"{'='*60}\n")
    
    def get_safe_worker_count(self, 
                            task_type: str = "general",
                            requested_workers: Optional[int] = None,
                            leave_free_cores: int = 2) -> int:
        """
        Get safe number of worker processes for parallel execution.
        
        Args:
            task_type: Type of task ("hpo", "model_parallel", "data_loading", etc.)
            requested_workers: User-requested number of workers (None = auto)
            leave_free_cores: Number of cores to leave free for system (laptop-safe)
            
        Returns:
            Safe number of workers that won't overwhelm the system
        """
        # If user explicitly requested workers, validate and use
        if requested_workers is not None:
            max_safe = max(1, self.cpu_count - leave_free_cores)
            if requested_workers > max_safe:
                warnings.warn(f"Requested {requested_workers} workers exceeds safe limit "
                            f"({max_safe}). Using {max_safe} instead.")
                return max_safe
            return max(1, requested_workers)
        
        # Auto-detection based on system type and task
        if self.system_type == "LAPTOP":
            # Very conservative for laptops
            if task_type == "hpo":
                return max(1, min(2, self.cpu_count - 2))
            else:
                return 1  # Single-threaded by default on laptops
                
        elif self.system_type == "DESKTOP":
            # Moderate parallelism for desktops
            if task_type == "hpo":
                return max(1, min(4, self.cpu_count - 2))
            elif task_type == "model_parallel":
                return max(1, min(2, self.cpu_count // 2))
            elif task_type == "window_parallel":
                return max(1, min(2, self.cpu_count // 4))
            else:
                return max(1, self.cpu_count // 4)
                
        elif self.system_type == "WORKSTATION":
            # Good parallelism for workstations
            if task_type == "hpo":
                return max(1, min(8, self.cpu_count - 2))
            elif task_type == "model_parallel":
                return max(1, min(4, self.cpu_count // 2))
            elif task_type == "window_parallel":
                return max(1, min(3, self.cpu_count // 3))
            else:
                return max(1, self.cpu_count // 2)
                
        else:  # HPC_SERVER
            # Aggressive parallelism for servers
            if task_type == "hpo":
                return max(1, min(32, self.cpu_count - 4))
            elif task_type == "model_parallel":
                return min(8, self.cpu_count // 4)
            elif task_type == "window_parallel":
                return max(1, min(4, self.cpu_count // 8))
            else:
                return self.cpu_count // 2
    
    def get_optimal_batch_size(self, 
                             model_type: str = "Net1",
                             device: str = "cpu") -> int:
        """
        Get optimal batch size based on available memory.
        
        Args:
            model_type: Neural network model type
            device: Target device ("cpu" or "cuda")
            
        Returns:
            Recommended batch size that fits in memory
        """
        # Model memory requirements (approximate)
        model_memory_mb = {
            "Net1": 100, "Net2": 150, "Net3": 200,
            "Net4": 250, "Net5": 400,
            "DNet1": 300, "DNet2": 350, "DNet3": 400
        }
        
        base_memory = model_memory_mb.get(model_type, 200)
        
        if device == "cuda" and self.gpu_count > 0:
            # GPU batch sizing
            min_gpu_memory = min(self.gpu_memory_gb) if self.gpu_memory_gb else 4
            if min_gpu_memory > 8:
                return 512
            elif min_gpu_memory > 4:
                return 256
            else:
                return 128
        else:
            # CPU batch sizing based on available memory
            if self.available_memory_gb > 16:
                return 256
            elif self.available_memory_gb > 8:
                return 128
            else:
                return 64
    
    def should_enable_parallelism(self, 
                                explicit_request: bool = False,
                                min_cores_required: int = 4) -> bool:
        """
        Determine if parallelism should be enabled.
        
        Args:
            explicit_request: User explicitly requested parallelism
            min_cores_required: Minimum cores needed for parallelism
            
        Returns:
            True if parallelism should be enabled
        """
        # Always respect explicit user request
        if explicit_request:
            return True
        
        # Otherwise, only enable on suitable systems
        return (self.cpu_count >= min_cores_required and 
                self.system_type in ["WORKSTATION", "HPC_SERVER"])
    
    def get_threading_config(self, experiment_type: str, 
                           user_config: Optional[Dict] = None) -> Dict:
        """
        Get complete threading configuration for an experiment.
        
        Args:
            experiment_type: Type of experiment being run
            user_config: User-provided configuration overrides
            
        Returns:
            Complete configuration with safe defaults
        """
        # Start with safe defaults
        config = {
            'parallel_models': False,
            'parallel_trials': False,
            'hpo_jobs': 1,
            'model_workers': 1,
            'data_workers': 0,
            'batch_size': 128,
            'device': 'cpu',
            'memory_limit_gb': None,
            'verbose_threading': False
        }
        
        # Apply user overrides if provided
        if user_config:
            config.update(user_config)
        
        # Only enable parallelism if explicitly requested or on suitable hardware
        if config.get('parallel_models') or config.get('parallel_trials'):
            # User explicitly requested parallelism
            config['hpo_jobs'] = self.get_safe_worker_count("hpo", 
                                                           config.get('hpo_jobs'))
            config['model_workers'] = self.get_safe_worker_count("model_parallel",
                                                               config.get('model_workers'))
        elif self.should_enable_parallelism():
            # Auto-enable on suitable hardware (but keep it conservative)
            if self.verbose:
                print(f"Note: Parallelism available on this {self.system_type} system. "
                      f"Use --parallel-models or --parallel-trials to enable.")
        
        # Adjust batch size based on device and memory
        if config['batch_size'] == 128:  # Default, can be optimized
            config['batch_size'] = self.get_optimal_batch_size(
                experiment_type.split('_')[0], config['device'])
        
        return config
    
    def monitor_resource_usage(self) -> Dict:
        """Monitor current resource usage for adaptive throttling."""
        return {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'available_memory_gb': psutil.virtual_memory().available / (1024**3),
            'load_average': os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0
        }
    
    def should_throttle(self, thresholds: Optional[Dict] = None) -> bool:
        """
        Check if we should throttle parallelism due to resource constraints.
        
        Args:
            thresholds: Custom thresholds for throttling
            
        Returns:
            True if system is under stress and we should reduce parallelism
        """
        if thresholds is None:
            thresholds = {
                'cpu_percent': 90,
                'memory_percent': 85,
                'min_available_gb': 2
            }
        
        usage = self.monitor_resource_usage()
        
        return (usage['cpu_percent'] > thresholds['cpu_percent'] or
                usage['memory_percent'] > thresholds['memory_percent'] or
                usage['available_memory_gb'] < thresholds['min_available_gb'])


# Convenience functions for backward compatibility
_default_manager = None

def get_resource_manager(verbose: bool = False) -> ResourceManager:
    """Get or create the default resource manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = ResourceManager(verbose=verbose)
    return _default_manager

def get_safe_n_jobs(task_type: str = "general", 
                   requested: Optional[int] = None) -> int:
    """Convenience function to get safe worker count."""
    manager = get_resource_manager()
    return manager.get_safe_worker_count(task_type, requested)