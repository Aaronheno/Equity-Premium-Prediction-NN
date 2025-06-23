'use client'

import { motion } from 'framer-motion'
import { useState } from 'react'
import { Code, Play, Database, Settings, Target, BarChart3, Zap, CheckCircle, GitBranch, Clock } from 'lucide-react'
import CodeBlock from '@/components/shared/CodeBlock'
import MathFormula from '@/components/shared/MathFormula'
import NavigationButtons from '@/components/shared/NavigationButtons'

const pipelineStages = [
  {
    stage: 'Data Preparation',
    description: 'Load, validate, and preprocess financial indicators',
    components: ['Data Loading', 'Missing Value Handling', 'Feature Engineering', 'Temporal Alignment'],
    duration: 'Fast',
    icon: Database,
    color: 'accent-blue'
  },
  {
    stage: 'Model Training',
    description: 'Train neural networks with hyperparameter optimization',
    components: ['Architecture Selection', 'Hyperparameter Search', 'Training Loop', 'Validation'],
    duration: 'Variable',
    icon: Settings,
    color: 'accent-orange'
  },
  {
    stage: 'Prediction Generation',
    description: 'Generate forecasts and process outputs',
    components: ['Model Inference', 'Output Scaling', 'Prediction Storage', 'Results Compilation'],
    duration: 'Fast',
    icon: Target,
    color: 'accent-green'
  },
  {
    stage: 'Performance Evaluation',
    description: 'Comprehensive model assessment and validation',
    components: ['Regression Metrics', 'Financial Metrics', 'Statistical Tests', 'Reporting'],
    duration: 'Moderate',
    icon: BarChart3,
    color: 'accent-purple'
  }
]

const configurationOptions = [
  {
    category: 'Data Configuration',
    options: [
      { name: 'data_file', default: 'ml_equity_premium_data.xlsx', description: 'Input data file path' },
      { name: 'target_column', default: 'equity_premium', description: 'Target variable column name' },
      { name: 'train_ratio', default: 0.7, description: 'Training data proportion' },
      { name: 'val_ratio', default: 0.15, description: 'Validation data proportion' }
    ]
  },
  {
    category: 'Model Configuration',
    options: [
      { name: 'models_to_train', default: "['Net1', 'Net3', 'DNet2']", description: 'Neural network architectures' },
      { name: 'optimization_method', default: 'bayesian', description: 'Hyperparameter optimization approach' },
      { name: 'n_trials', default: 100, description: 'Number of optimization trials' },
      { name: 'max_epochs', default: 200, description: 'Maximum training epochs' }
    ]
  },
  {
    category: 'Evaluation Configuration',
    options: [
      { name: 'evaluation_metrics', default: "['rmse', 'r2', 'sharpe']", description: 'Performance metrics to compute' },
      { name: 'significance_level', default: 0.05, description: 'Statistical significance threshold' },
      { name: 'bootstrap_samples', default: 1000, description: 'Bootstrap iterations for tests' },
      { name: 'output_dir', default: 'results/', description: 'Output directory for results' }
    ]
  }
]

const workflowSteps = [
  {
    step: 1,
    title: 'Environment Setup',
    description: 'Initialize dependencies and configuration',
    code: 'pipeline.setup_environment(config)',
    status: 'Fast'
  },
  {
    step: 2,
    title: 'Data Loading',
    description: 'Load and validate financial data',
    code: 'data = pipeline.load_data(config.data_file)',
    status: 'Fast'
  },
  {
    step: 3,
    title: 'Preprocessing',
    description: 'Scale features and create temporal splits',
    code: 'X_train, y_train = pipeline.preprocess(data)',
    status: 'Fast'
  },
  {
    step: 4,
    title: 'Model Training',
    description: 'Train models with hyperparameter optimization',
    code: 'models = pipeline.train_models(X_train, y_train)',
    status: 'Intensive'
  },
  {
    step: 5,
    title: 'Prediction',
    description: 'Generate out-of-sample predictions',
    code: 'predictions = pipeline.predict(models, X_test)',
    status: 'Fast'
  },
  {
    step: 6,
    title: 'Evaluation',
    description: 'Comprehensive performance assessment',
    code: 'results = pipeline.evaluate(y_test, predictions)',
    status: 'Moderate'
  },
  {
    step: 7,
    title: 'Reporting',
    description: 'Generate reports and visualizations',
    code: 'pipeline.generate_report(results)',
    status: 'Fast'
  }
]

export default function CompletePipelineContent() {
  const [activeStage, setActiveStage] = useState(0)
  const [selectedConfig, setSelectedConfig] = useState(0)

  return (
    <div className="min-h-screen bg-bg-primary pt-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-16"
        >
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold mb-6">
            <span className="gradient-text">Complete Pipeline</span>
          </h1>
          <p className="max-w-4xl mx-auto text-xl text-text-secondary leading-relaxed">
            End-to-end implementation: unified framework integrating all components from data 
            preprocessing through model training to comprehensive evaluation and reporting.
          </p>
        </motion.div>

        {/* Pipeline Overview */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Play className="w-8 h-8 text-accent-blue mr-3" />
            End-to-End Workflow
          </h2>
          
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8 mb-8">
            <p className="text-text-secondary text-lg mb-6 leading-relaxed">
              The complete pipeline integrates all components into a unified framework that can be 
              executed with a single command. From raw financial data to publication-ready results, 
              every step is automated while maintaining full transparency and configurability.
            </p>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div>
                <h3 className="text-xl font-semibold text-text-primary mb-4">Pipeline Benefits:</h3>
                <div className="space-y-3 text-sm">
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-blue mt-2"></div>
                    <span className="text-text-secondary"><strong>Reproducibility:</strong> Consistent results across runs</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-blue mt-2"></div>
                    <span className="text-text-secondary"><strong>Automation:</strong> Minimal manual intervention required</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-blue mt-2"></div>
                    <span className="text-text-secondary"><strong>Configurability:</strong> Easy parameter adjustment</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-blue mt-2"></div>
                    <span className="text-text-secondary"><strong>Scalability:</strong> Handles multiple models and datasets</span>
                  </div>
                </div>
              </div>
              
              <div>
                <h3 className="text-xl font-semibold text-text-primary mb-4">Quick Start:</h3>
                <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 space-y-2 text-sm font-mono">
                  <div className="text-accent-green"># Install dependencies</div>
                  <div>pip install -r requirements.txt</div>
                  <div className="text-accent-green"># Run Bayesian OOS experiment</div>
                  <div className="text-accent-blue">python -m src.cli run --method bayes_oos --models Net1 Net2 Net3</div>
                  <div className="text-accent-green"># Or use direct script</div>
                  <div className="text-accent-blue">python -m src.experiments.bayes_oos_1</div>
                </div>
              </div>
            </div>
          </div>
        </motion.section>

        {/* Four-Stage Pipeline */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <GitBranch className="w-8 h-8 text-accent-orange mr-3" />
            Four-Stage Architecture
          </h2>
          
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8 mb-8">
            <p className="text-text-secondary text-lg mb-8">
              The pipeline is organized into four main stages, each handling specific aspects 
              of the machine learning workflow. Click each stage to explore its components 
              and implementation details.
            </p>

            {/* Stage Selector */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
              {pipelineStages.map((stage, index) => {
                const Icon = stage.icon
                const isActive = activeStage === index
                return (
                  <button
                    key={stage.stage}
                    onClick={() => setActiveStage(index)}
                    className={`p-4 rounded-lg border transition-all text-left ${
                      isActive 
                        ? `bg-${stage.color}/10 border-${stage.color}/30` 
                        : 'bg-bg-primary border-bg-tertiary hover:border-accent-blue/30'
                    }`}
                  >
                    <div className="flex items-center space-x-3 mb-3">
                      <div className={`inline-flex items-center justify-center w-8 h-8 rounded-lg bg-${stage.color}/10 border border-${stage.color}/20`}>
                        <Icon className={`w-4 h-4 text-${stage.color}`} />
                      </div>
                      <div>
                        <div className="text-xs text-text-muted">Stage {index + 1}</div>
                        <div className="font-semibold text-text-primary text-sm">{stage.stage}</div>
                      </div>
                    </div>
                    <div className="text-xs text-accent-blue">{stage.duration}</div>
                  </button>
                )
              })}
            </div>

            {/* Active Stage Details */}
            {pipelineStages.map((stage, index) => {
              const Icon = stage.icon
              if (activeStage !== index) return null
              
              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                  className={`bg-${stage.color}/5 border border-${stage.color}/20 rounded-xl p-8`}
                >
                  <div className="flex items-center space-x-3 mb-6">
                    <div className={`inline-flex items-center justify-center w-12 h-12 rounded-lg bg-${stage.color}/10 border border-${stage.color}/20`}>
                      <Icon className={`w-6 h-6 text-${stage.color}`} />
                    </div>
                    <div>
                      <h3 className="text-2xl font-bold text-text-primary">{stage.stage}</h3>
                      <p className="text-text-secondary">{stage.description}</p>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    <div>
                      <h4 className="font-semibold text-text-primary mb-3">Key Components:</h4>
                      <div className="space-y-2 text-sm">
                        {stage.components.map((component, idx) => (
                          <div key={idx} className="flex items-center space-x-3">
                            <div className={`w-2 h-2 rounded-full bg-${stage.color}`}></div>
                            <span className="text-text-secondary">{component}</span>
                          </div>
                        ))}
                      </div>
                      
                      <div className={`mt-4 bg-${stage.color}/10 border border-${stage.color}/20 rounded-lg p-4`}>
                        <div className={`text-${stage.color} text-sm font-medium mb-1`}>Execution Time:</div>
                        <div className="text-text-secondary text-sm">{stage.duration}</div>
                      </div>
                    </div>
                    
                    <div>
                      <h4 className="font-semibold text-text-primary mb-3">Implementation Highlights:</h4>
                      <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 text-sm">
                        {index === 0 && (
                          <div className="space-y-2">
                            <div>• Automated data validation and quality checks</div>
                            <div>• Configurable feature engineering pipeline</div>
                            <div>• Temporal alignment ensuring no look-ahead bias</div>
                            <div>• Robust handling of missing values and outliers</div>
                          </div>
                        )}
                        {index === 1 && (
                          <div className="space-y-2">
                            <div>• Multiple architecture support (Net1-5, DNet1-3)</div>
                            <div>• Bayesian optimization for hyperparameter tuning</div>
                            <div>• Early stopping and checkpoint management</div>
                            <div>• GPU acceleration with automatic fallback to CPU</div>
                          </div>
                        )}
                        {index === 2 && (
                          <div className="space-y-2">
                            <div>• Out-of-sample prediction generation</div>
                            <div>• Inverse scaling of model outputs</div>
                            <div>• CSV output for analysis and visualization</div>
                            <div>• Prediction tracking across time periods</div>
                          </div>
                        )}
                        {index === 3 && (
                          <div className="space-y-2">
                            <div>• Out-of-sample R², Success Ratio, CER calculation</div>
                            <div>• Clark-West and Pesaran-Timmermann tests</div>
                            <div>• CSV metrics output for further analysis</div>
                            <div>• Model comparison and ranking</div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </motion.div>
              )
            })}
          </div>
        </motion.section>

        {/* Configuration Management */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Settings className="w-8 h-8 text-accent-green mr-3" />
            Configuration Management
          </h2>
          
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8 mb-8">
            <p className="text-text-secondary text-lg mb-8">
              The pipeline is fully configurable through command-line arguments and Python configuration files. 
              Search spaces are defined in src/configs/search_spaces.py, enabling systematic experimentation 
              across different model architectures and hyperparameter ranges.
            </p>

            {/* Configuration Categories */}
            <div className="flex flex-wrap gap-2 mb-8">
              {configurationOptions.map((category, index) => (
                <button
                  key={category.category}
                  onClick={() => setSelectedConfig(index)}
                  className={`px-4 py-2 rounded-lg font-medium transition-all ${
                    selectedConfig === index
                      ? 'bg-accent-green text-white'
                      : 'bg-bg-primary text-text-secondary hover:text-accent-green'
                  }`}
                >
                  {category.category}
                </button>
              ))}
            </div>

            {/* Active Configuration Options */}
            <motion.div
              key={selectedConfig}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              className="space-y-4"
            >
              {configurationOptions[selectedConfig].options.map((option) => (
                <div key={option.name} className="bg-bg-primary border border-bg-tertiary rounded-lg p-4">
                  <div className="flex items-start justify-between mb-2">
                    <div>
                      <h4 className="font-mono font-semibold text-text-primary">{option.name}</h4>
                      <p className="text-text-secondary text-sm">{option.description}</p>
                    </div>
                    <div className="font-mono text-accent-green text-sm bg-accent-green/10 px-2 py-1 rounded">
                      {option.default.toString()}
                    </div>
                  </div>
                </div>
              ))}
            </motion.div>

            <div className="mt-8 bg-accent-green/5 border border-accent-green/20 rounded-lg p-6">
              <h3 className="text-xl font-semibold text-accent-green mb-4">Example Configuration File</h3>
              <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 overflow-x-auto">
                <pre className="text-sm font-mono text-text-secondary">
{`# config.yaml
data:
  file: "data/ml_equity_premium_data.xlsx"
  target_column: "equity_premium"
  train_ratio: 0.7
  val_ratio: 0.15

models:
  architectures: ["Net3", "DNet2"]
  optimization:
    method: "bayesian"
    n_trials: 100
    timeout: 3600  # 1 hour

training:
  max_epochs: 200
  early_stopping_patience: 20
  device: "auto"  # auto-detect GPU/CPU

evaluation:
  metrics: ["rmse", "r2", "sharpe_ratio", "hit_rate"]
  significance_level: 0.05
  bootstrap_samples: 1000

output:
  directory: "results/"
  save_models: true
  generate_plots: true`}
                </pre>
              </div>
            </div>
          </div>
        </motion.section>

        {/* Step-by-Step Execution */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.5 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Clock className="w-8 h-8 text-accent-purple mr-3" />
            Execution Workflow
          </h2>
          
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8 mb-8">
            <p className="text-text-secondary text-lg mb-8">
              The pipeline executes in seven sequential steps, each with clear inputs, outputs, 
              and estimated execution times. Progress is tracked and logged throughout the process.
            </p>

            <div className="space-y-4">
              {workflowSteps.map((step, index) => (
                <motion.div
                  key={step.step}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.6, delay: 0.6 + index * 0.1 }}
                  className="flex items-center space-x-6 bg-bg-primary border border-bg-tertiary rounded-lg p-6"
                >
                  <div className="flex-shrink-0">
                    <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-accent-purple/10 border border-accent-purple/20">
                      <span className="text-accent-purple font-bold text-lg">{step.step}</span>
                    </div>
                  </div>
                  
                  <div className="flex-1">
                    <h3 className="font-semibold text-text-primary text-lg mb-1">{step.title}</h3>
                    <p className="text-text-secondary text-sm mb-2">{step.description}</p>
                    <code className="text-accent-blue text-sm font-mono bg-accent-blue/5 px-2 py-1 rounded">
                      {step.code}
                    </code>
                  </div>
                  
                  <div className="flex-shrink-0 text-right">
                    <div className="text-accent-purple font-mono text-sm">{step.status}</div>
                    <div className="text-text-muted text-xs">Complexity</div>
                  </div>
                </motion.div>
              ))}
            </div>

            <div className="mt-8 bg-accent-purple/5 border border-accent-purple/20 rounded-lg p-6">
              <h3 className="text-xl font-semibold text-accent-purple mb-4">Pipeline Modes</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-sm">
                <div className="text-center">
                  <div className="font-mono text-accent-purple text-2xl">Full</div>
                  <div className="text-text-muted">Complete Pipeline</div>
                  <div className="text-text-secondary">Including hyperparameter optimization</div>
                </div>
                <div className="text-center">
                  <div className="font-mono text-accent-purple text-2xl">Fast</div>
                  <div className="text-text-muted">Pre-trained Models</div>
                  <div className="text-text-secondary">Using saved hyperparameters</div>
                </div>
                <div className="text-center">
                  <div className="font-mono text-accent-purple text-2xl">Quick</div>
                  <div className="text-text-muted">Prediction Only</div>
                  <div className="text-text-secondary">Using existing trained models</div>
                </div>
              </div>
            </div>
          </div>
        </motion.section>

        {/* Complete Implementation */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Code className="w-8 h-8 text-accent-blue mr-3" />
            Complete Pipeline Implementation
          </h2>
          
          <p className="text-text-secondary text-lg mb-6">
            The complete pipeline implementation brings together all components in a unified, 
            configurable framework that can handle the entire machine learning workflow from 
            data loading to final evaluation and reporting.
          </p>

          <CodeBlock
            language="python"
            title="Complete Neural Network Pipeline for Equity Premium Prediction"
            code={`#!/usr/bin/env python3
"""
Complete Pipeline for Neural Network Equity Premium Prediction

This script provides an end-to-end pipeline for training and evaluating
neural networks on financial data for equity premium prediction.

Usage:
    python main.py --config config.yaml
    python main.py --quick-run  # Use default configuration
"""

import os
import sys
import yaml
import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Import our modules
from src.models.nns import Net1, Net2, Net3, Net4, Net5, DNet1, DNet2, DNet3
from src.utils.io import load_data, save_results
from src.utils.training_optuna import HyperparameterOptimizer
from src.utils.metrics_unified import FinancialEvaluator
from src.experiments.bayes_oos_1 import setup_logging

class EquityPremiumPipeline:
    """
    Complete pipeline for equity premium prediction using neural networks.
    
    Features:
    - Configurable data preprocessing
    - Multiple neural network architectures
    - Hyperparameter optimization
    - Comprehensive evaluation
    - Automated reporting
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logging()
        self.results = {}
        self.models = {}
        self.scalers = {}
        
        # Create output directory
        self.output_dir = Path(config['output']['directory'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seeds for reproducibility
        self._set_random_seeds(config.get('random_seed', 42))
        
        self.logger.info("Pipeline initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        log_dir = self.output_dir / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        return logging.getLogger(__name__)
    
    def _set_random_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def load_and_preprocess_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess financial data.
        
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        self.logger.info("Loading and preprocessing data...")
        start_time = time.time()
        
        # Load data
        data_file = self.config['data']['file']
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        df = pd.read_excel(data_file)
        self.logger.info(f"Loaded data with shape: {df.shape}")
        
        # Extract features and target
        target_col = self.config['data']['target_column']
        feature_cols = [col for col in df.columns if col != target_col and col != 'date']
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        # Handle missing values
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            self.logger.warning("Missing values detected, applying forward fill")
            df_clean = df.fillna(method='ffill').fillna(method='bfill')
            X = df_clean[feature_cols].values
            y = df_clean[target_col].values
        
        # Temporal split (maintaining time order)
        n_samples = len(X)
        train_end = int(self.config['data']['train_ratio'] * n_samples)
        val_end = int((self.config['data']['train_ratio'] + self.config['data']['val_ratio']) * n_samples)
        
        X_train = X[:train_end]
        y_train = y[:train_end]
        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]
        X_test = X[val_end:]
        y_test = y[val_end:]
        
        # Scale features
        self.scalers['X'] = StandardScaler()
        self.scalers['y'] = StandardScaler()
        
        X_train_scaled = self.scalers['X'].fit_transform(X_train)
        X_val_scaled = self.scalers['X'].transform(X_val)
        X_test_scaled = self.scalers['X'].transform(X_test)
        
        y_train_scaled = self.scalers['y'].fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = self.scalers['y'].transform(y_val.reshape(-1, 1)).flatten()
        y_test_scaled = self.scalers['y'].transform(y_test.reshape(-1, 1)).flatten()
        
        # Store data splits
        self.data_splits = {
            'train': (X_train_scaled, y_train_scaled),
            'val': (X_val_scaled, y_val_scaled),
            'test': (X_test_scaled, y_test_scaled),
            'feature_names': feature_cols
        }
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Data preprocessing completed in {elapsed_time:.2f}s")
        self.logger.info(f"Splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled
    
    def train_models(self) -> Dict[str, nn.Module]:
        """
        Train neural network models with hyperparameter optimization.
        
        Returns:
            Dictionary of trained models
        """
        self.logger.info("Starting model training...")
        start_time = time.time()
        
        # Available model classes
        model_classes = {
            'Net1': Net1, 'Net2': Net2, 'Net3': Net3, 'Net4': Net4, 'Net5': Net5,
            'DNet1': DNet1, 'DNet2': DNet2, 'DNet3': DNet3
        }
        
        # Models to train
        models_to_train = self.config['models']['architectures']
        X_train, y_train = self.data_splits['train']
        X_val, y_val = self.data_splits['val']
        
        trained_models = {}
        
        for model_name in models_to_train:
            if model_name not in model_classes:
                self.logger.warning(f"Unknown model: {model_name}, skipping")
                continue
            
            self.logger.info(f"Training {model_name}...")
            model_start = time.time()
            
            # Create hyperparameter optimizer
            optimizer = HyperparameterOptimizer(
                X_train, y_train, X_val, y_val,
                model_class=model_classes[model_name],
                n_trials=self.config['models']['optimization']['n_trials']
            )
            
            # Run optimization
            study = optimizer.optimize(study_name=f"{model_name}_optimization")
            
            # Train final model with best parameters
            best_params = study.best_params
            
            # Create and train final model
            model_params = {k: v for k, v in best_params.items() 
                           if k not in ['lr', 'batch_size', 'optimizer', 'l1_lambda', 'l2_lambda']}
            model_params['n_feature'] = X_train.shape[1]
            model_params['n_output'] = 1
            
            final_model = model_classes[model_name](**model_params)
            
            # Train final model (simplified - would use full training loop)
            final_model.eval()  # For demonstration
            
            trained_models[model_name] = {
                'model': final_model,
                'best_params': best_params,
                'best_score': study.best_value,
                'study': study
            }
            
            model_elapsed = time.time() - model_start
            self.logger.info(f"{model_name} training completed in {model_elapsed:.2f}s")
            self.logger.info(f"Best validation loss: {study.best_value:.6f}")
        
        self.models = trained_models
        
        # Save models if requested
        if self.config['output']['save_models']:
            self._save_models()
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"All model training completed in {elapsed_time:.2f}s")
        
        return trained_models
    
    def generate_predictions(self) -> Dict[str, np.ndarray]:
        """
        Generate predictions for all trained models.
        
        Returns:
            Dictionary mapping model names to predictions
        """
        self.logger.info("Generating predictions...")
        start_time = time.time()
        
        X_test, y_test = self.data_splits['test']
        X_test_tensor = torch.FloatTensor(X_test)
        
        predictions = {}
        
        for model_name, model_info in self.models.items():
            model = model_info['model']
            model.eval()
            
            with torch.no_grad():
                raw_predictions = model(X_test_tensor)
                
                # Convert back to original scale
                pred_scaled = raw_predictions.numpy()
                pred_original = self.scalers['y'].inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
                
                predictions[model_name] = pred_original
        
        # Add baseline prediction
        y_train_original = self.scalers['y'].inverse_transform(
            self.data_splits['train'][1].reshape(-1, 1)
        ).flatten()
        historical_mean = np.mean(y_train_original)
        predictions['historical_mean'] = np.full(len(y_test), historical_mean)
        
        self.predictions = predictions
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Prediction generation completed in {elapsed_time:.2f}s")
        
        return predictions
    
    def evaluate_performance(self) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Returns:
            Dictionary with evaluation results
        """
        self.logger.info("Starting performance evaluation...")
        start_time = time.time()
        
        # Get true values in original scale
        y_test_original = self.scalers['y'].inverse_transform(
            self.data_splits['test'][1].reshape(-1, 1)
        ).flatten()
        
        # Create evaluator
        evaluator = FinancialEvaluator()
        
        # Comprehensive evaluation
        evaluation_df = evaluator.comprehensive_evaluation(y_test_original, self.predictions)
        
        # Statistical tests
        statistical_tests = evaluator.statistical_tests_summary(
            y_test_original, self.predictions, benchmark_model='historical_mean'
        )
        
        # Store results
        evaluation_results = {
            'metrics_summary': evaluation_df,
            'statistical_tests': statistical_tests,
            'data_summary': {
                'n_train': len(self.data_splits['train'][0]),
                'n_val': len(self.data_splits['val'][0]),
                'n_test': len(self.data_splits['test'][0]),
                'feature_count': len(self.data_splits['feature_names'])
            }
        }
        
        self.results = evaluation_results
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Performance evaluation completed in {elapsed_time:.2f}s")
        
        return evaluation_results
    
    def generate_report(self):
        """Generate comprehensive report with results and visualizations."""
        self.logger.info("Generating reports...")
        start_time = time.time()
        
        # Save evaluation metrics
        metrics_file = self.output_dir / 'evaluation_metrics.csv'
        self.results['metrics_summary'].to_csv(metrics_file, index=False)
        
        # Save statistical tests
        import json
        tests_file = self.output_dir / 'statistical_tests.json'
        with open(tests_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            serializable_tests = {}
            for model, tests in self.results['statistical_tests'].items():
                serializable_tests[model] = {
                    test_name: {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                              for k, v in test_result.items()}
                    for test_name, test_result in tests.items()
                }
            json.dump(serializable_tests, f, indent=2)
        
        # Save predictions
        pred_df = pd.DataFrame(self.predictions)
        pred_file = self.output_dir / 'predictions.csv'
        pred_df.to_csv(pred_file, index=False)
        
        # Generate summary report
        self._generate_summary_report()
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Report generation completed in {elapsed_time:.2f}s")
        self.logger.info(f"Results saved to: {self.output_dir}")
    
    def _generate_summary_report(self):
        """Generate a human-readable summary report."""
        report_file = self.output_dir / 'summary_report.txt'
        
        with open(report_file, 'w') as f:
            f.write("Neural Network Equity Premium Prediction - Summary Report\\n")
            f.write("=" * 60 + "\\n\\n")
            
            # Data summary
            f.write("DATA SUMMARY\\n")
            f.write("-" * 20 + "\\n")
            f.write(f"Training samples: {self.results['data_summary']['n_train']}\\n")
            f.write(f"Validation samples: {self.results['data_summary']['n_val']}\\n")
            f.write(f"Test samples: {self.results['data_summary']['n_test']}\\n")
            f.write(f"Features: {self.results['data_summary']['feature_count']}\\n\\n")
            
            # Model performance
            f.write("MODEL PERFORMANCE\\n")
            f.write("-" * 20 + "\\n")
            df = self.results['metrics_summary']
            best_model = df.iloc[0]  # Already sorted by overall rank
            
            f.write(f"Best performing model: {best_model['model']}\\n")
            f.write(f"RMSE: {best_model['rmse']:.6f}\\n")
            f.write(f"R-squared: {best_model['r_squared']:.6f}\\n")
            f.write(f"Sharpe Ratio: {best_model['sharpe_ratio']:.6f}\\n")
            f.write(f"Hit Rate: {best_model['hit_rate']:.1%}\\n\\n")
            
            # Statistical significance
            f.write("STATISTICAL SIGNIFICANCE\\n")
            f.write("-" * 25 + "\\n")
            for model, tests in self.results['statistical_tests'].items():
                if 'diebold_mariano' in tests:
                    dm_p = tests['diebold_mariano']['p_value']
                    significant = "✓" if dm_p < 0.05 else "✗"
                    f.write(f"{model}: p-value = {dm_p:.3f} {significant}\\n")
    
    def _save_models(self):
        """Save trained models to disk."""
        models_dir = self.output_dir / 'models'
        models_dir.mkdir(exist_ok=True)
        
        for model_name, model_info in self.models.items():
            model_file = models_dir / f"{model_name}.pth"
            torch.save({
                'model_state_dict': model_info['model'].state_dict(),
                'best_params': model_info['best_params'],
                'best_score': model_info['best_score']
            }, model_file)
        
        # Save scalers
        import joblib
        scaler_file = models_dir / 'scalers.pkl'
        joblib.dump(self.scalers, scaler_file)
    
    def run_complete_pipeline(self):
        """Execute the complete pipeline from start to finish."""
        self.logger.info("Starting complete pipeline execution...")
        pipeline_start = time.time()
        
        try:
            # Stage 1: Data preparation
            self.load_and_preprocess_data()
            
            # Stage 2: Model training
            self.train_models()
            
            # Stage 3: Prediction generation
            self.generate_predictions()
            
            # Stage 4: Performance evaluation
            self.evaluate_performance()
            
            # Stage 5: Report generation
            self.generate_report()
            
            pipeline_elapsed = time.time() - pipeline_start
            self.logger.info(f"Complete pipeline executed successfully in {pipeline_elapsed:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed with error: {str(e)}")
            raise

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_default_config() -> Dict[str, Any]:
    """Get default configuration for quick runs."""
    return {
        'data': {
            'file': 'data/ml_equity_premium_data.xlsx',
            'target_column': 'equity_premium',
            'train_ratio': 0.7,
            'val_ratio': 0.15
        },
        'models': {
            'architectures': ['Net3', 'DNet2'],
            'optimization': {
                'method': 'bayesian',
                'n_trials': 50,  # Reduced for quick run
                'timeout': 1800  # 30 minutes
            }
        },
        'training': {
            'max_epochs': 200,
            'early_stopping_patience': 20,
            'device': 'auto'
        },
        'evaluation': {
            'metrics': ['rmse', 'r2', 'sharpe_ratio', 'hit_rate'],
            'significance_level': 0.05,
            'bootstrap_samples': 1000
        },
        'output': {
            'directory': 'results/',
            'save_models': True,
            'generate_plots': False  # Disabled for quick run
        },
        'random_seed': 42
    }

def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(description='Neural Network Equity Premium Prediction Pipeline')
    parser.add_argument('--config', type=str, help='Path to configuration YAML file')
    parser.add_argument('--quick-run', action='store_true', help='Run with default configuration')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.quick_run:
        config = get_default_config()
        print("Running with default configuration...")
    elif args.config:
        config = load_config(args.config)
        print(f"Loaded configuration from: {args.config}")
    else:
        print("Error: Must specify either --config or --quick-run")
        sys.exit(1)
    
    # Create and run pipeline
    pipeline = EquityPremiumPipeline(config)
    pipeline.run_complete_pipeline()
    
    print("\\nPipeline completed successfully!")
    print(f"Results saved to: {pipeline.output_dir}")

if __name__ == "__main__":
    main()`}
          />
        </motion.section>

        {/* Navigation */}
        <NavigationButtons 
          prevHref="/evaluation"
          prevLabel="Evaluation"
          nextHref="/interactive-architecture"
          nextLabel="Interactive Architecture"
        />
      </div>
    </div>
  )
}