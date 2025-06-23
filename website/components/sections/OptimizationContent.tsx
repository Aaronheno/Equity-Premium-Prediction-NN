'use client'

import { motion } from 'framer-motion'
import { useState } from 'react'
import { TrendingUp, Settings, Zap, Target, BarChart3, Play, CheckCircle } from 'lucide-react'
import CodeBlock from '@/components/shared/CodeBlock'
import MathFormula from '@/components/shared/MathFormula'
import NavigationButtons from '@/components/shared/NavigationButtons'

const optimizers = [
  {
    name: 'Adam',
    description: 'Adaptive moment estimation with bias correction',
    formula: 'm_t = β₁m_{t-1} + (1-β₁)g_t, v_t = β₂v_{t-1} + (1-β₂)g_t²',
    advantages: ['Adaptive learning rates', 'Works well with sparse gradients', 'Bias correction'],
    hyperparams: { lr: '1e-3 to 1e-4', beta1: '0.9', beta2: '0.999', eps: '1e-8' },
    useCase: 'Default choice for most financial neural networks',
    icon: Zap,
    color: 'accent-blue'
  },
  {
    name: 'RMSprop', 
    description: 'Root mean square propagation with adaptive learning rates',
    formula: 'v_t = αv_{t-1} + (1-α)g_t², θ_{t+1} = θ_t - η/√(v_t + ε) · g_t',
    advantages: ['Handles non-stationary objectives', 'Good for RNNs', 'Simple implementation'],
    hyperparams: { lr: '1e-3 to 1e-2', alpha: '0.99', eps: '1e-8', momentum: '0.0' },
    useCase: 'Time series with changing patterns',
    icon: BarChart3,
    color: 'accent-orange'
  },
  {
    name: 'SGD',
    description: 'Stochastic gradient descent with optional momentum',
    formula: 'v_t = μv_{t-1} + g_t, θ_{t+1} = θ_t - η · v_t',
    advantages: ['Simple and robust', 'Well understood theory', 'Good final convergence'],
    hyperparams: { lr: '1e-2 to 1e-1', momentum: '0.9', dampening: '0.0', nesterov: 'false' },
    useCase: 'When simplicity and interpretability matter',
    icon: Target,
    color: 'accent-purple'
  }
]

const learningRateStrategies = [
  {
    name: 'HPO Static Rates',
    description: 'Bayesian optimization finds optimal fixed learning rates',
    schedule: 'lr = optimal_static_rate (via HPO)',
    params: { search_range: '1e-5 to 1e-2', distribution: 'log-uniform' },
    useCase: 'Automated optimal rate discovery'
  },
  {
    name: 'Early Stopping',
    description: 'Monitor validation loss and stop when no improvement',
    schedule: 'stop_training (when val_loss plateaus)',
    params: { patience: '10 epochs', monitor: 'val_loss' },
    useCase: 'Prevents overfitting naturally'
  },
  {
    name: 'Multi-Optimizer',
    description: 'Test Adam, RMSprop, SGD to find best combination',
    schedule: 'optimizer_choice (via HPO)',
    params: { choices: 'Adam, RMSprop, SGD', selection: 'automatic' },
    useCase: 'Comprehensive optimizer evaluation'
  }
]

const optimizationChallenges = [
  {
    challenge: 'Vanishing Gradients',
    description: 'Gradients become extremely small in deep networks',
    symptoms: ['Training stagnation', 'Lower layers learn slowly', 'Activation saturation'],
    solutions: ['ReLU activations', 'Residual connections', 'Gradient clipping', 'Batch normalization'],
    financial_context: 'Common in deep networks processing long financial time series'
  },
  {
    challenge: 'Exploding Gradients', 
    description: 'Gradients grow exponentially large during backpropagation',
    symptoms: ['Loss oscillation', 'NaN values', 'Unstable training'],
    solutions: ['Gradient clipping', 'Lower learning rates', 'Better initialization', 'Regularization'],
    financial_context: 'Triggered by volatile market periods or outlier events'
  },
  {
    challenge: 'Local Minima',
    description: 'Optimizer gets stuck in suboptimal solutions',
    symptoms: ['Training plateaus', 'Poor validation performance', 'High variance'],
    solutions: ['Multiple random initializations', 'Learning rate scheduling', 'Momentum methods'],
    financial_context: 'Financial data has many spurious patterns that can trap models'
  },
  {
    challenge: 'Overfitting',
    description: 'Model memorizes training data instead of learning patterns',
    symptoms: ['Large train-validation gap', 'Poor generalization', 'High model complexity'],
    solutions: ['Dropout', 'L1/L2 regularization', 'Early stopping', 'Data augmentation'],
    financial_context: 'Markets change over time, making historical patterns unreliable'
  }
]

export default function OptimizationContent() {
  const [activeOptimizer, setActiveOptimizer] = useState(0)
  const [selectedChallenge, setSelectedChallenge] = useState(0)

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
            <span className="gradient-text">Optimization</span>
          </h1>
          <p className="max-w-4xl mx-auto text-xl text-text-secondary leading-relaxed">
            Advanced optimization algorithms and techniques: Adam, RMSprop, SGD with hyperparameter-optimized 
            static learning rates, early stopping, and solving common training challenges in financial neural networks.
          </p>
        </motion.div>

        {/* Optimization Overview */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <TrendingUp className="w-8 h-8 text-accent-blue mr-3" />
            The Optimization Process
          </h2>
          
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8 mb-8">
            <p className="text-text-secondary text-lg mb-6 leading-relaxed">
              Optimization is the process of iteratively updating neural network weights to minimize 
              the loss function. Modern optimizers use sophisticated algorithms that adapt learning 
              rates and incorporate momentum to navigate the complex loss landscape efficiently.
            </p>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div>
                <h3 className="text-xl font-semibold text-text-primary mb-4">Core Concepts:</h3>
                <div className="space-y-3 text-sm">
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-blue mt-2"></div>
                    <span className="text-text-secondary"><strong>Gradient Descent:</strong> Move weights opposite to gradient direction</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-blue mt-2"></div>
                    <span className="text-text-secondary"><strong>Learning Rate:</strong> Controls step size of weight updates</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-blue mt-2"></div>
                    <span className="text-text-secondary"><strong>Momentum:</strong> Accumulates gradients for smoother convergence</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-blue mt-2"></div>
                    <span className="text-text-secondary"><strong>Adaptive Rates:</strong> Adjust learning rates per parameter</span>
                  </div>
                </div>
              </div>
              
              <div>
                <h3 className="text-xl font-semibold text-text-primary mb-4">Update Rule:</h3>
                <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 space-y-3 text-sm">
                  <div>
                    <div className="text-text-muted mb-1">Basic Gradient Descent:</div>
                    <MathFormula latex="θ_{t+1} = θ_t - η \nabla L(θ_t)" />
                  </div>
                  <div>
                    <div className="text-text-muted mb-1">With Momentum:</div>
                    <MathFormula latex="v_t = μv_{t-1} + η \nabla L(θ_t)" />
                  </div>
                  <div>
                    <MathFormula latex="θ_{t+1} = θ_t - v_t" />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </motion.section>

        {/* Optimizer Comparison */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Settings className="w-8 h-8 text-accent-orange mr-3" />
            Optimizer Algorithms
          </h2>
          
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8 mb-8">
            <p className="text-text-secondary text-lg mb-8">
              Each optimizer has unique characteristics suited for different financial modeling scenarios. 
              Click each optimizer to explore its mathematical formulation and practical considerations.
            </p>

            {/* Optimizer Selector */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
              {optimizers.map((optimizer, index) => {
                const Icon = optimizer.icon
                const isActive = activeOptimizer === index
                return (
                  <button
                    key={optimizer.name}
                    onClick={() => setActiveOptimizer(index)}
                    className={`p-6 rounded-lg border transition-all text-left ${
                      isActive 
                        ? `bg-${optimizer.color}/10 border-${optimizer.color}/30` 
                        : 'bg-bg-primary border-bg-tertiary hover:border-accent-blue/30'
                    }`}
                  >
                    <div className="flex items-center space-x-3 mb-4">
                      <div className={`inline-flex items-center justify-center w-10 h-10 rounded-lg bg-${optimizer.color}/10 border border-${optimizer.color}/20`}>
                        <Icon className={`w-5 h-5 text-${optimizer.color}`} />
                      </div>
                      <div>
                        <h3 className="font-bold text-text-primary">{optimizer.name}</h3>
                        <p className="text-text-muted text-xs">{optimizer.useCase}</p>
                      </div>
                    </div>
                    <p className="text-text-secondary text-sm">{optimizer.description}</p>
                  </button>
                )
              })}
            </div>

            {/* Active Optimizer Details */}
            {optimizers.map((optimizer, index) => {
              const Icon = optimizer.icon
              if (activeOptimizer !== index) return null
              
              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                  className={`bg-${optimizer.color}/5 border border-${optimizer.color}/20 rounded-xl p-8`}
                >
                  <div className="flex items-center space-x-3 mb-6">
                    <div className={`inline-flex items-center justify-center w-12 h-12 rounded-lg bg-${optimizer.color}/10 border border-${optimizer.color}/20`}>
                      <Icon className={`w-6 h-6 text-${optimizer.color}`} />
                    </div>
                    <div>
                      <h3 className="text-2xl font-bold text-text-primary">{optimizer.name} Optimizer</h3>
                      <p className="text-text-secondary">{optimizer.description}</p>
                    </div>
                  </div>

                  <div className="mb-6">
                    <h4 className="font-semibold text-text-primary mb-3">Update Equations:</h4>
                    <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4">
                      <MathFormula latex={optimizer.formula} block={true} />
                    </div>
                  </div>

                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    <div>
                      <h4 className="font-semibold text-text-primary mb-3">Key Advantages:</h4>
                      <div className="space-y-2 text-sm">
                        {optimizer.advantages.map((advantage, idx) => (
                          <div key={idx} className="flex items-start space-x-3">
                            <div className={`w-2 h-2 rounded-full bg-${optimizer.color} mt-2`}></div>
                            <span className="text-text-secondary">{advantage}</span>
                          </div>
                        ))}
                      </div>
                      
                      <div className={`mt-4 bg-${optimizer.color}/10 border border-${optimizer.color}/20 rounded-lg p-4`}>
                        <div className={`text-${optimizer.color} text-sm font-medium mb-1`}>Financial Use Case:</div>
                        <div className="text-text-secondary text-sm">{optimizer.useCase}</div>
                      </div>
                    </div>
                    
                    <div>
                      <h4 className="font-semibold text-text-primary mb-3">Hyperparameters:</h4>
                      <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 space-y-2 text-sm">
                        {Object.entries(optimizer.hyperparams).map(([param, value]) => (
                          <div key={param} className="flex justify-between">
                            <span className="font-mono text-text-primary">{param}:</span>
                            <span className={`font-mono text-${optimizer.color}`}>{value}</span>
                          </div>
                        ))}
                      </div>
                      
                      {optimizer.name === 'Adam' && (
                        <div className="mt-4 text-xs text-text-muted">
                          <div><strong>β₁:</strong> Exponential decay rate for first moment estimates</div>
                          <div><strong>β₂:</strong> Exponential decay rate for second moment estimates</div>
                          <div><strong>ε:</strong> Small constant for numerical stability</div>
                        </div>
                      )}
                    </div>
                  </div>
                </motion.div>
              )
            })}
          </div>
        </motion.section>

        {/* Learning Rate Optimization */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <BarChart3 className="w-8 h-8 text-accent-green mr-3" />
            Learning Rate Optimization Strategy
          </h2>
          
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8 mb-8">
            <p className="text-text-secondary text-lg mb-8">
              Our approach uses <strong>static learning rates determined through hyperparameter optimization</strong> 
              rather than dynamic scheduling. This strategy combines automated rate discovery with early stopping 
              for robust, efficient training.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {learningRateStrategies.map((strategy, index) => (
                <div key={strategy.name} className="bg-bg-primary border border-bg-tertiary rounded-lg p-6">
                  <h3 className="font-semibold text-text-primary mb-3">{strategy.name}</h3>
                  <p className="text-text-secondary text-sm mb-4">{strategy.description}</p>
                  
                  <div className="mb-4">
                    <div className="text-text-muted text-xs mb-2">Implementation:</div>
                    <code className="text-accent-green text-xs font-mono">{strategy.schedule}</code>
                  </div>
                  
                  <div className="space-y-2 text-xs">
                    {Object.entries(strategy.params).map(([param, value]) => (
                      <div key={param} className="flex justify-between">
                        <span className="text-text-muted">{param}:</span>
                        <span className="font-mono text-accent-green">{value}</span>
                      </div>
                    ))}
                  </div>
                  
                  <div className="mt-4 pt-3 border-t border-bg-tertiary">
                    <div className="text-text-muted text-xs mb-1">Best For:</div>
                    <div className="text-text-secondary text-xs">{strategy.useCase}</div>
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-8 bg-accent-green/5 border border-accent-green/20 rounded-lg p-6">
              <h3 className="text-xl font-semibold text-accent-green mb-4">Our Actual Implementation Strategy</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm">
                <div>
                  <h4 className="font-semibold text-text-primary mb-2">HPO Search Process:</h4>
                  <div className="space-y-1 text-text-secondary">
                    <div><strong>Search Range:</strong> 1e-5 to 1e-2 (log-uniform)</div>
                    <div><strong>Method:</strong> Bayesian optimization (Optuna)</div>
                    <div><strong>Result:</strong> Optimal static rate per model</div>
                    <div><strong>Example:</strong> Net2 → lr=0.00341 (via HPO)</div>
                  </div>
                </div>
                <div>
                  <h4 className="font-semibold text-text-primary mb-2">Training Control:</h4>
                  <div className="space-y-1 text-text-secondary">
                    <div>• Early stopping with 10-epoch patience</div>
                    <div>• Validation loss monitoring</div>
                    <div>• Automatic best weight restoration</div>
                    <div>• No manual scheduling overhead</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </motion.section>

        {/* Optimization Challenges */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.5 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Target className="w-8 h-8 text-accent-purple mr-3" />
            Common Optimization Challenges
          </h2>
          
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8 mb-8">
            <p className="text-text-secondary text-lg mb-8">
              Financial neural networks face unique optimization challenges due to noisy data, 
              non-stationary patterns, and complex market dynamics. Understanding these challenges 
              helps in designing robust training procedures.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
              {optimizationChallenges.map((challenge, index) => (
                <button
                  key={challenge.challenge}
                  onClick={() => setSelectedChallenge(index)}
                  className={`p-4 rounded-lg border transition-all text-left ${
                    selectedChallenge === index
                      ? 'bg-accent-purple/10 border-accent-purple/30'
                      : 'bg-bg-primary border-bg-tertiary hover:border-accent-blue/30'
                  }`}
                >
                  <h3 className="font-semibold text-text-primary mb-2">{challenge.challenge}</h3>
                  <p className="text-text-secondary text-sm">{challenge.description}</p>
                </button>
              ))}
            </div>

            {/* Selected Challenge Details */}
            <motion.div
              key={selectedChallenge}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              className="bg-accent-purple/5 border border-accent-purple/20 rounded-xl p-8"
            >
              <h3 className="text-2xl font-bold text-text-primary mb-6">{optimizationChallenges[selectedChallenge].challenge}</h3>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div>
                  <h4 className="font-semibold text-text-primary mb-3">Symptoms:</h4>
                  <div className="space-y-2 text-sm mb-6">
                    {optimizationChallenges[selectedChallenge].symptoms.map((symptom, idx) => (
                      <div key={idx} className="flex items-start space-x-3">
                        <div className="w-2 h-2 rounded-full bg-accent-red mt-2"></div>
                        <span className="text-text-secondary">{symptom}</span>
                      </div>
                    ))}
                  </div>
                  
                  <h4 className="font-semibold text-text-primary mb-3">Solutions:</h4>
                  <div className="space-y-2 text-sm">
                    {optimizationChallenges[selectedChallenge].solutions.map((solution, idx) => (
                      <div key={idx} className="flex items-start space-x-3">
                        <div className="w-2 h-2 rounded-full bg-accent-green mt-2"></div>
                        <span className="text-text-secondary">{solution}</span>
                      </div>
                    ))}
                  </div>
                </div>
                
                <div>
                  <h4 className="font-semibold text-text-primary mb-3">Financial Context:</h4>
                  <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 mb-4">
                    <p className="text-text-secondary text-sm">{optimizationChallenges[selectedChallenge].financial_context}</p>
                  </div>
                  
                  <div className="bg-accent-purple/10 border border-accent-purple/20 rounded-lg p-4">
                    <h5 className="font-semibold text-accent-purple mb-2">Prevention Strategy:</h5>
                    <div className="text-text-secondary text-sm space-y-1">
                      {selectedChallenge === 0 && (
                        <>
                          <div>• Use ReLU activations instead of sigmoid/tanh</div>
                          <div>• Implement residual connections for very deep networks</div>
                          <div>• Apply gradient clipping with threshold 1.0-5.0</div>
                          <div>• Monitor gradient norms during training</div>
                        </>
                      )}
                      {selectedChallenge === 1 && (
                        <>
                          <div>• Clip gradients to maximum norm of 1.0</div>
                          <div>• Reduce learning rate if loss becomes unstable</div>
                          <div>• Use better weight initialization (Xavier/He)</div>
                          <div>• Add batch normalization between layers</div>
                        </>
                      )}
                      {selectedChallenge === 2 && (
                        <>
                          <div>• Train multiple models with different initializations</div>
                          <div>• Use learning rate scheduling to escape plateaus</div>
                          <div>• Apply momentum-based optimizers (Adam, RMSprop)</div>
                          <div>• Consider ensemble methods for robustness</div>
                        </>
                      )}
                      {selectedChallenge === 3 && (
                        <>
                          <div>• Implement dropout (0.2-0.5 for financial data)</div>
                          <div>• Use L1/L2 regularization with λ = 1e-4 to 1e-6</div>
                          <div>• Apply early stopping based on validation loss</div>
                          <div>• Validate on out-of-sample time periods</div>
                        </>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        </motion.section>

        {/* Implementation Example */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Play className="w-8 h-8 text-accent-blue mr-3" />
            Optimization Implementation
          </h2>
          
          <p className="text-text-secondary text-lg mb-6">
            Complete implementation showing our actual approach: hyperparameter optimization for 
            static learning rates, early stopping, and multi-optimizer evaluation for robust 
            training of financial neural networks.
          </p>

          <CodeBlock
            language="python"
            title="Our Actual Optimization Implementation - Static LR + Early Stopping"
            code={`import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
from src.utils.distributions import FloatDistribution, CategoricalDistribution

class ActualOptimizationManager:
    """
    Our actual optimization approach: HPO for static learning rates 
    with early stopping - no dynamic scheduling.
    """
    
    def __init__(self, model, trial=None):
        self.model = model
        self.trial = trial
        
        # HPO determines these static values
        if trial:
            self.lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            self.optimizer_name = trial.suggest_categorical(
                'optimizer_choice', ['Adam', 'RMSprop', 'SGD']
            )
            self.weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-2, log=True)
        else:
            # Use previously optimized values
            self.lr = 0.00341  # Example: found via HPO
            self.optimizer_name = 'Adam'
            self.weight_decay = 1e-4
        
        # Create optimizer with STATIC learning rate
        self.optimizer = self._create_optimizer()
        
        # NO SCHEDULER - we use static rates!
        # Instead: early stopping for adaptive training control
        
        # Early stopping configuration
        self.early_stopping_patience = 10
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [], 'epochs_trained': 0
        }
        
    def _create_optimizer(self):
        """Create optimizer based on HPO selection - STATIC learning rate."""
        if self.optimizer_name.lower() == 'adam':
            return optim.Adam(
                self.model.parameters(), 
                lr=self.lr,  # STATIC rate from HPO
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name.lower() == 'rmsprop':
            return optim.RMSprop(
                self.model.parameters(),
                lr=self.lr,  # STATIC rate from HPO
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name.lower() == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.lr,  # STATIC rate from HPO
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
    
    def training_step(self, batch_x, batch_y, loss_fn):
        """
        Execute single training step - no scheduling, just static LR.
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        predictions = self.model(batch_x)
        loss = loss_fn(predictions, batch_y)
        
        # Backward pass
        loss.backward()
        
        # Optimizer step (learning rate stays CONSTANT)
        self.optimizer.step()
        
        return loss.item()
    
    def validation_step(self, val_x, val_y, loss_fn):
        """Validation step with early stopping - our adaptive mechanism."""
        self.model.eval()
        
        with torch.no_grad():
            val_predictions = self.model(val_x)
            val_loss = loss_fn(val_predictions, val_y).item()
        
        # NO LEARNING RATE SCHEDULING!
        # Instead: early stopping for adaptive training control
        
        # Early stopping check
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            # Save best model state
            self.best_model_state = self.model.state_dict().copy()
        else:
            self.patience_counter += 1
        
        # Log validation loss
        self.history['val_loss'].append(val_loss)
        
        # Return whether to stop training
        should_stop = self.patience_counter >= self.early_stopping_patience
        return val_loss, should_stop
    
    def restore_best_model(self):
        """Restore best model weights when training completes."""
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
def hpo_optimization_example():
    """
    Demonstrate our actual HPO approach for learning rate optimization.
    """
    from src.models.nns import Net2
    
    def objective(trial):
        """Optuna objective function - this is our actual approach."""
        
        # Create model
        model = Net2(
            n_feature=32,  # Our actual number of features
            n_hidden1=trial.suggest_int('n_hidden1', 16, 192),
            n_hidden2=trial.suggest_int('n_hidden2', 8, 128),
            n_output=1,
            dropout=trial.suggest_float('dropout', 0.0, 0.6)
        )
        
        # Create optimization manager with HPO
        opt_manager = ActualOptimizationManager(model, trial)
        
        # Simulated financial data (32 features)
        train_x = torch.randn(1000, 32)
        train_y = torch.randn(1000, 1) * 0.02 + 0.08  # ~8% equity premium
        val_x = torch.randn(200, 32)
        val_y = torch.randn(200, 1) * 0.02 + 0.08
        
        # Loss function
        loss_fn = nn.MSELoss()
        
        # Training with STATIC learning rate
        max_epochs = 75  # Our typical epoch count
        batch_size = 128
        
        for epoch in range(max_epochs):
            # Training phase
            epoch_losses = []
            for i in range(0, len(train_x), batch_size):
                batch_x = train_x[i:i+batch_size]
                batch_y = train_y[i:i+batch_size]
                
                # Training step with STATIC LR
                loss = opt_manager.training_step(batch_x, batch_y, loss_fn)
                epoch_losses.append(loss)
            
            avg_train_loss = np.mean(epoch_losses)
            opt_manager.history['train_loss'].append(avg_train_loss)
            
            # Validation with early stopping
            val_loss, should_stop = opt_manager.validation_step(val_x, val_y, loss_fn)
            
            # Early stopping - our adaptive mechanism
            if should_stop:
                opt_manager.restore_best_model()
                break
        
        # Return final validation loss for HPO
        return opt_manager.best_val_loss
    
    # Run Bayesian optimization - this finds optimal static LR
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    
    print("HPO Results - Optimal Static Learning Rate Configuration:")
    print(f"Best validation loss: {study.best_value:.6f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Demonstrate final training with optimal static rate
    print("\\nTraining final model with optimal static configuration...")
    
    # Create final model with best parameters
    best_model = Net2(
        n_feature=32,
        n_hidden1=study.best_params['n_hidden1'],
        n_hidden2=study.best_params['n_hidden2'], 
        n_output=1,
        dropout=study.best_params['dropout']
    )
    
    # Use optimal static learning rate (no scheduling!)
    optimal_lr = study.best_params['learning_rate']
    optimal_optimizer = study.best_params['optimizer_choice']
    
    print(f"Optimal static learning rate: {optimal_lr:.2e}")
    print(f"Optimal optimizer: {optimal_optimizer}")
    print(f"Learning rate will remain CONSTANT at {optimal_lr:.2e} throughout training")
    
    return study.best_params

def compare_static_vs_scheduled():
    """
    Compare our static LR approach vs traditional scheduling.
    """
    print("COMPARISON: Static LR (Our Approach) vs Scheduled LR")
    print("="*60)
    
    # Our approach: HPO finds optimal static rate
    print("\\n1. OUR APPROACH - Static LR + Early Stopping:")
    print("   ✅ HPO finds optimal rate: lr=0.00341")
    print("   ✅ Rate stays constant throughout training")
    print("   ✅ Early stopping prevents overfitting")
    print("   ✅ No scheduler overhead")
    print("   ✅ Automated via Optuna")
    
    # Traditional approach: Manual scheduling
    print("\\n2. TRADITIONAL APPROACH - LR Scheduling:")
    print("   ❌ Manual schedule design required")
    print("   ❌ More hyperparameters to tune")
    print("   ❌ Scheduler computational overhead")
    print("   ❌ Risk of premature rate reduction")
    print("   ❌ Not implemented in our codebase")
    
    print("\\n3. WHY OUR APPROACH WORKS:")
    print("   • HPO explores learning rate space automatically")
    print("   • Static rates avoid premature convergence")
    print("   • Early stopping provides natural adaptation")
    print("   • Simpler = fewer things that can go wrong")
    print("   • Proven effective in financial time series")

# Demonstration of our actual implementation
if __name__ == "__main__":
    print("Our Actual Optimization Strategy Demo")
    print("="*50)
    
    # Show HPO approach
    best_config = hpo_optimization_example()
    
    print("\\n" + "="*50)
    
    # Compare approaches
    compare_static_vs_scheduled()`

}
          />
        </motion.section>

        {/* Navigation */}
        <NavigationButtons 
          prevHref="/backpropagation"
          prevLabel="Backpropagation"
          nextHref="/hyperparameter-optimization"
          nextLabel="Hyperparameter Optimization"
        />
      </div>
    </div>
  )
}