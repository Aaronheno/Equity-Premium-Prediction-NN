'use client'

import { motion } from 'framer-motion'
import { useState } from 'react'
import { Play, Zap, Target, RefreshCw, TrendingUp, Settings, Code2, Clock, Brain, ChevronRight } from 'lucide-react'
import CodeBlock from '@/components/shared/CodeBlock'
import MathFormula from '@/components/shared/MathFormula'
import NavigationButtons from '@/components/shared/NavigationButtons'

const trainingSteps = [
  {
    step: 1,
    title: 'Forward Pass',
    icon: Play,
    color: 'accent-blue',
    description: 'Data flows through network layers to produce predictions',
    operations: ['Linear transformation (Wx + b)', 'ReLU activation', 'Dropout regularization'],
    purpose: 'Transform 30 financial indicators into equity premium prediction'
  },
  {
    step: 2,
    title: 'Loss Calculation',
    icon: Target,
    color: 'accent-orange',
    description: 'Compare predictions with actual returns to measure error',
    operations: ['Mean Squared Error', 'L1 regularization', 'L2 regularization'],
    purpose: 'Quantify prediction accuracy and add penalty for model complexity'
  },
  {
    step: 3,
    title: 'Backpropagation',
    icon: RefreshCw,
    color: 'accent-red',
    description: 'Calculate gradients to determine weight update directions',
    operations: ['Chain rule application', 'Gradient computation', 'Error propagation'],
    purpose: 'Determine how to adjust each weight to reduce prediction error'
  },
  {
    step: 4,
    title: 'Weight Update',
    icon: TrendingUp,
    color: 'accent-green',
    description: 'Optimizer adjusts weights based on computed gradients',
    operations: ['Adam optimizer', 'Learning rate scaling', 'Momentum application'],
    purpose: 'Update network parameters to improve future predictions'
  }
]

const activationFunctions = [
  {
    name: 'ReLU',
    formula: 'max(0, x)',
    description: 'Sets negative values to zero, preserves positive values',
    benefits: ['Computational efficiency', 'Prevents vanishing gradients', 'Creates sparse representations'],
    usage: 'Primary activation for all hidden layers'
  },
  {
    name: 'Linear',
    formula: 'f(x) = x',
    description: 'No transformation applied to the output',
    benefits: ['Allows full range output', 'Suitable for regression', 'No saturation issues'],
    usage: 'Output layer for continuous predictions'
  }
]

const epochProgress = [
  { epoch: 1, loss: 0.0125, r2: -0.12, status: 'Random initialization - no patterns learned' },
  { epoch: 5, loss: 0.0089, r2: 0.03, status: 'Basic trends emerging' },
  { epoch: 10, loss: 0.0067, r2: 0.15, status: 'Momentum and volatility patterns recognized' },
  { epoch: 25, loss: 0.0051, r2: 0.28, status: 'Valuation ratios relationships learned' },
  { epoch: 50, loss: 0.0042, r2: 0.34, status: 'Complex interactions between indicators' },
  { epoch: 75, loss: 0.0041, r2: 0.35, status: 'Fine-tuning correlations' },
  { epoch: 100, loss: 0.0040, r2: 0.36, status: 'Optimal performance achieved' }
]

export default function TrainingContent() {
  const [activeStep, setActiveStep] = useState(1)
  const [showEpochDetails, setShowEpochDetails] = useState(false)

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
            <span className="gradient-text">Training Process</span>
          </h1>
          <p className="max-w-4xl mx-auto text-xl text-text-secondary leading-relaxed">
            Complete neural network training workflow: forward pass mechanics, backpropagation, 
            optimization algorithms, and epoch management for financial prediction models.
          </p>
        </motion.div>

        {/* Training Overview */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Brain className="w-8 h-8 text-accent-blue mr-3" />
            Training Process Overview
          </h2>
          
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8 mb-8">
            <p className="text-text-secondary text-lg mb-8 leading-relaxed">
              Neural network training is an iterative process that adjusts network weights to minimize 
              prediction errors on financial data. Each training cycle consists of four fundamental steps 
              that work together to gradually improve model performance.
            </p>

            <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-4 gap-6">
              {trainingSteps.map((step, index) => {
                const Icon = step.icon
                const isActive = activeStep === step.step
                return (
                  <motion.div
                    key={step.step}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6, delay: 0.3 + index * 0.1 }}
                    onClick={() => setActiveStep(step.step)}
                    className={`cursor-pointer p-6 rounded-xl border transition-all ${
                      isActive 
                        ? `bg-${step.color}/10 border-${step.color}/30` 
                        : 'bg-bg-primary border-bg-tertiary hover:border-accent-blue/30'
                    }`}
                  >
                    <div className="flex items-center space-x-3 mb-4">
                      <div className={`inline-flex items-center justify-center w-10 h-10 rounded-lg bg-${step.color}/10 border border-${step.color}/20`}>
                        <Icon className={`w-5 h-5 text-${step.color}`} />
                      </div>
                      <div>
                        <div className="text-xs text-text-muted">Step {step.step}</div>
                        <h3 className="font-semibold text-text-primary">{step.title}</h3>
                      </div>
                    </div>
                    <p className="text-text-secondary text-sm mb-4">{step.description}</p>
                    <div className={`bg-${step.color}/5 border border-${step.color}/20 rounded p-3`}>
                      <div className={`text-${step.color} text-xs font-medium mb-1`}>Purpose:</div>
                      <div className="text-text-secondary text-xs">{step.purpose}</div>
                    </div>
                  </motion.div>
                )
              })}
            </div>
          </div>
        </motion.section>

        {/* Forward Pass Details */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Play className="w-8 h-8 text-accent-blue mr-3" />
            Forward Pass: Data Flow Through Layers
          </h2>
          
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8 mb-8">
            <p className="text-text-secondary text-lg mb-8">
              The forward pass transforms 30 financial indicators through multiple layers to produce 
              an equity premium prediction. Each layer performs three sequential operations.
            </p>

            <div className="space-y-8">
              {/* Linear Transformation */}
              <div className="bg-bg-primary border border-bg-tertiary rounded-xl p-6">
                <div className="flex items-center space-x-3 mb-4">
                  <div className="inline-flex items-center justify-center w-8 h-8 rounded-lg bg-accent-blue/10 border border-accent-blue/20">
                    <span className="text-accent-blue font-bold">1</span>
                  </div>
                  <h3 className="text-xl font-semibold text-text-primary">Linear Transformation</h3>
                </div>

                <div className="mb-6">
                  <MathFormula latex="h = Wx + b" block={true} />
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  <div>
                    <h4 className="font-semibold text-text-primary mb-3">Components:</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex items-start space-x-3">
                        <MathFormula latex="W" />
                        <span className="text-text-secondary">Weight matrix determining input influence</span>
                      </div>
                      <div className="flex items-start space-x-3">
                        <MathFormula latex="x" />
                        <span className="text-text-secondary">Input features (30 financial indicators)</span>
                      </div>
                      <div className="flex items-start space-x-3">
                        <MathFormula latex="b" />
                        <span className="text-text-secondary">Bias vector providing learned offset</span>
                      </div>
                      <div className="flex items-start space-x-3">
                        <MathFormula latex="h" />
                        <span className="text-text-secondary">Output transformation</span>
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="font-semibold text-text-primary mb-3">Example Calculation:</h4>
                    <div className="bg-bg-secondary border border-bg-tertiary rounded-lg p-4 space-y-2 text-sm font-mono">
                      <div><strong>Input:</strong> [DY=0.02, TMS=0.01, INFL=0.03]</div>
                      <div><strong>Weights:</strong> [0.5, 0.3, -0.2]</div>
                      <div><strong>Bias:</strong> 0.1</div>
                      <div className="border-t border-bg-tertiary pt-2">
                        <strong>h =</strong> (0.02×0.5) + (0.01×0.3) + (0.03×-0.2) + 0.1
                      </div>
                      <div className="text-accent-blue">
                        <strong>Result:</strong> h = 0.097
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* ReLU Activation */}
              <div className="bg-bg-primary border border-bg-tertiary rounded-xl p-6">
                <div className="flex items-center space-x-3 mb-4">
                  <div className="inline-flex items-center justify-center w-8 h-8 rounded-lg bg-accent-orange/10 border border-accent-orange/20">
                    <span className="text-accent-orange font-bold">2</span>
                  </div>
                  <h3 className="text-xl font-semibold text-text-primary">ReLU Activation</h3>
                </div>

                <div className="mb-6">
                  <MathFormula latex="\\text{output} = \\max(0, \\text{input})" block={true} />
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  <div>
                    <h4 className="font-semibold text-text-primary mb-3">Why ReLU is Essential:</h4>
                    <div className="space-y-3 text-sm">
                      <div className="flex items-start space-x-3">
                        <div className="w-2 h-2 rounded-full bg-accent-orange mt-2"></div>
                        <span className="text-text-secondary"><strong>Non-linearity:</strong> Enables learning complex patterns</span>
                      </div>
                      <div className="flex items-start space-x-3">
                        <div className="w-2 h-2 rounded-full bg-accent-orange mt-2"></div>
                        <span className="text-text-secondary"><strong>Sparsity:</strong> Creates sparse representations</span>
                      </div>
                      <div className="flex items-start space-x-3">
                        <div className="w-2 h-2 rounded-full bg-accent-orange mt-2"></div>
                        <span className="text-text-secondary"><strong>Efficiency:</strong> Simple max operation</span>
                      </div>
                      <div className="flex items-start space-x-3">
                        <div className="w-2 h-2 rounded-full bg-accent-orange mt-2"></div>
                        <span className="text-text-secondary"><strong>Gradients:</strong> Prevents vanishing gradient problem</span>
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="font-semibold text-text-primary mb-3">Financial Interpretation:</h4>
                    <div className="space-y-3 text-sm">
                      <div className="flex items-start space-x-3">
                        <div className="w-2 h-2 rounded-full bg-accent-green mt-2"></div>
                        <span className="text-text-secondary"><strong>Positive activations:</strong> Meaningful patterns detected</span>
                      </div>
                      <div className="flex items-start space-x-3">
                        <div className="w-2 h-2 rounded-full bg-accent-red mt-2"></div>
                        <span className="text-text-secondary"><strong>Zero activations:</strong> Patterns below detection threshold</span>
                      </div>
                      <div className="flex items-start space-x-3">
                        <div className="w-2 h-2 rounded-full bg-accent-blue mt-2"></div>
                        <span className="text-text-secondary"><strong>Sparse activations:</strong> Focus on relevant indicators</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Dropout Regularization */}
              <div className="bg-bg-primary border border-bg-tertiary rounded-xl p-6">
                <div className="flex items-center space-x-3 mb-4">
                  <div className="inline-flex items-center justify-center w-8 h-8 rounded-lg bg-accent-purple/10 border border-accent-purple/20">
                    <span className="text-accent-purple font-bold">3</span>
                  </div>
                  <h3 className="text-xl font-semibold text-text-primary">Dropout Regularization</h3>
                </div>

                <div className="mb-6">
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div>
                      <div className="bg-bg-secondary rounded-lg p-4 font-mono text-sm">
                        <div><strong>Training:</strong> y = x ⊙ mask / (1-p)</div>
                        <div><strong>Inference:</strong> y = x</div>
                      </div>
                    </div>
                    <div className="text-sm text-text-secondary">
                      <div><strong>p:</strong> dropout probability (e.g., 0.3 = 30% dropped)</div>
                      <div><strong>mask:</strong> random binary mask (0 or 1)</div>
                      <div><strong>⊙:</strong> element-wise multiplication</div>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  <div>
                    <h4 className="font-semibold text-text-primary mb-3">Why Dropout Works:</h4>
                    <div className="space-y-3 text-sm">
                      <div className="flex items-start space-x-3">
                        <div className="w-2 h-2 rounded-full bg-accent-purple mt-2"></div>
                        <span className="text-text-secondary"><strong>Prevents co-adaptation:</strong> Forces independent learning</span>
                      </div>
                      <div className="flex items-start space-x-3">
                        <div className="w-2 h-2 rounded-full bg-accent-purple mt-2"></div>
                        <span className="text-text-secondary"><strong>Ensemble effect:</strong> Trains multiple sub-networks</span>
                      </div>
                      <div className="flex items-start space-x-3">
                        <div className="w-2 h-2 rounded-full bg-accent-purple mt-2"></div>
                        <span className="text-text-secondary"><strong>Robustness:</strong> Works with missing information</span>
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="font-semibold text-text-primary mb-3">Financial Context:</h4>
                    <div className="space-y-3 text-sm">
                      <div className="flex items-start space-x-3">
                        <div className="w-2 h-2 rounded-full bg-accent-green mt-2"></div>
                        <span className="text-text-secondary"><strong>Market regime changes:</strong> Some indicators unavailable</span>
                      </div>
                      <div className="flex items-start space-x-3">
                        <div className="w-2 h-2 rounded-full bg-accent-orange mt-2"></div>
                        <span className="text-text-secondary"><strong>Data quality issues:</strong> Noisy financial indicators</span>
                      </div>
                      <div className="flex items-start space-x-3">
                        <div className="w-2 h-2 rounded-full bg-accent-red mt-2"></div>
                        <span className="text-text-secondary"><strong>Overfitting prevention:</strong> Spurious pattern avoidance</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </motion.section>

        {/* Training Implementation */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Code2 className="w-8 h-8 text-accent-green mr-3" />
            Training Implementation
          </h2>
          
          <p className="text-text-secondary text-lg mb-6">
            Here's how the complete training process is implemented using Skorch (PyTorch wrapper) 
            with Adam optimization and L1/L2 regularization for financial neural networks.
          </p>

          <CodeBlock
            language="python"
            title="Actual Training Implementation with L1/L2 Regularization"
            codeType="actual"
            code={`# From src/utils/training_optuna.py

import torch
import torch.nn as nn
from skorch.regressor import NeuralNetRegressor
from skorch.callbacks import EarlyStopping

# L1 Regularization Mixin
class L1Mixin:
    def __init__(self, l1_lambda=0.0, **kwargs):
        self.l1_lambda = l1_lambda
        super().__init__(**kwargs)

    def get_loss(self, y_pred, y_true, X=None, training=False):
        # Get base loss (MSE) from parent class
        base_loss = super().get_loss(y_pred, y_true, X=X, training=training)
        
        # Add L1 penalty during training if lambda > 0
        if self.l1_lambda > 0 and training:
            l1_penalty = 0.0
            for param in self.module_.parameters():
                if param.requires_grad:
                    l1_penalty += torch.norm(param, 1)
            base_loss = base_loss + self.l1_lambda * l1_penalty
        return base_loss

# Custom Skorch Regressor with L1
class L1Net(L1Mixin, NeuralNetRegressor):
    pass  # Multiple inheritance handles everything

# Export for use in experiments
OptunaSkorchNet = L1Net

# Objective function for Optuna HPO
def _objective(trial, model_module_class, skorch_net_class_to_use, 
               hpo_config_fn_for_trial, X_train_tensor, y_train_tensor, 
               X_val_tensor, y_val_tensor, n_features_val, epochs_val, 
               device_val, batch_size_default_val):
    """Objective function for Optuna study."""
    
    # Get hyperparameters from the config function
    params_from_optuna = hpo_config_fn_for_trial(trial, n_features_val)
    
    # Extract non-module parameters
    module_specific_params = {k: v for k, v in params_from_optuna.items() 
                             if k.startswith("module__")}
    non_module_params = {k: v for k, v in params_from_optuna.items() 
                        if not k.startswith("module__")}
    
    # Create Skorch neural network
    net = skorch_net_class_to_use(
        module=model_module_class,
        module__n_feature=n_features_val,
        module__n_output=1,
        **module_specific_params,
        optimizer=getattr(torch.optim, non_module_params.get("optimizer", "Adam")),
        optimizer__lr=non_module_params.get("lr", 0.001),
        optimizer__weight_decay=non_module_params.get("weight_decay", 0.0),
        l1_lambda=non_module_params.get("l1_lambda", 0.0),
        batch_size=non_module_params.get("batch_size", batch_size_default_val),
        max_epochs=epochs_val,
        device=device_val,
    )
    
    try:
        net.fit(X_train_tensor, y_train_tensor)
        validation_loss = net.history[-1, 'valid_loss']
        return validation_loss
    except Exception as e:
        return float('inf')  # Return large value for failed trials`}
          />
        </motion.section>

        {/* Epoch Training */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.5 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Clock className="w-8 h-8 text-accent-orange mr-3" />
            Understanding Epochs in Training
          </h2>
          
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8 mb-8">
            <p className="text-text-secondary text-lg mb-6">
              An epoch represents one complete pass through the entire training dataset. Neural networks 
              require multiple epochs to gradually learn financial patterns and relationships.
            </p>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
              <div>
                <h3 className="text-xl font-semibold text-text-primary mb-4">What Happens in Each Epoch:</h3>
                <div className="space-y-3 text-sm">
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-orange mt-2"></div>
                    <span className="text-text-secondary"><strong>Data processing:</strong> All training samples processed exactly once</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-orange mt-2"></div>
                    <span className="text-text-secondary"><strong>Batch iterations:</strong> Data divided into mini-batches for efficiency</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-orange mt-2"></div>
                    <span className="text-text-secondary"><strong>Weight updates:</strong> Parameters adjusted after each batch</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-orange mt-2"></div>
                    <span className="text-text-secondary"><strong>Pattern learning:</strong> Network gradually improves predictions</span>
                  </div>
                </div>
              </div>
              
              <div>
                <h3 className="text-xl font-semibold text-text-primary mb-4">Example with Financial Data:</h3>
                <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 space-y-2 text-sm font-mono">
                  <div>Training samples: 1,000 months</div>
                  <div>Batch size: 128</div>
                  <div>Batches per epoch: 8 (7 full + 1 partial)</div>
                  <div className="border-t border-bg-tertiary pt-2">
                    <div className="text-accent-orange">Each sample seen once per epoch</div>
                    <div className="text-accent-orange">All financial relationships considered</div>
                  </div>
                </div>
              </div>
            </div>

            <div className="mb-6">
              <button
                onClick={() => setShowEpochDetails(!showEpochDetails)}
                className="flex items-center space-x-2 text-accent-orange hover:text-accent-orange/80 transition-colors"
              >
                <span>View Learning Progress Across Epochs</span>
                <ChevronRight className={`w-4 h-4 transition-transform ${showEpochDetails ? 'rotate-90' : ''}`} />
              </button>
            </div>

            {showEpochDetails && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                transition={{ duration: 0.3 }}
                className="overflow-hidden"
              >
                <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-6">
                  <h4 className="font-semibold text-text-primary mb-4">Learning Progress Example:</h4>
                  <div className="space-y-3">
                    {epochProgress.map((progress, index) => (
                      <motion.div
                        key={progress.epoch}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.3, delay: index * 0.1 }}
                        className="flex items-center space-x-4 p-3 bg-bg-secondary rounded-lg"
                      >
                        <div className="flex-shrink-0 w-16 text-accent-orange font-mono text-sm">
                          Epoch {progress.epoch}
                        </div>
                        <div className="flex-shrink-0 w-20 font-mono text-sm">
                          Loss: {progress.loss}
                        </div>
                        <div className="flex-shrink-0 w-16 font-mono text-sm">
                          R²: {progress.r2 >= 0 ? '+' : ''}{progress.r2}
                        </div>
                        <div className="flex-1 text-text-secondary text-sm">
                          {progress.status}
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </div>
              </motion.div>
            )}
          </div>
        </motion.section>

        {/* Optimization Details */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Settings className="w-8 h-8 text-accent-purple mr-3" />
            Optimization & Regularization
          </h2>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Adam Optimizer */}
            <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-6">
              <h3 className="text-xl font-semibold text-text-primary mb-4">Adam Optimizer</h3>
              <p className="text-text-secondary mb-4">
                Adaptive optimization algorithm that combines momentum and adaptive learning rates 
                for efficient training on financial data.
              </p>
              
              <div className="space-y-3 text-sm">
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 rounded-full bg-accent-purple mt-2"></div>
                  <span className="text-text-secondary"><strong>Momentum:</strong> Accelerates convergence in consistent directions</span>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 rounded-full bg-accent-purple mt-2"></div>
                  <span className="text-text-secondary"><strong>Adaptive rates:</strong> Different learning rates for each parameter</span>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 rounded-full bg-accent-purple mt-2"></div>
                  <span className="text-text-secondary"><strong>Bias correction:</strong> Handles initialization bias in early epochs</span>
                </div>
              </div>
              
              <div className="mt-4 bg-bg-primary border border-bg-tertiary rounded p-3">
                <div className="text-accent-purple text-xs font-medium mb-1">Typical Settings:</div>
                <div className="font-mono text-xs space-y-1">
                  <div>learning_rate: 0.001</div>
                  <div>beta1: 0.9 (momentum)</div>
                  <div>beta2: 0.999 (variance)</div>
                  <div>epsilon: 1e-8</div>
                </div>
              </div>
            </div>

            {/* Regularization */}
            <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-6">
              <h3 className="text-xl font-semibold text-text-primary mb-4">Regularization Techniques</h3>
              <p className="text-text-secondary mb-4">
                Multiple regularization methods prevent overfitting and improve generalization 
                to unseen financial data.
              </p>
              
              <div className="space-y-4">
                <div>
                  <h4 className="font-medium text-text-primary mb-2">L1 Regularization (Lasso)</h4>
                  <div className="text-sm text-text-secondary mb-2">Adds penalty for absolute weight values, promoting sparsity</div>
                  <div className="bg-bg-primary border border-bg-tertiary rounded p-2 font-mono text-xs">
                    λ₁ × Σ|wᵢ| → Some weights become exactly zero
                  </div>
                </div>
                
                <div>
                  <h4 className="font-medium text-text-primary mb-2">L2 Regularization (Ridge)</h4>
                  <div className="text-sm text-text-secondary mb-2">Adds penalty for squared weight values, reducing weight magnitudes</div>
                  <div className="bg-bg-primary border border-bg-tertiary rounded p-2 font-mono text-xs">
                    λ₂ × Σwᵢ² → All weights become smaller
                  </div>
                </div>
                
                <div>
                  <h4 className="font-medium text-text-primary mb-2">Dropout</h4>
                  <div className="text-sm text-text-secondary">Random neuron deactivation during training (typically 30% for financial data)</div>
                </div>
              </div>
            </div>
          </div>
        </motion.section>

        {/* Navigation */}
        <NavigationButtons 
          prevHref="/architecture"
          prevLabel="Neural Network Architecture"
          nextHref="/hyperparameters"
          nextLabel="Hyperparameter Optimization"
        />
      </div>
    </div>
  )
}