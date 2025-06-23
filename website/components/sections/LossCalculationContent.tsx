'use client'

import { motion } from 'framer-motion'
import { useState } from 'react'
import { Calculator, Target, Settings, TrendingDown, AlertTriangle, CheckCircle, Zap } from 'lucide-react'
import CodeBlock from '@/components/shared/CodeBlock'
import MathFormula from '@/components/shared/MathFormula'
import NavigationButtons from '@/components/shared/NavigationButtons'

const lossComponents = [
  {
    component: 'Primary Loss',
    function: 'Mean Squared Error (MSE)',
    formula: String.raw`MSE = \frac{1}{n} \sum_{i=1}^{n} (y_{\text{true}} - y_{\text{pred}})^2`,
    purpose: 'Measures prediction accuracy',
    weight: '1.0',
    icon: Target,
    color: 'accent-blue'
  },
  {
    component: 'L1 Regularization',
    function: 'Lasso Penalty',
    formula: String.raw`L1 = \lambda_1 \sum_{j} |w_j|`,
    purpose: 'Promotes sparsity in weights',
    weight: 'λ₁ (tunable)',
    icon: Settings,
    color: 'accent-orange'
  },
  {
    component: 'L2 Regularization',
    function: 'Ridge Penalty',
    formula: String.raw`L2 = \lambda_2 \sum_{j} w_j^2`,
    purpose: 'Prevents weight explosion',
    weight: 'λ₂ (tunable)',
    icon: AlertTriangle,
    color: 'accent-purple'
  }
]

const lossExamples = [
  {
    scenario: 'Perfect Prediction',
    actual: 0.08,
    predicted: 0.08,
    error: 0.00,
    squaredError: 0.0000,
    interpretation: 'Model perfectly captures equity premium'
  },
  {
    scenario: 'Small Error',
    actual: 0.08,
    predicted: 0.075,
    error: 0.005,
    squaredError: 0.0000025,
    interpretation: 'Close prediction with minimal loss'
  },
  {
    scenario: 'Moderate Error',
    actual: 0.08,
    predicted: 0.06,
    error: 0.02,
    squaredError: 0.0004,
    interpretation: 'Noticeable prediction error requiring adjustment'
  },
  {
    scenario: 'Large Error',
    actual: 0.08,
    predicted: 0.02,
    error: 0.06,
    squaredError: 0.0036,
    interpretation: 'Significant misprediction driving strong gradient signal'
  }
]

export default function LossCalculationContent() {
  const [activeComponent, setActiveComponent] = useState(0)
  const [selectedExample, setSelectedExample] = useState(0)

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
            <span className="gradient-text">Loss Calculation</span>
          </h1>
          <p className="max-w-4xl mx-auto text-xl text-text-secondary leading-relaxed">
            Understanding how neural networks quantify prediction errors through MSE loss and 
            regularization techniques for robust equity premium forecasting.
          </p>
        </motion.div>

        {/* Loss Function Overview */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Calculator className="w-8 h-8 text-accent-blue mr-3" />
            Loss Function Components
          </h2>
          
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8 mb-8">
            <p className="text-text-secondary text-lg mb-8 leading-relaxed">
              The total loss function combines prediction accuracy (MSE) with regularization terms 
              to balance fitting the training data while maintaining model generalization capability.
            </p>

            <div className="mb-8">
              <h3 className="text-xl font-semibold text-text-primary mb-4">Complete Loss Function:</h3>
              <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-6">
                <MathFormula latex={String.raw`L_{total} = MSE + \lambda_1 \cdot L1 + \lambda_2 \cdot L2`} block={true} />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {lossComponents.map((component, index) => {
                const Icon = component.icon
                const isActive = activeComponent === index
                return (
                  <button
                    key={component.component}
                    onClick={() => setActiveComponent(index)}
                    className={`p-6 rounded-lg border transition-all text-left ${
                      isActive 
                        ? `bg-${component.color}/10 border-${component.color}/30` 
                        : 'bg-bg-primary border-bg-tertiary hover:border-accent-blue/30'
                    }`}
                  >
                    <div className="flex items-center space-x-3 mb-4">
                      <div className={`inline-flex items-center justify-center w-10 h-10 rounded-lg bg-${component.color}/10 border border-${component.color}/20`}>
                        <Icon className={`w-5 h-5 text-${component.color}`} />
                      </div>
                      <div>
                        <h3 className="font-semibold text-text-primary text-sm">{component.component}</h3>
                        <p className="text-text-muted text-xs">Weight: {component.weight}</p>
                      </div>
                    </div>
                    <MathFormula latex={component.formula} className="mb-2" />
                    <p className="text-text-secondary text-sm">{component.purpose}</p>
                  </button>
                )
              })}
            </div>

            {/* Active Component Details */}
            {lossComponents.map((component, index) => {
              const Icon = component.icon
              if (activeComponent !== index) return null
              
              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                  className={`mt-8 bg-${component.color}/5 border border-${component.color}/20 rounded-xl p-8`}
                >
                  <div className="flex items-center space-x-3 mb-6">
                    <div className={`inline-flex items-center justify-center w-12 h-12 rounded-lg bg-${component.color}/10 border border-${component.color}/20`}>
                      <Icon className={`w-6 h-6 text-${component.color}`} />
                    </div>
                    <div>
                      <h3 className="text-2xl font-bold text-text-primary">{component.function}</h3>
                      <p className="text-text-secondary">{component.purpose}</p>
                    </div>
                  </div>

                  <div className="mb-6">
                    <MathFormula latex={component.formula} block={true} />
                  </div>

                  {index === 0 && (
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                      <div>
                        <h4 className="font-semibold text-text-primary mb-3">Why MSE for Financial Data:</h4>
                        <div className="space-y-3 text-sm">
                          <div className="flex items-start space-x-3">
                            <div className="w-2 h-2 rounded-full bg-accent-blue mt-2"></div>
                            <span className="text-text-secondary"><strong>Penalizes large errors:</strong> Squared term heavily weights outliers</span>
                          </div>
                          <div className="flex items-start space-x-3">
                            <div className="w-2 h-2 rounded-full bg-accent-blue mt-2"></div>
                            <span className="text-text-secondary"><strong>Smooth gradients:</strong> Differentiable everywhere for optimization</span>
                          </div>
                          <div className="flex items-start space-x-3">
                            <div className="w-2 h-2 rounded-full bg-accent-blue mt-2"></div>
                            <span className="text-text-secondary"><strong>Financial interpretation:</strong> Prediction variance directly impacts portfolio returns</span>
                          </div>
                        </div>
                      </div>
                      <div>
                        <h4 className="font-semibold text-text-primary mb-3">Mathematical Properties:</h4>
                        <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 space-y-2 text-sm">
                          <div><strong>Convex:</strong> Single global minimum guarantees convergence</div>
                          <div><strong>Scale sensitive:</strong> Larger errors contribute exponentially more</div>
                          <div><strong>Unbiased estimator:</strong> Expected value equals true error variance</div>
                        </div>
                      </div>
                    </div>
                  )}

                  {index === 1 && (
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                      <div>
                        <h4 className="font-semibold text-text-primary mb-3">L1 Regularization Effects:</h4>
                        <div className="space-y-3 text-sm">
                          <div className="flex items-start space-x-3">
                            <div className="w-2 h-2 rounded-full bg-accent-orange mt-2"></div>
                            <span className="text-text-secondary"><strong>Feature selection:</strong> Drives irrelevant weights to exactly zero</span>
                          </div>
                          <div className="flex items-start space-x-3">
                            <div className="w-2 h-2 rounded-full bg-accent-orange mt-2"></div>
                            <span className="text-text-secondary"><strong>Sparsity promotion:</strong> Creates interpretable models</span>
                          </div>
                          <div className="flex items-start space-x-3">
                            <div className="w-2 h-2 rounded-full bg-accent-orange mt-2"></div>
                            <span className="text-text-secondary"><strong>Robust to outliers:</strong> Linear penalty vs squared in L2</span>
                          </div>
                        </div>
                      </div>
                      <div>
                        <h4 className="font-semibold text-text-primary mb-3">Financial Application:</h4>
                        <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 space-y-2 text-sm">
                          <div><strong>Indicator selection:</strong> Identifies most predictive financial variables</div>
                          <div><strong>Model simplification:</strong> Reduces overfitting in noisy markets</div>
                          <div><strong>Interpretability:</strong> Highlights key economic relationships</div>
                        </div>
                      </div>
                    </div>
                  )}

                  {index === 2 && (
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                      <div>
                        <h4 className="font-semibold text-text-primary mb-3">L2 Regularization Effects:</h4>
                        <div className="space-y-3 text-sm">
                          <div className="flex items-start space-x-3">
                            <div className="w-2 h-2 rounded-full bg-accent-purple mt-2"></div>
                            <span className="text-text-secondary"><strong>Weight decay:</strong> Prevents any single weight from becoming too large</span>
                          </div>
                          <div className="flex items-start space-x-3">
                            <div className="w-2 h-2 rounded-full bg-accent-purple mt-2"></div>
                            <span className="text-text-secondary"><strong>Smooth solutions:</strong> Encourages distributed weight patterns</span>
                          </div>
                          <div className="flex items-start space-x-3">
                            <div className="w-2 h-2 rounded-full bg-accent-purple mt-2"></div>
                            <span className="text-text-secondary"><strong>Stability:</strong> Improves conditioning of optimization problem</span>
                          </div>
                        </div>
                      </div>
                      <div>
                        <h4 className="font-semibold text-text-primary mb-3">Typical Values:</h4>
                        <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 space-y-2 text-sm">
                          <div><strong>λ₂ range:</strong> 1e-6 to 1e-2 for financial models</div>
                          <div><strong>Tuning:</strong> Validated through cross-validation</div>
                          <div><strong>Balance:</strong> Too high = underfitting, too low = overfitting</div>
                        </div>
                      </div>
                    </div>
                  )}
                </motion.div>
              )
            })}
          </div>
        </motion.section>

        {/* MSE Calculation Examples */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <TrendingDown className="w-8 h-8 text-accent-green mr-3" />
            MSE Calculation Examples
          </h2>
          
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8 mb-8">
            <p className="text-text-secondary text-lg mb-8">
              Understanding how different prediction errors contribute to the overall loss helps 
              interpret model performance and training dynamics.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
              {lossExamples.map((example, index) => (
                <button
                  key={example.scenario}
                  onClick={() => setSelectedExample(index)}
                  className={`p-4 rounded-lg border transition-all text-left ${
                    selectedExample === index
                      ? 'bg-accent-green/10 border-accent-green/30'
                      : 'bg-bg-primary border-bg-tertiary hover:border-accent-blue/30'
                  }`}
                >
                  <div className="text-sm font-semibold text-text-primary mb-2">{example.scenario}</div>
                  <div className="text-xs text-text-muted mb-1">Error: {example.error.toFixed(3)}</div>
                  <div className="text-xs font-mono text-accent-blue">MSE: {example.squaredError.toFixed(6)}</div>
                </button>
              ))}
            </div>

            {/* Selected Example Details */}
            <motion.div
              key={selectedExample}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              className="bg-accent-green/5 border border-accent-green/20 rounded-xl p-8"
            >
              <h3 className="text-2xl font-bold text-text-primary mb-6">{lossExamples[selectedExample].scenario}</h3>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-6">
                <div>
                  <h4 className="font-semibold text-text-primary mb-4">Calculation Steps:</h4>
                  <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 space-y-3 text-sm font-mono">
                    <div><strong>Actual Equity Premium:</strong> {lossExamples[selectedExample].actual.toFixed(3)}</div>
                    <div><strong>Predicted Value:</strong> {lossExamples[selectedExample].predicted.toFixed(3)}</div>
                    <div className="border-t border-bg-tertiary pt-2">
                      <strong>Error:</strong> {lossExamples[selectedExample].actual.toFixed(3)} - {lossExamples[selectedExample].predicted.toFixed(3)} = {lossExamples[selectedExample].error.toFixed(3)}
                    </div>
                    <div>
                      <strong>Squared Error:</strong> ({lossExamples[selectedExample].error.toFixed(3)})² = {lossExamples[selectedExample].squaredError.toFixed(6)}
                    </div>
                  </div>
                </div>
                
                <div>
                  <h4 className="font-semibold text-text-primary mb-4">Interpretation:</h4>
                  <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4">
                    <p className="text-text-secondary text-sm mb-3">{lossExamples[selectedExample].interpretation}</p>
                    <div className="space-y-2 text-xs">
                      <div className="flex justify-between">
                        <span>Gradient Signal:</span>
                        <span className={`font-mono ${
                          Math.abs(lossExamples[selectedExample].error) > 0.03 ? 'text-accent-red' :
                          Math.abs(lossExamples[selectedExample].error) > 0.01 ? 'text-accent-orange' : 'text-accent-green'
                        }`}>
                          {Math.abs(lossExamples[selectedExample].error) > 0.03 ? 'Strong' :
                           Math.abs(lossExamples[selectedExample].error) > 0.01 ? 'Moderate' : 'Weak'}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Weight Update:</span>
                        <span className="font-mono">{(lossExamples[selectedExample].error * 2).toFixed(4)}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-accent-green/10 border border-accent-green/20 rounded-lg p-4">
                <h4 className="font-semibold text-accent-green mb-2">Financial Impact:</h4>
                <p className="text-text-secondary text-sm">
                  A {Math.abs(lossExamples[selectedExample].error * 100).toFixed(1)}% prediction error in equity premium 
                  translates to significant portfolio allocation differences and potential return impact.
                </p>
              </div>
            </motion.div>
          </div>
        </motion.section>

        {/* Batch Loss Calculation */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <CheckCircle className="w-8 h-8 text-accent-orange mr-3" />
            Batch Loss Computation
          </h2>
          
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8 mb-8">
            <p className="text-text-secondary text-lg mb-6">
              Neural networks process data in batches, computing loss across multiple samples 
              simultaneously for efficient training and stable gradient estimates.
            </p>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
              <div>
                <h3 className="text-xl font-semibold text-text-primary mb-4">Mathematical Formulation:</h3>
                <div className="space-y-4">
                  <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4">
                    <div className="text-sm text-text-muted mb-2">For batch size B:</div>
                    <MathFormula latex={String.raw`MSE_{batch} = \frac{1}{B} \sum_{i=1}^{B} (y_i - \hat{y}_i)^2`} block={true} />
                  </div>
                  <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4">
                    <div className="text-sm text-text-muted mb-2">With regularization:</div>
                    <MathFormula latex={String.raw`L_{total} = MSE_{batch} + \lambda_1 \|W\|_1 + \lambda_2 \|W\|_2^2`} block={true} />
                  </div>
                </div>
              </div>
              
              <div>
                <h3 className="text-xl font-semibold text-text-primary mb-4">Batch Processing Benefits:</h3>
                <div className="space-y-3 text-sm">
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-orange mt-2"></div>
                    <span className="text-text-secondary"><strong>Computational efficiency:</strong> Parallel matrix operations</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-orange mt-2"></div>
                    <span className="text-text-secondary"><strong>Gradient stability:</strong> Averaged gradients reduce noise</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-orange mt-2"></div>
                    <span className="text-text-secondary"><strong>Memory optimization:</strong> Better GPU utilization</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-orange mt-2"></div>
                    <span className="text-text-secondary"><strong>Regularization effectiveness:</strong> Consistent penalty application</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-accent-orange/5 border border-accent-orange/20 rounded-lg p-6">
              <h4 className="font-semibold text-accent-orange mb-3">Typical Batch Sizes for Financial Data:</h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                <div className="text-center">
                  <div className="font-mono text-accent-orange text-lg">32-64</div>
                  <div className="text-text-muted">Small datasets</div>
                  <div className="text-text-secondary">Higher variance, faster updates</div>
                </div>
                <div className="text-center">
                  <div className="font-mono text-accent-orange text-lg">128-256</div>
                  <div className="text-text-muted">Medium datasets</div>
                  <div className="text-text-secondary">Balanced efficiency and stability</div>
                </div>
                <div className="text-center">
                  <div className="font-mono text-accent-orange text-lg">512+</div>
                  <div className="text-text-muted">Large datasets</div>
                  <div className="text-text-secondary">Lower variance, stable gradients</div>
                </div>
              </div>
            </div>
          </div>
        </motion.section>

        {/* Implementation Example */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.5 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Zap className="w-8 h-8 text-accent-blue mr-3" />
            Loss Calculation Implementation
          </h2>
          
          <p className="text-text-secondary text-lg mb-6">
            Here's how the complete loss function is implemented in PyTorch, including MSE calculation, 
            regularization terms, and batch processing for financial prediction models.
          </p>

          <CodeBlock
            language="python"
            title="Complete Loss Function Implementation"
            code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class FinancialLoss(nn.Module):
    """
    Complete loss function for financial neural networks with MSE + L1/L2 regularization.
    """
    def __init__(self, l1_lambda=0.0, l2_lambda=0.0):
        super().__init__()
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions, targets, model):
        """
        Compute total loss = MSE + L1 + L2 regularization.
        
        Args:
            predictions: Model predictions [batch_size, 1]
            targets: True equity premiums [batch_size, 1] 
            model: Neural network model for weight extraction
            
        Returns:
            total_loss: Combined loss value
            loss_components: Dictionary with individual loss terms
        """
        # Primary MSE loss
        mse_loss = self.mse_loss(predictions, targets)
        
        # L1 regularization (sum of absolute weights)
        l1_penalty = 0.0
        if self.l1_lambda > 0:
            for param in model.parameters():
                if param.requires_grad:
                    l1_penalty += torch.sum(torch.abs(param))
        
        # L2 regularization (sum of squared weights)
        l2_penalty = 0.0
        if self.l2_lambda > 0:
            for param in model.parameters():
                if param.requires_grad:
                    l2_penalty += torch.sum(param ** 2)
        
        # Total loss
        total_loss = mse_loss + self.l1_lambda * l1_penalty + self.l2_lambda * l2_penalty
        
        # Return components for monitoring
        loss_components = {
            'mse': mse_loss.item(),
            'l1': l1_penalty.item() if self.l1_lambda > 0 else 0.0,
            'l2': l2_penalty.item() if self.l2_lambda > 0 else 0.0,
            'total': total_loss.item()
        }
        
        return total_loss, loss_components

# Batch loss calculation example
def calculate_batch_loss_example():
    """
    Demonstrate loss calculation on a batch of financial predictions.
    """
    # Simulated batch of equity premium predictions
    batch_size = 32
    predictions = torch.tensor([
        [0.078], [0.065], [0.082], [0.071], [0.088], [0.063], [0.075], [0.069],
        [0.084], [0.077], [0.062], [0.089], [0.073], [0.081], [0.067], [0.085],
        [0.079], [0.072], [0.086], [0.064], [0.083], [0.076], [0.068], [0.087],
        [0.074], [0.080], [0.066], [0.090], [0.070], [0.082], [0.078], [0.073]
    ], dtype=torch.float32)
    
    # True equity premiums
    targets = torch.tensor([
        [0.080], [0.070], [0.085], [0.075], [0.090], [0.065], [0.078], [0.072],
        [0.087], [0.080], [0.065], [0.092], [0.076], [0.084], [0.070], [0.088],
        [0.082], [0.075], [0.089], [0.067], [0.086], [0.079], [0.071], [0.090],
        [0.077], [0.083], [0.069], [0.093], [0.073], [0.085], [0.081], [0.076]
    ], dtype=torch.float32)
    
    print(f"Batch size: {batch_size}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")
    
    # Calculate individual squared errors
    squared_errors = (predictions - targets) ** 2
    print(f"\\nSquared errors (first 5): {squared_errors[:5].flatten()}")
    
    # Batch MSE
    batch_mse = torch.mean(squared_errors)
    print(f"Batch MSE: {batch_mse.item():.6f}")
    
    # Manual verification
    manual_mse = torch.sum(squared_errors) / batch_size
    print(f"Manual MSE calculation: {manual_mse.item():.6f}")
    
    # Individual sample contributions
    print(f"\\nLargest error: {torch.max(squared_errors).item():.6f}")
    print(f"Smallest error: {torch.min(squared_errors).item():.6f}")
    print(f"Error std: {torch.std(squared_errors).item():.6f}")
    
    return batch_mse

# Example usage with model
def training_step_example():
    """
    Complete training step showing loss calculation and backpropagation.
    """
    from src.models.nns import Net3  # Assuming Net3 model available
    
    # Initialize model and loss function
    model = Net3(n_feature=30, n_hidden1=64, n_hidden2=32, n_hidden3=16, 
                 n_output=1, dropout=0.2)
    
    loss_fn = FinancialLoss(l1_lambda=1e-5, l2_lambda=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Simulated financial indicators (batch_size=32, features=30)
    financial_data = torch.randn(32, 30)
    targets = torch.randn(32, 1) * 0.02 + 0.08  # Mean ~8% with 2% std
    
    # Forward pass
    model.train()
    predictions = model(financial_data)
    
    # Loss calculation
    total_loss, loss_components = loss_fn(predictions, targets, model)
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    # Print loss breakdown
    print("Loss Components:")
    print(f"  MSE Loss: {loss_components['mse']:.6f}")
    print(f"  L1 Penalty: {loss_components['l1']:.6f}")
    print(f"  L2 Penalty: {loss_components['l2']:.6f}")
    print(f"  Total Loss: {loss_components['total']:.6f}")
    
    return loss_components

# Run examples
if __name__ == "__main__":
    print("=== Batch Loss Calculation ===")
    calculate_batch_loss_example()
    
    print("\\n=== Training Step with Regularization ===")
    training_step_example()`}
          />
        </motion.section>

        {/* Navigation */}
        <NavigationButtons 
          prevHref="/forward-pass"
          prevLabel="Forward Pass"
          nextHref="/backpropagation"
          nextLabel="Backpropagation"
        />
      </div>
    </div>
  )
}