'use client'

import { motion } from 'framer-motion'
import { useState } from 'react'
import { ArrowLeft, Calculator, Zap, GitBranch, Target, Brain, ChevronRight, Play } from 'lucide-react'
import CodeBlock from '@/components/shared/CodeBlock'
import MathFormula from '@/components/shared/MathFormula'
import NavigationButtons from '@/components/shared/NavigationButtons'

interface BackpropStep {
  step: number;
  title: string;
  description: string;
  formula: string;
  purpose: string;
  details: string;
}

const backpropSteps: BackpropStep[] = [
  {
    step: 1,
    title: 'Output Layer Gradient',
    description: 'Compute gradient of loss with respect to final predictions',
    formula: '∂L/∂ŷ = 2(ŷ - y)',
    purpose: 'Initiates the gradient flow backward through the network',
    details: 'MSE derivative provides the starting point for all subsequent gradient calculations'
  },
  {
    step: 2,  
    title: 'Output Weights Gradient',
    description: 'Calculate gradients for final layer weights and biases',
    formula: '∂L/∂W₄ = (∂L/∂ŷ) · h₃ᵀ',
    purpose: 'Updates output layer parameters to minimize prediction error',
    details: 'Direct connection between hidden layer activations and output gradients'
  },
  {
    step: 3,
    title: 'Hidden Layer Error',
    description: 'Propagate error signal backward through activation functions',
    formula: '∂L/∂h₃ = W₄ᵀ · (∂L/∂ŷ)',
    purpose: 'Distributes output error to hidden layer neurons',
    details: 'Chain rule application connecting output error to hidden representations'
  },
  {
    step: 4,
    title: 'Activation Gradient',
    description: 'Apply derivative of ReLU activation function',
    formula: '∂L/∂z₃ = (∂L/∂h₃) ⊙ ReLU\'(z₃)',
    purpose: 'Accounts for non-linear transformation in gradient flow',
    details: 'ReLU derivative is 1 for positive values, 0 for negative values'
  },
  {
    step: 5,
    title: 'Weight Update Gradients',
    description: 'Calculate gradients for hidden layer weights',
    formula: '∂L/∂W₃ = (∂L/∂z₃) · h₂ᵀ',
    purpose: 'Determines how to adjust hidden layer parameters',
    details: 'Links previous layer activations to current layer error signals'
  },
  {
    step: 6,
    title: 'Recursive Propagation',
    description: 'Repeat process for all remaining layers',
    formula: 'Continue until ∂L/∂W₁',
    purpose: 'Ensures all network parameters receive gradient updates',
    details: 'Each layer follows the same pattern: error → activation → weights'
  }
]

const chainRuleExample = [
  {
    layer: 'Output',
    operation: 'Loss Function',
    input: 'ŷ = W₄h₃ + b₄',
    gradient: '∂L/∂ŷ = 2(ŷ - y)',
    explanation: 'MSE derivative with respect to predictions'
  },
  {
    layer: 'Layer 3',
    operation: 'Linear + ReLU',
    input: 'h₃ = ReLU(W₃h₂ + b₃)',
    gradient: '∂L/∂W₃ = (∂L/∂h₃) · h₂ᵀ',
    explanation: 'Chain rule: output gradient × input activations'
  },
  {
    layer: 'Layer 2', 
    operation: 'Linear + ReLU',
    input: 'h₂ = ReLU(W₂h₁ + b₂)',
    gradient: '∂L/∂W₂ = (∂L/∂h₂) · h₁ᵀ',
    explanation: 'Propagated error from layer 3 × layer 1 activations'
  },
  {
    layer: 'Layer 1',
    operation: 'Linear + ReLU',
    input: 'h₁ = ReLU(W₁x + b₁)',
    gradient: '∂L/∂W₁ = (∂L/∂h₁) · xᵀ',
    explanation: 'Final gradient using original input features'
  }
]

export default function BackpropagationContent() {
  const [activeStep, setActiveStep] = useState(1)
  const [selectedLayer, setSelectedLayer] = useState(0)

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
            <span className="gradient-text">Backpropagation</span>
          </h1>
          <p className="max-w-4xl mx-auto text-xl text-text-secondary leading-relaxed">
            The chain rule in action: how neural networks compute gradients to learn optimal weights 
            for equity premium prediction through efficient error propagation.
          </p>
        </motion.div>

        {/* Backpropagation Overview */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <ArrowLeft className="w-8 h-8 text-accent-blue mr-3" />
            What is Backpropagation?
          </h2>
          
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8 mb-8">
            <p className="text-text-secondary text-lg mb-6 leading-relaxed">
              Backpropagation is the algorithm that enables neural networks to learn by computing 
              gradients of the loss function with respect to each weight in the network. It works 
              backward from the output error to efficiently calculate how much each parameter 
              contributed to the total prediction error.
            </p>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div>
                <h3 className="text-xl font-semibold text-text-primary mb-4">Core Principle:</h3>
                <div className="space-y-3 text-sm">
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-blue mt-2"></div>
                    <span className="text-text-secondary"><strong>Chain Rule Application:</strong> Derivatives flow backward through compositions</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-blue mt-2"></div>
                    <span className="text-text-secondary"><strong>Efficient Computation:</strong> Reuses intermediate calculations</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-blue mt-2"></div>
                    <span className="text-text-secondary"><strong>Error Attribution:</strong> Assigns responsibility to each parameter</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-blue mt-2"></div>
                    <span className="text-text-secondary"><strong>Local Gradients:</strong> Each layer computes its own derivatives</span>
                  </div>
                </div>
              </div>
              
              <div>
                <h3 className="text-xl font-semibold text-text-primary mb-4">Mathematical Foundation:</h3>
                <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 space-y-3 text-sm">
                  <div>
                    <div className="text-text-muted mb-1">Chain Rule:</div>
                    <MathFormula latex="\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial W}" />
                  </div>
                  <div>
                    <div className="text-text-muted mb-1">Multi-layer Extension:</div>
                    <MathFormula latex="\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial h_3} \cdot \frac{\partial h_3}{\partial h_2} \cdot \frac{\partial h_2}{\partial W_1}" />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </motion.section>

        {/* Six-Step Process */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Calculator className="w-8 h-8 text-accent-orange mr-3" />
            Six-Step Backpropagation Process
          </h2>
          
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8 mb-8">
            <p className="text-text-secondary text-lg mb-8">
              Follow the systematic process of computing gradients layer by layer, starting from 
              the output error and working backward to the input weights. Click each step to 
              explore the mathematical details.
            </p>

            {/* Step Selector */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
              {backpropSteps.map((step) => {
                const isActive = activeStep === step.step
                return (
                  <button
                    key={step.step}
                    onClick={() => setActiveStep(step.step)}
                    className={`p-4 rounded-lg border transition-all text-left ${
                      isActive 
                        ? 'bg-accent-orange/10 border-accent-orange/30' 
                        : 'bg-bg-primary border-bg-tertiary hover:border-accent-blue/30'
                    }`}
                  >
                    <div className="flex items-center space-x-3 mb-2">
                      <div className={`inline-flex items-center justify-center w-8 h-8 rounded-lg ${
                        isActive ? 'bg-accent-orange/20' : 'bg-bg-tertiary'
                      }`}>
                        <span className={`font-bold text-sm ${
                          isActive ? 'text-accent-orange' : 'text-text-muted'
                        }`}>{step.step}</span>
                      </div>
                      <div>
                        <div className="font-semibold text-text-primary text-sm">{step.title}</div>
                      </div>
                    </div>
                    <div className="font-mono text-xs text-accent-blue">{step.formula}</div>
                  </button>
                )
              })}
            </div>

            {/* Active Step Details */}
            {backpropSteps.map((step) => {
              if (activeStep !== step.step) return null
              
              return (
                <motion.div
                  key={step.step}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                  className="bg-accent-orange/5 border border-accent-orange/20 rounded-xl p-8"
                >
                  <div className="flex items-center space-x-3 mb-6">
                    <div className="inline-flex items-center justify-center w-12 h-12 rounded-lg bg-accent-orange/10 border border-accent-orange/20">
                      <span className="text-accent-orange font-bold text-lg">{step.step}</span>
                    </div>
                    <div>
                      <h3 className="text-2xl font-bold text-text-primary">{step.title}</h3>
                      <p className="text-text-secondary">{step.description}</p>
                    </div>
                  </div>

                  <div className="mb-6">
                    <MathFormula latex={step.formula} block={true} />
                  </div>

                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    <div>
                      <h4 className="font-semibold text-text-primary mb-3">Purpose:</h4>
                      <p className="text-text-secondary mb-4">{step.purpose}</p>
                      
                      <div className="bg-accent-orange/10 border border-accent-orange/20 rounded-lg p-4">
                        <div className="text-accent-orange text-sm font-medium mb-1">Implementation Detail:</div>
                        <div className="text-text-secondary text-sm">{step.details}</div>
                      </div>
                    </div>
                    
                    <div>
                      {step.step === 1 && (
                        <div>
                          <h4 className="font-semibold text-text-primary mb-3">MSE Gradient Calculation:</h4>
                          <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 space-y-2 text-sm font-mono">
                            <div><strong>Loss:</strong> L = (ŷ - y)²</div>
                            <div><strong>Derivative:</strong> ∂L/∂ŷ = 2(ŷ - y)</div>
                            <div className="border-t border-bg-tertiary pt-2">
                              <strong>Example:</strong> If ŷ=0.075, y=0.08
                            </div>
                            <div className="text-accent-orange">
                              <strong>Gradient:</strong> 2(0.075 - 0.08) = -0.01
                            </div>
                          </div>
                        </div>
                      )}
                      
                      {step.step === 2 && (
                        <div>
                          <h4 className="font-semibold text-text-primary mb-3">Weight Gradient Matrix:</h4>
                          <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 space-y-2 text-sm font-mono">
                            <div><strong>Input h₃:</strong> [16×1] hidden activations</div>
                            <div><strong>Output grad:</strong> [1×1] loss gradient</div>
                            <div><strong>Weight grad:</strong> [1×16] = [1×1] × [16×1]<sup>T</sup></div>
                            <div className="text-accent-orange">Each weight gets its own gradient value</div>
                          </div>
                        </div>
                      )}
                      
                      {step.step === 3 && (
                        <div>
                          <h4 className="font-semibold text-text-primary mb-3">Error Distribution:</h4>
                          <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 space-y-2 text-sm font-mono">
                            <div><strong>Weight Matrix:</strong> W₄ [1×16]</div>
                            <div><strong>Output Error:</strong> ∂L/∂ŷ [1×1]</div>
                            <div><strong>Hidden Error:</strong> [16×1] = W₄ᵀ × ∂L/∂ŷ</div>
                            <div className="text-accent-orange">Each hidden unit receives proportional error</div>
                          </div>
                        </div>
                      )}
                      
                      {step.step === 4 && (
                        <div>
                          <h4 className="font-semibold text-text-primary mb-3">ReLU Derivative:</h4>
                          <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 space-y-2 text-sm font-mono">
                            <div>If z₃ &gt; 0: ReLU'(z₃) = 1 <span className="text-accent-green">(gradient passes)</span></div>
                            <div>If z₃ ≤ 0: ReLU'(z₃) = 0 <span className="text-accent-red">(gradient blocked)</span></div>
                            <div className="border-t border-bg-tertiary pt-2">
                              <div className="text-accent-orange">Element-wise multiplication with error</div>
                            </div>
                          </div>
                        </div>
                      )}
                      
                      {step.step === 5 && (
                        <div>
                          <h4 className="font-semibold text-text-primary mb-3">Hidden Weight Updates:</h4>
                          <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 space-y-2 text-sm font-mono">
                            <div><strong>Error Signal:</strong> ∂L/∂z₃ [16×1]</div>
                            <div><strong>Previous Activation:</strong> h₂ [32×1]</div>
                            <div><strong>Weight Gradient:</strong> [16×32] matrix</div>
                            <div className="text-accent-orange">∂L/∂W₃[i,j] = ∂L/∂z₃[i] × h₂[j]</div>
                          </div>
                        </div>
                      )}
                      
                      {step.step === 6 && (
                        <div>
                          <h4 className="font-semibold text-text-primary mb-3">Layer-by-Layer Pattern:</h4>
                          <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 space-y-2 text-sm">
                            <div><strong>Layer 3 → 2:</strong> Repeat gradient computation</div>
                            <div><strong>Layer 2 → 1:</strong> Continue backward propagation</div>
                            <div><strong>Layer 1:</strong> Use original input features</div>
                            <div className="text-accent-orange"><strong>Result:</strong> All weights have gradients</div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </motion.div>
              )
            })}
          </div>
        </motion.section>

        {/* Chain Rule Visualization */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <GitBranch className="w-8 h-8 text-accent-green mr-3" />
            Chain Rule Through Network Layers
          </h2>
          
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8 mb-8">
            <p className="text-text-secondary text-lg mb-8">
              The chain rule enables efficient gradient computation by decomposing the complex 
              derivative into manageable layer-by-layer calculations. Click each layer to see 
              how gradients flow through the network.
            </p>

            <div className="space-y-4 mb-8">
              {chainRuleExample.map((layer, index) => (
                <button
                  key={layer.layer}
                  onClick={() => setSelectedLayer(index)}
                  className={`w-full flex items-center justify-between p-6 rounded-lg border transition-all ${
                    selectedLayer === index
                      ? 'bg-accent-green/10 border-accent-green/30'
                      : 'bg-bg-primary border-bg-tertiary hover:border-accent-blue/30'
                  }`}
                >
                  <div className="flex items-center space-x-4">
                    <div className={`inline-flex items-center justify-center w-10 h-10 rounded-lg ${
                      selectedLayer === index ? 'bg-accent-green/20' : 'bg-bg-tertiary'
                    }`}>
                      <Brain className={`w-5 h-5 ${
                        selectedLayer === index ? 'text-accent-green' : 'text-text-muted'
                      }`} />
                    </div>
                    <div className="text-left">
                      <h3 className="font-semibold text-text-primary">{layer.layer}</h3>
                      <p className="text-text-secondary text-sm">{layer.operation}</p>
                    </div>
                  </div>
                  <ChevronRight className={`w-5 h-5 transition-transform ${
                    selectedLayer === index ? 'rotate-90 text-accent-green' : 'text-text-muted'
                  }`} />
                </button>
              ))}
            </div>

            {/* Selected Layer Details */}
            <motion.div
              key={selectedLayer}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              className="bg-accent-green/5 border border-accent-green/20 rounded-xl p-8"
            >
              <h3 className="text-2xl font-bold text-text-primary mb-6">{chainRuleExample[selectedLayer].layer} Layer</h3>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div>
                  <h4 className="font-semibold text-text-primary mb-4">Forward Pass:</h4>
                  <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 mb-4">
                    <code className="text-accent-blue">{chainRuleExample[selectedLayer].input}</code>
                  </div>
                  
                  <h4 className="font-semibold text-text-primary mb-4">Gradient Calculation:</h4>
                  <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4">
                    <MathFormula latex={chainRuleExample[selectedLayer].gradient} />
                  </div>
                </div>
                
                <div>
                  <h4 className="font-semibold text-text-primary mb-4">Explanation:</h4>
                  <p className="text-text-secondary mb-4">{chainRuleExample[selectedLayer].explanation}</p>
                  
                  <div className="bg-accent-green/10 border border-accent-green/20 rounded-lg p-4">
                    <h5 className="font-semibold text-accent-green mb-2">Chain Rule Application:</h5>
                    <div className="text-text-secondary text-sm space-y-1">
                      {selectedLayer === 0 && (
                        <>
                          <div>• Direct derivative of MSE loss function</div>
                          <div>• Starting point for all subsequent calculations</div>
                          <div>• Magnitude indicates prediction error severity</div>
                        </>
                      )}
                      {selectedLayer === 1 && (
                        <>
                          <div>• Combines output gradient with layer 2 activations</div>
                          <div>• Each weight receives gradient proportional to input</div>
                          <div>• Matrix multiplication: outer product computation</div>
                        </>
                      )}
                      {selectedLayer === 2 && (
                        <>
                          <div>• Error propagated from layer 3 through weights</div>
                          <div>• ReLU derivative filters gradient flow</div>
                          <div>• Local gradient multiplied by upstream error</div>
                        </>
                      )}
                      {selectedLayer === 3 && (
                        <>
                          <div>• Uses original financial indicators as inputs</div>
                          <div>• Connects raw features to final prediction error</div>
                          <div>• Completes end-to-end gradient computation</div>
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
          transition={{ duration: 0.8, delay: 0.5 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Target className="w-8 h-8 text-accent-purple mr-3" />
            Backpropagation Implementation
          </h2>
          
          <p className="text-text-secondary text-lg mb-6">
            Here's how backpropagation is implemented in PyTorch, showing the automatic 
            differentiation process and manual gradient computation for understanding 
            the underlying mathematics.
          </p>

          <CodeBlock
            language="python"
            title="Backpropagation Implementation and Visualization"
            code={`import torch
import torch.nn as nn
import torch.nn.functional as F

def manual_backpropagation_example():
    """
    Manual implementation of backpropagation through a simple network.
    Demonstrates the mathematical steps for educational purposes.
    """
    # Simple 2-layer network: 30 → 16 → 1
    torch.manual_seed(42)  # Reproducible results
    
    # Network parameters
    W1 = torch.randn(16, 30, requires_grad=True)  # Layer 1 weights
    b1 = torch.randn(16, requires_grad=True)      # Layer 1 bias
    W2 = torch.randn(1, 16, requires_grad=True)   # Output weights
    b2 = torch.randn(1, requires_grad=True)       # Output bias
    
    # Input financial data (batch_size=1 for simplicity)
    x = torch.randn(1, 30)  # Single sample of 30 financial indicators
    y_true = torch.tensor([[0.08]])  # True equity premium
    
    print("=== Forward Pass ===")
    
    # Layer 1: Linear + ReLU
    z1 = torch.mm(x, W1.t()) + b1  # Linear transformation
    h1 = torch.relu(z1)            # ReLU activation
    print(f"Layer 1 output shape: {h1.shape}")
    print(f"Active neurons in layer 1: {torch.sum(h1 > 0).item()}/16")
    
    # Output layer: Linear only
    z2 = torch.mm(h1, W2.t()) + b2  # Final prediction
    y_pred = z2
    print(f"Prediction: {y_pred.item():.4f}")
    print(f"True value: {y_true.item():.4f}")
    
    # Loss calculation
    loss = F.mse_loss(y_pred, y_true)
    print(f"MSE Loss: {loss.item():.6f}")
    
    print("\\n=== Backward Pass (Manual) ===")
    
    # Step 1: Gradient of loss w.r.t. prediction
    dL_dy_pred = 2 * (y_pred - y_true)
    print(f"∂L/∂ŷ = {dL_dy_pred.item():.6f}")
    
    # Step 2: Gradients for output layer
    dL_dW2 = torch.mm(dL_dy_pred, h1)  # Gradient w.r.t. W2
    dL_db2 = dL_dy_pred                # Gradient w.r.t. b2
    print(f"∂L/∂W2 shape: {dL_dW2.shape}")
    print(f"∂L/∂b2: {dL_db2.item():.6f}")
    
    # Step 3: Gradient w.r.t. hidden layer activations
    dL_dh1 = torch.mm(dL_dy_pred, W2)
    print(f"∂L/∂h1 max: {torch.max(torch.abs(dL_dh1)).item():.6f}")
    
    # Step 4: Gradient through ReLU
    dL_dz1 = dL_dh1.clone()
    dL_dz1[z1 <= 0] = 0  # ReLU derivative: 1 if z>0, 0 if z<=0
    active_gradients = torch.sum(dL_dz1 != 0).item()
    print(f"Gradients flowing through ReLU: {active_gradients}/16")
    
    # Step 5: Gradients for first layer
    dL_dW1 = torch.mm(dL_dz1.t(), x)
    dL_db1 = dL_dz1.squeeze()
    print(f"∂L/∂W1 shape: {dL_dW1.shape}")
    print(f"∂L/∂b1 non-zero: {torch.sum(dL_db1 != 0).item()}")
    
    print("\\n=== PyTorch Automatic Differentiation ===")
    
    # Reset gradients and use PyTorch autodiff
    if W1.grad is not None: W1.grad.zero_()
    if b1.grad is not None: b1.grad.zero_()
    if W2.grad is not None: W2.grad.zero_()
    if b2.grad is not None: b2.grad.zero_()
    
    loss.backward()
    
    print(f"PyTorch ∂L/∂W2 matches manual: {torch.allclose(dL_dW2, W2.grad)}")
    print(f"PyTorch ∂L/∂b2 matches manual: {torch.allclose(dL_db2, b2.grad)}")
    print(f"PyTorch ∂L/∂W1 matches manual: {torch.allclose(dL_dW1, W1.grad)}")
    print(f"PyTorch ∂L/∂b1 matches manual: {torch.allclose(dL_db1, b1.grad)}")

def visualize_gradient_flow():
    """
    Visualize how gradients flow through a neural network.
    """
    from src.models.nns import Net3
    
    # Create Net3 model
    model = Net3(n_feature=30, n_hidden1=64, n_hidden2=32, n_hidden3=16, 
                 n_output=1, dropout=0.0)  # No dropout for gradient analysis
    
    # Generate sample data
    x = torch.randn(4, 30)  # Small batch for clear visualization
    y_true = torch.randn(4, 1) * 0.02 + 0.08  # Target equity premiums
    
    # Forward pass
    model.train()
    y_pred = model(x)
    loss = F.mse_loss(y_pred, y_true)
    
    print("=== Gradient Flow Analysis ===")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y_pred.shape}")
    print(f"Loss: {loss.item():.6f}")
    
    # Backward pass
    loss.backward()
    
    # Analyze gradient magnitudes by layer
    layer_names = []
    gradient_norms = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = torch.norm(param.grad).item()
            layer_names.append(name)
            gradient_norms.append(grad_norm)
            print(f"{name:20s}: grad_norm = {grad_norm:.6f}")
    
    # Check for vanishing/exploding gradients
    max_grad = max(gradient_norms)
    min_grad = min([g for g in gradient_norms if g > 0])
    
    print(f"\\nGradient Analysis:")
    print(f"Max gradient norm: {max_grad:.6f}")
    print(f"Min gradient norm: {min_grad:.6f}")
    print(f"Gradient ratio: {max_grad/min_grad:.2f}")
    
    if max_grad/min_grad > 1000:
        print("⚠️  Potential exploding gradient problem")
    elif max_grad < 1e-6:
        print("⚠️  Potential vanishing gradient problem")
    else:
        print("✅ Healthy gradient flow")

def gradient_checking_example():
    """
    Numerical gradient checking to verify backpropagation implementation.
    """
    def finite_difference_check(model, x, y_true, epsilon=1e-5):
        """
        Compare analytical gradients with numerical approximation.
        """
        # Get analytical gradients
        model.zero_grad()
        loss = F.mse_loss(model(x), y_true)
        loss.backward()
        
        analytical_grads = []
        for param in model.parameters():
            if param.grad is not None:
                analytical_grads.append(param.grad.clone())
        
        # Compute numerical gradients
        numerical_grads = []
        param_idx = 0
        
        for param in model.parameters():
            if param.grad is not None:
                param_numerical_grad = torch.zeros_like(param)
                
                # For each parameter, compute numerical gradient
                it = torch.nditer(param.data.numpy(), flags=['multi_index'])
                while not it.finished:
                    idx = it.multi_index
                    
                    # f(x + h)
                    old_value = param.data[idx].item()
                    param.data[idx] = old_value + epsilon
                    loss_plus = F.mse_loss(model(x), y_true)
                    
                    # f(x - h)
                    param.data[idx] = old_value - epsilon
                    loss_minus = F.mse_loss(model(x), y_true)
                    
                    # Numerical gradient: (f(x+h) - f(x-h)) / (2*h)
                    param_numerical_grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)
                    
                    # Restore original value
                    param.data[idx] = old_value
                    it.iternext()
                
                numerical_grads.append(param_numerical_grad)
                param_idx += 1
        
        # Compare gradients
        print("=== Gradient Checking ===")
        for i, (analytical, numerical) in enumerate(zip(analytical_grads, numerical_grads)):
            relative_error = torch.norm(analytical - numerical) / (torch.norm(analytical) + torch.norm(numerical))
            print(f"Layer {i}: Relative error = {relative_error.item():.2e}")
            
            if relative_error < 1e-6:
                print(f"Layer {i}: ✅ Excellent match")
            elif relative_error < 1e-4:
                print(f"Layer {i}: ✅ Good match")
            else:
                print(f"Layer {i}: ❌ Poor match - check implementation")
    
    # Test on small network for feasibility
    small_model = nn.Sequential(
        nn.Linear(3, 4),
        nn.ReLU(),
        nn.Linear(4, 1)
    )
    
    x_small = torch.randn(1, 3)
    y_small = torch.randn(1, 1)
    
    finite_difference_check(small_model, x_small, y_small)

# Run examples
if __name__ == "__main__":
    print("Running manual backpropagation example...")
    manual_backpropagation_example()
    
    print("\\n" + "="*60 + "\\n")
    
    print("Running gradient flow visualization...")
    visualize_gradient_flow()
    
    print("\\n" + "="*60 + "\\n")
    
    print("Running gradient checking...")
    gradient_checking_example()`}
          />
        </motion.section>

        {/* Navigation */}
        <NavigationButtons 
          prevHref="/loss-calculation"
          prevLabel="Loss Calculation"
          nextHref="/optimization"
          nextLabel="Optimization"
        />
      </div>
    </div>
  )
}