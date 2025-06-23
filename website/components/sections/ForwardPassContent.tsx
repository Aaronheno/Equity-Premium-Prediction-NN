'use client'

import { motion } from 'framer-motion'
import { useState } from 'react'
import { Play, Zap, Settings, ArrowRight, Database, Target, Brain } from 'lucide-react'
import CodeBlock from '@/components/shared/CodeBlock'
import MathFormula from '@/components/shared/MathFormula'
import NavigationButtons from '@/components/shared/NavigationButtons'

const forwardPassSteps = [
  {
    step: 1,
    title: 'Linear Transformation',
    formula: String.raw`h = Wx + b`,
    icon: ArrowRight,
    color: 'accent-blue',
    description: 'Weighted combination of inputs with learned parameters',
    purpose: 'Core computation that transforms input features',
    details: 'Each neuron receives weighted contributions from all 32 financial indicators'
  },
  {
    step: 2,
    title: 'ReLU Activation',
    formula: String.raw`\text{output} = \max(0, \text{input})`,
    icon: Zap,
    color: 'accent-orange',
    description: 'Non-linear function that zeros negative values',
    purpose: 'Introduces non-linearity essential for complex pattern learning',
    details: 'Creates natural thresholds between signal and noise in financial data'
  },
  {
    step: 3,
    title: 'Dropout Regularization',
    formula: String.raw`y = x \odot \text{mask} / (1-p)`,
    icon: Settings,
    color: 'accent-purple',
    description: 'Randomly deactivates neurons during training',
    purpose: 'Prevents overfitting and improves generalization',
    details: 'Particularly valuable for noisy financial data and regime changes'
  }
]

const layerFlow = [
  { from: '32 Financial Indicators', to: '64 Hidden Units', layer: 'Layer 1' },
  { from: '64 Hidden Units', to: '32 Hidden Units', layer: 'Layer 2' },
  { from: '32 Hidden Units', to: '16 Hidden Units', layer: 'Layer 3' },
  { from: '16 Hidden Units', to: '1 Prediction', layer: 'Output' }
]

export default function ForwardPassContent() {
  const [activeStep, setActiveStep] = useState(1)

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
            <span className="gradient-text">Forward Pass</span>
          </h1>
          <p className="max-w-4xl mx-auto text-xl text-text-secondary leading-relaxed">
            Information flow through neural networks: how 32 financial indicators are transformed 
            through multiple layers to produce equity premium predictions.
          </p>
        </motion.div>

        {/* Forward Pass Overview */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Play className="w-8 h-8 text-accent-blue mr-3" />
            What is a Forward Pass?
          </h2>
          
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8 mb-8">
            <p className="text-text-secondary text-lg mb-6 leading-relaxed">
              The forward pass is the process where input data flows through the neural network to produce 
              a prediction. It represents the network's inference process - transforming financial data 
              into equity premium predictions without any learning.
            </p>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div>
                <h3 className="text-xl font-semibold text-text-primary mb-4">Key Characteristics:</h3>
                <div className="space-y-3 text-sm">
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-blue mt-2"></div>
                    <span className="text-text-secondary"><strong>Sequential processing:</strong> Data flows from input through hidden layers to output</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-blue mt-2"></div>
                    <span className="text-text-secondary"><strong>Deterministic computation:</strong> Same input + weights = same output</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-blue mt-2"></div>
                    <span className="text-text-secondary"><strong>Mathematical transformations:</strong> Linear operations + non-linear activations</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-blue mt-2"></div>
                    <span className="text-text-secondary"><strong>Dimension changes:</strong> Progressive transformation to final prediction</span>
                  </div>
                </div>
              </div>
              
              <div>
                <h3 className="text-xl font-semibold text-text-primary mb-4">In Financial Context:</h3>
                <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 space-y-2 text-sm">
                  <div><strong>Input:</strong> 32 standardized financial indicators</div>
                  <div><strong>Processing:</strong> Multiple hidden layers extract patterns</div>
                  <div><strong>Output:</strong> Single equity premium prediction</div>
                  <div className="border-t border-bg-tertiary pt-2">
                    <div className="text-accent-blue">Transforms market data → investment signal</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </motion.section>

        {/* Information Flow */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Database className="w-8 h-8 text-accent-green mr-3" />
            Information Flow Through Net3
          </h2>
          
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8 mb-8">
            <p className="text-text-secondary text-lg mb-8">
              Here's how information flows through a 3-layer network (Net3), showing the progressive 
              transformation from 32 financial indicators to 1 equity premium prediction.
            </p>

            <div className="space-y-6">
              {layerFlow.map((layer, index) => (
                <motion.div
                  key={layer.layer}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.6, delay: 0.4 + index * 0.1 }}
                  className="flex items-center space-x-6"
                >
                  <div className="flex-shrink-0 w-24 text-right text-sm text-text-muted">
                    {layer.layer}
                  </div>
                  
                  <div className="flex-1 flex items-center space-x-4">
                    <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 flex-1 text-center">
                      <div className="font-semibold text-text-primary">{layer.from}</div>
                    </div>
                    
                    <div className="flex-shrink-0">
                      <ArrowRight className="w-6 h-6 text-accent-blue" />
                    </div>
                    
                    <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 flex-1 text-center">
                      <div className="font-semibold text-text-primary">{layer.to}</div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>

            <div className="mt-8 bg-accent-green/5 border border-accent-green/20 rounded-lg p-6">
              <h3 className="text-xl font-semibold text-accent-green mb-4">Progressive Dimension Reduction</h3>
              <p className="text-text-secondary mb-4">
                Each layer reduces the number of features while extracting increasingly abstract patterns:
              </p>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div className="text-center">
                  <div className="font-mono text-accent-green">32 → 64</div>
                  <div className="text-text-muted">Feature expansion</div>
                </div>
                <div className="text-center">
                  <div className="font-mono text-accent-green">64 → 32</div>
                  <div className="text-text-muted">Pattern extraction</div>
                </div>
                <div className="text-center">
                  <div className="font-mono text-accent-green">32 → 16</div>
                  <div className="text-text-muted">Feature compression</div>
                </div>
                <div className="text-center">
                  <div className="font-mono text-accent-green">16 → 1</div>
                  <div className="text-text-muted">Final prediction</div>
                </div>
              </div>
            </div>
          </div>
        </motion.section>

        {/* Three-Step Process */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Brain className="w-8 h-8 text-accent-purple mr-3" />
            Three-Step Layer Process
          </h2>
          
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8 mb-8">
            <p className="text-text-secondary text-lg mb-8">
              Each hidden layer performs three critical operations in sequence. Click each step 
              to explore the mathematical details and financial interpretation.
            </p>

            {/* Step Selector */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
              {forwardPassSteps.map((step) => {
                const Icon = step.icon
                const isActive = activeStep === step.step
                return (
                  <button
                    key={step.step}
                    onClick={() => setActiveStep(step.step)}
                    className={`p-4 rounded-lg border transition-all text-left ${
                      isActive 
                        ? `bg-${step.color}/10 border-${step.color}/30` 
                        : 'bg-bg-primary border-bg-tertiary hover:border-accent-blue/30'
                    }`}
                  >
                    <div className="flex items-center space-x-3 mb-2">
                      <div className={`inline-flex items-center justify-center w-8 h-8 rounded-lg bg-${step.color}/10 border border-${step.color}/20`}>
                        <Icon className={`w-4 h-4 text-${step.color}`} />
                      </div>
                      <div>
                        <div className="text-xs text-text-muted">Step {step.step}</div>
                        <div className="font-semibold text-text-primary">{step.title}</div>
                      </div>
                    </div>
                    <MathFormula latex={step.formula} />
                  </button>
                )
              })}
            </div>

            {/* Active Step Details */}
            {forwardPassSteps.map((step) => {
              const Icon = step.icon
              if (activeStep !== step.step) return null
              
              return (
                <motion.div
                  key={step.step}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                  className={`bg-${step.color}/5 border border-${step.color}/20 rounded-xl p-8`}
                >
                  <div className="flex items-center space-x-3 mb-6">
                    <div className={`inline-flex items-center justify-center w-12 h-12 rounded-lg bg-${step.color}/10 border border-${step.color}/20`}>
                      <Icon className={`w-6 h-6 text-${step.color}`} />
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
                      
                      <div className={`bg-${step.color}/10 border border-${step.color}/20 rounded-lg p-4`}>
                        <div className={`text-${step.color} text-sm font-medium mb-1`}>Financial Context:</div>
                        <div className="text-text-secondary text-sm">{step.details}</div>
                      </div>
                    </div>
                    
                    <div>
                      {step.step === 1 && (
                        <div>
                          <h4 className="font-semibold text-text-primary mb-3">Example Calculation:</h4>
                          <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 space-y-2 text-sm font-mono">
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
                      )}
                      
                      {step.step === 2 && (
                        <div>
                          <h4 className="font-semibold text-text-primary mb-3">Example Transformations:</h4>
                          <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 space-y-2 text-sm font-mono">
                            <div>If h = 1.23 → ReLU(h) = 1.23 <span className="text-accent-green">(positive passes through)</span></div>
                            <div>If h = -0.45 → ReLU(h) = 0.0 <span className="text-accent-red">(negative becomes zero)</span></div>
                            <div>If h = 0.0 → ReLU(h) = 0.0 <span className="text-accent-blue">(zero stays zero)</span></div>
                          </div>
                        </div>
                      )}
                      
                      {step.step === 3 && (
                        <div>
                          <h4 className="font-semibold text-text-primary mb-3">Training vs Inference:</h4>
                          <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 space-y-2 text-sm">
                            <div><strong>Training:</strong> Random neurons set to 0 (with probability p)</div>
                            <div><strong>Inference:</strong> All neurons active, no dropout applied</div>
                            <div><strong>Scaling:</strong> Outputs scaled by 1/(1-p) during training</div>
                            <div className="text-accent-purple"><strong>Typical p:</strong> 0.3 for financial data (30% dropout)</div>
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

        {/* Implementation Example */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.5 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Target className="w-8 h-8 text-accent-blue mr-3" />
            Forward Pass Implementation
          </h2>
          
          <p className="text-text-secondary text-lg mb-6">
            Here's how the forward pass is implemented in PyTorch, showing the sequential 
            transformation of financial data through Net3's three hidden layers.
          </p>

          <CodeBlock
            language="python"
            title="Conceptual: Step-by-Step Forward Pass"
            codeType="conceptual"
            actualImplementationPath="src/models/nns.py"
            code={`# Educational example showing what happens inside each layer

def net3_forward_pass_example():
    """
    Demonstrate forward pass through Net3 with actual financial data.
    """
    
    # Simulated financial indicators (standardized)
    financial_indicators = torch.tensor([
        [0.5, -1.2, 0.8, 0.0, -0.5, 0.3, -0.2, 1.1, 0.7, -0.4,  # 10 indicators
         0.2, -0.8, 1.3, 0.6, -0.3, 0.9, -0.1, 0.4, -0.7, 0.8,  # 10 more
         -0.6, 0.1, 0.5, -0.9, 1.2, 0.3, -0.5, 0.7, 0.2, -0.4]  # Final 10
    ], dtype=torch.float32)  # Shape: [1, 30] (1 sample, 30 features)
    
    print(f"Input shape: {financial_indicators.shape}")
    print(f"Sample input: {financial_indicators[0][:5]}...")  # First 5 indicators
    
    # Net3 architecture: 30 → 64 → 32 → 16 → 1
    
    # Layer 1: 30 → 64
    linear1 = nn.Linear(30, 64)
    h1_linear = linear1(financial_indicators)
    print(f"\\nLayer 1 linear output shape: {h1_linear.shape}")
    print(f"Sample values: {h1_linear[0][:5]}")
    
    h1_activated = torch.relu(h1_linear)
    print(f"After ReLU: {h1_activated[0][:5]}")
    
    h1_dropout = torch.dropout(h1_activated, p=0.3, training=True)
    print(f"After dropout: {h1_dropout[0][:5]}")
    
    # Layer 2: 64 → 32  
    linear2 = nn.Linear(64, 32)
    h2_linear = linear2(h1_dropout)
    h2_activated = torch.relu(h2_linear)
    h2_dropout = torch.dropout(h2_activated, p=0.3, training=True)
    
    print(f"\\nLayer 2 output shape: {h2_dropout.shape}")
    print(f"Sample values: {h2_dropout[0][:5]}")
    
    # Layer 3: 32 → 16
    linear3 = nn.Linear(32, 16)
    h3_linear = linear3(h2_dropout)
    h3_activated = torch.relu(h3_linear)
    h3_dropout = torch.dropout(h3_activated, p=0.3, training=True)
    
    print(f"\\nLayer 3 output shape: {h3_dropout.shape}")
    print(f"Sample values: {h3_dropout[0][:5]}")
    
    # Output layer: 16 → 1 (no activation, no dropout)
    output_linear = nn.Linear(16, 1)
    prediction = output_linear(h3_dropout)
    
    print(f"\\nFinal prediction shape: {prediction.shape}")
    print(f"Equity premium prediction: {prediction.item():.6f}")
    
    return prediction

# Actual Net3 class implementation
class Net3(nn.Module):
    def __init__(self, n_feature=30, n_hidden1=64, n_hidden2=32, n_hidden3=16, 
                 n_output=1, dropout=0.3):
        super().__init__()
        
        self.layers = nn.Sequential(
            # Layer 1
            nn.Linear(n_feature, n_hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 2
            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU(), 
            nn.Dropout(dropout),
            
            # Layer 3
            nn.Linear(n_hidden2, n_hidden3),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Output layer
            nn.Linear(n_hidden3, n_output)
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, 30]
            
        Returns:
            Predictions of shape [batch_size, 1]
        """
        return self.layers(x)

# Usage example
model = Net3()
financial_data = torch.randn(32, 30)  # Batch of 32 samples
predictions = model(financial_data)
print(f"Batch predictions shape: {predictions.shape}")  # [32, 1]`}
          />

          {/* Actual Implementation */}
          <div className="mt-8">
            <h3 className="text-xl font-semibold text-text-primary mb-4">Actual Implementation</h3>
            <p className="text-text-secondary mb-6">
              Here's the actual implementation from our codebase, which uses a modular architecture 
              to support all 8 model variants (Net1-Net5, DNet1-DNet3) efficiently:
            </p>
            
            <CodeBlock
              language="python"
              title="Actual Net3 Implementation"
              codeType="actual"
              code={`# From src/models/nns.py

class _Base(nn.Module):
    """Base class for all standard neural networks (Net1-Net5)."""
    def __init__(self, layers, dropout=0.0, act="relu"):
        super().__init__()
        act_fn = _ACT[act.lower()]  # Map string to activation function
        seq = []
        for i in range(len(layers) - 2):
            seq.extend([nn.Linear(layers[i], layers[i+1]), act_fn, nn.Dropout(dropout)])
        seq.append(nn.Linear(layers[-2], layers[-1]))  # Output layer (no activation)
        self.net = nn.Sequential(*seq)

    def forward(self, x):
        return self.net(x)

class Net3(_Base):
    """3-layer neural network for equity premium prediction."""
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_hidden3, n_output, 
                 dropout=0.0, activation_hidden="relu", **kw):
        # Pass layer sizes to base class
        super().__init__([n_feature, n_hidden1, n_hidden2, n_hidden3, n_output], 
                         dropout, activation_hidden)

# Activation function mapping
_ACT = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "selu": nn.SELU(),
    "elu": nn.ELU(),
}`}
            />
            
            <div className="bg-accent-green/10 border border-accent-green/20 rounded-lg p-4 mt-4">
              <h4 className="font-semibold text-accent-green mb-2">Key Design Decisions:</h4>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>• <strong>Sequential Module:</strong> PyTorch's Sequential automatically chains operations</li>
                <li>• <strong>Modular Architecture:</strong> Single base class handles Net1 through Net5</li>
                <li>• <strong>No Output Activation:</strong> Linear output for regression tasks</li>
                <li>• <strong>Configurable Activation:</strong> ReLU is default but other options available</li>
              </ul>
            </div>
          </div>
        </motion.section>

        {/* Navigation */}
        <NavigationButtons 
          prevHref="/architecture"
          prevLabel="Model Architecture"
          nextHref="/loss-calculation"
          nextLabel="Loss Calculation"
        />
      </div>
    </div>
  )
}