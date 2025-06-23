'use client'

import { motion } from 'framer-motion'
import { useState } from 'react'
import { Layers, Settings, Zap, GitBranch, Code2, Brain, Network, Target, Play, ChevronDown, ChevronRight } from 'lucide-react'
import CodeBlock from '@/components/shared/CodeBlock'
import MathFormula from '@/components/shared/MathFormula'
import NavigationButtons from '@/components/shared/NavigationButtons'

const modelTypes = [
  {
    category: 'Standard Neural Networks',
    description: 'Traditional feedforward architectures with ReLU activation and dropout',
    models: [
      { name: 'Net1', layers: 1, description: 'Simplest architecture for basic pattern detection', neurons: '16-256' },
      { name: 'Net2', layers: 2, description: 'Two hidden layers for moderate complexity', neurons: '16-192, 8-128' },
      { name: 'Net3', layers: 3, description: 'Three layers for standard financial modeling', neurons: '16-128, 8-96, 4-64' },
      { name: 'Net4', layers: 4, description: 'Four layers for complex pattern recognition', neurons: '32-192, 16-128, 8-96, 4-64' },
      { name: 'Net5', layers: 5, description: 'Deepest standard architecture', neurons: '32-256, 16-192, 8-128, 8-96, 4-64' }
    ],
    color: 'accent-blue'
  },
  {
    category: 'Deep Neural Networks',
    description: 'Enhanced architectures with Batch Normalization for improved training stability',
    models: [
      { name: 'DNet1', layers: 4, description: 'BatchNorm-enhanced 4-layer network', neurons: '64-384, 32-256, 16-192, 16-128' },
      { name: 'DNet2', layers: 5, description: 'Five layers with batch normalization', neurons: '64-384, 48-256, 32-192, 24-128, 12-64' },
      { name: 'DNet3', layers: 5, description: 'Largest architecture for maximum capacity', neurons: '128-512, 64-384, 48-256, 32-192, 16-128' }
    ],
    color: 'accent-purple'
  }
]

const architectureFeatures = [
  {
    title: 'Feedforward Information Flow',
    icon: GitBranch,
    description: 'Information flows sequentially from input through hidden layers to output',
    benefit: 'Clear signal propagation and efficient computation'
  },
  {
    title: 'Progressive Dimension Reduction',
    icon: Layers,
    description: 'Each layer reduces feature dimensions through learned transformations',
    benefit: 'Hierarchical feature learning from simple to complex patterns'
  },
  {
    title: 'ReLU Activation Functions',
    icon: Zap,
    description: 'Non-linear activations enable learning of complex relationships',
    benefit: 'Computational efficiency and stable gradient flow'
  },
  {
    title: 'Dropout Regularization',
    icon: Settings,
    description: 'Random neuron deactivation prevents overfitting during training',
    benefit: 'Improved generalization to unseen financial data'
  }
]

const layerOperations = [
  {
    step: 1,
    operation: 'Linear Transformation',
    formula: 'h = Wx + b',
    description: 'Weighted combination of inputs with learned parameters',
    purpose: 'Core computation that transforms input features'
  },
  {
    step: 2,
    operation: 'ReLU Activation',
    formula: 'output = max(0, input)',
    description: 'Non-linear function that zeros negative values',
    purpose: 'Introduces non-linearity essential for complex pattern learning'
  },
  {
    step: 3,
    operation: 'Dropout Regularization',
    formula: 'randomly set neurons to 0',
    description: 'Randomly deactivates neurons during training',
    purpose: 'Prevents overfitting and improves generalization'
  }
]

export default function ArchitectureContent() {
  const [activeModel, setActiveModel] = useState('Net3')
  const [expandedSection, setExpandedSection] = useState<string | null>('standard')

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
            <span className="gradient-text">Neural Network Architecture</span>
          </h1>
          <p className="max-w-4xl mx-auto text-xl text-text-secondary leading-relaxed">
            Comprehensive guide to 8 neural network architectures optimized for equity premium prediction: 
            from simple feedforward networks to advanced BatchNorm-enhanced models.
          </p>
        </motion.div>

        {/* Architecture Overview */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Brain className="w-8 h-8 text-accent-blue mr-3" />
            Architecture Overview
          </h2>
          
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8 mb-8">
            <p className="text-text-secondary text-lg mb-6 leading-relaxed">
              The neural networks employ feedforward architectures where financial indicators flow through 
              multiple hidden layers to produce equity premium predictions. Each architecture is optimized 
              for different complexity levels and computational requirements.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {architectureFeatures.map((feature, index) => {
                const Icon = feature.icon
                return (
                  <motion.div
                    key={feature.title}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6, delay: 0.3 + index * 0.1 }}
                    className="bg-bg-primary border border-bg-tertiary rounded-lg p-6"
                  >
                    <div className="flex items-center space-x-3 mb-4">
                      <div className="inline-flex items-center justify-center w-10 h-10 rounded-lg bg-accent-blue/10 border border-accent-blue/20">
                        <Icon className="w-5 h-5 text-accent-blue" />
                      </div>
                      <h3 className="font-semibold text-text-primary text-sm">{feature.title}</h3>
                    </div>
                    <p className="text-text-secondary text-sm mb-3">{feature.description}</p>
                    <div className="bg-accent-blue/5 border border-accent-blue/20 rounded p-3">
                      <p className="text-accent-blue text-xs font-medium">{feature.benefit}</p>
                    </div>
                  </motion.div>
                )
              })}
            </div>
          </div>
        </motion.section>

        {/* Model Categories */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Network className="w-8 h-8 text-accent-purple mr-3" />
            8 Neural Network Architectures
          </h2>
          
          <div className="space-y-8">
            {modelTypes.map((category, categoryIndex) => (
              <div key={category.category}>
                <button
                  onClick={() => setExpandedSection(expandedSection === category.category.toLowerCase().replace(' ', '') ? null : category.category.toLowerCase().replace(' ', ''))}
                  className="w-full flex items-center justify-between p-6 bg-bg-secondary border border-bg-tertiary rounded-xl hover:border-accent-blue/30 transition-all"
                >
                  <div className="flex items-center space-x-4">
                    <div className={`inline-flex items-center justify-center w-12 h-12 rounded-lg bg-${category.color}/10 border border-${category.color}/20`}>
                      <Brain className={`w-6 h-6 text-${category.color}`} />
                    </div>
                    <div className="text-left">
                      <h3 className="text-2xl font-bold text-text-primary">{category.category}</h3>
                      <p className="text-text-secondary">{category.description}</p>
                    </div>
                  </div>
                  {expandedSection === category.category.toLowerCase().replace(' ', '') ? 
                    <ChevronDown className="w-6 h-6 text-text-secondary" /> : 
                    <ChevronRight className="w-6 h-6 text-text-secondary" />
                  }
                </button>

                {expandedSection === category.category.toLowerCase().replace(' ', '') && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    transition={{ duration: 0.3 }}
                    className="mt-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4"
                  >
                    {category.models.map((model, modelIndex) => (
                      <motion.div
                        key={model.name}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.4, delay: modelIndex * 0.1 }}
                        onClick={() => setActiveModel(model.name)}
                        className={`cursor-pointer p-6 rounded-lg border transition-all ${
                          activeModel === model.name
                            ? `bg-${category.color}/10 border-${category.color}/30`
                            : 'bg-bg-primary border-bg-tertiary hover:border-accent-blue/30'
                        }`}
                      >
                        <div className="flex items-center space-x-3 mb-3">
                          <div className={`inline-flex items-center justify-center w-8 h-8 rounded bg-${category.color}/20 text-${category.color} font-mono text-sm font-bold`}>
                            {model.name}
                          </div>
                          <div>
                            <div className="font-semibold text-text-primary text-sm">{model.layers} Hidden Layer{model.layers > 1 ? 's' : ''}</div>
                            <div className="text-text-muted text-xs">{model.neurons}</div>
                          </div>
                        </div>
                        <p className="text-text-secondary text-sm">{model.description}</p>
                      </motion.div>
                    ))}
                  </motion.div>
                )}
              </div>
            ))}
          </div>
        </motion.section>

        {/* Layer-by-Layer Construction */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Layers className="w-8 h-8 text-accent-green mr-3" />
            Layer-by-Layer Construction
          </h2>
          
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8 mb-8">
            <p className="text-text-secondary text-lg mb-8">
              Each hidden layer performs three critical operations in sequence. This three-step process 
              transforms financial data through increasingly abstract representations.
            </p>

            <div className="space-y-6">
              {layerOperations.map((operation, index) => (
                <motion.div
                  key={operation.step}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.6, delay: 0.5 + index * 0.1 }}
                  className="flex items-start space-x-6"
                >
                  <div className="flex-shrink-0 inline-flex items-center justify-center w-12 h-12 rounded-lg bg-accent-green/10 border border-accent-green/20">
                    <span className="text-accent-green font-bold text-lg">{operation.step}</span>
                  </div>
                  
                  <div className="flex-1">
                    <div className="flex items-center space-x-4 mb-3">
                      <h3 className="text-xl font-semibold text-text-primary">{operation.operation}</h3>
                      <code className="bg-code-bg px-3 py-1 rounded font-mono text-sm">{operation.formula}</code>
                    </div>
                    <p className="text-text-secondary mb-2">{operation.description}</p>
                    <div className="bg-accent-green/5 border border-accent-green/20 rounded-lg p-3">
                      <p className="text-accent-green text-sm font-medium">Purpose: {operation.purpose}</p>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </motion.section>

        {/* Mathematical Foundation */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.5 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Target className="w-8 h-8 text-accent-orange mr-3" />
            Mathematical Foundation
          </h2>
          
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8">
            <h3 className="text-xl font-semibold text-text-primary mb-6">Linear Transformation: The Core Operation</h3>
            
            <div className="mb-8">
              <MathFormula latex="h = Wx + b" block={true} />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
              <div>
                <h4 className="font-semibold text-text-primary mb-4">Components:</h4>
                <div className="space-y-3 text-sm">
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
                <h4 className="font-semibold text-text-primary mb-4">Example Calculation:</h4>
                <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 space-y-2 text-sm">
                  <div><strong>Input:</strong> [Dividend Yield=0.02, Term Spread=0.01, Inflation=0.03]</div>
                  <div><strong>Weights:</strong> [0.5, 0.3, -0.2]</div>
                  <div><strong>Bias:</strong> 0.1</div>
                  <div className="border-t border-bg-tertiary pt-2">
                    <strong>Calculation:</strong> h = (0.02×0.5) + (0.01×0.3) + (0.03×-0.2) + 0.1 = 0.097
                  </div>
                </div>
              </div>
            </div>

            <h3 className="text-xl font-semibold text-text-primary mb-6">Information Flow Through Net3</h3>
            
            <div className="bg-accent-orange/5 border border-accent-orange/20 rounded-lg p-6">
              <div className="space-y-4 text-sm">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <strong className="text-text-primary">Layer 1:</strong> 30 indicators → 64 hidden units<br/>
                    <code className="text-accent-orange">h₁ = ReLU(W₁·X + b₁)</code>
                  </div>
                  <div>
                    <strong className="text-text-primary">Layer 2:</strong> 64 → 32 hidden units<br/>
                    <code className="text-accent-orange">h₂ = ReLU(W₂·h₁ + b₂)</code>
                  </div>
                  <div>
                    <strong className="text-text-primary">Layer 3:</strong> 32 → 16 hidden units<br/>
                    <code className="text-accent-orange">h₃ = ReLU(W₃·h₂ + b₃)</code>
                  </div>
                  <div>
                    <strong className="text-text-primary">Output:</strong> 16 → 1 prediction<br/>
                    <code className="text-accent-orange">ŷ = W₄·h₃ + b₄</code>
                  </div>
                </div>
              </div>
            </div>
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
            <Code2 className="w-8 h-8 text-accent-blue mr-3" />
            Implementation: Building Net3
          </h2>
          
          <p className="text-text-secondary text-lg mb-6">
            Here's how Net3 is constructed step by step, showing the three-layer architecture 
            with progressive dimension reduction from 30 financial indicators to 1 prediction.
          </p>

          <CodeBlock
            language="python"
            title="Net3 Architecture Implementation"
            code={`# From src/models/nns.py - Base class for all NN models

class _Base(nn.Module):
    def __init__(self, layers, dropout=0.0, act="relu"):
        super().__init__()
        act_fn = _ACT[act.lower()]  # Maps string to activation function
        seq = []
        
        # Build hidden layers: Linear → Activation → Dropout
        for i in range(len(layers) - 2):
            seq.extend([
                nn.Linear(layers[i], layers[i+1]),  # Linear transformation
                act_fn,                             # Activation function  
                nn.Dropout(dropout)                 # Regularization
            ])
        
        # Output layer: Linear only (no activation)
        seq.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*seq)
    
    def forward(self, x):
        return self.net(x)

# Net3 Implementation
class Net3(_Base):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_hidden3, n_output, 
                 dropout=0.0, activation_hidden="relu", **kw):
        # Architecture: Input(30) → Hidden₁ → Hidden₂ → Hidden₃ → Output(1)
        super().__init__([n_feature, n_hidden1, n_hidden2, n_hidden3, n_output], 
                         dropout, activation_hidden)

# Manual construction example for understanding
def build_net3_manually(n_features=30, n_hidden1=64, n_hidden2=32, n_hidden3=16, dropout=0.2):
    """
    Manually construct Net3 to show the layer-by-layer process.
    """
    layers = []
    
    # Input → Hidden Layer 1
    layers.append(nn.Linear(n_features, n_hidden1))    # 30 → 64 neurons
    layers.append(nn.ReLU())                           # Non-linearity
    layers.append(nn.Dropout(dropout))                 # Regularization
    
    # Hidden Layer 1 → Hidden Layer 2
    layers.append(nn.Linear(n_hidden1, n_hidden2))    # 64 → 32 neurons
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(dropout))
    
    # Hidden Layer 2 → Hidden Layer 3
    layers.append(nn.Linear(n_hidden2, n_hidden3))    # 32 → 16 neurons
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(dropout))
    
    # Hidden Layer 3 → Output
    layers.append(nn.Linear(n_hidden3, 1))            # 16 → 1 prediction
    # No activation on output for regression
    
    return nn.Sequential(*layers)

# Usage example
model = build_net3_manually()
print(model)

# Output:
# Sequential(
#   (0): Linear(in_features=30, out_features=64, bias=True)
#   (1): ReLU()
#   (2): Dropout(p=0.2, inplace=False)
#   (3): Linear(in_features=64, out_features=32, bias=True)
#   (4): ReLU()
#   (5): Dropout(p=0.2, inplace=False)
#   (6): Linear(in_features=32, out_features=16, bias=True)
#   (7): ReLU()
#   (8): Dropout(p=0.2, inplace=False)
#   (9): Linear(in_features=16, out_features=1, bias=True)
# )`}
          />
        </motion.section>

        {/* Deep Networks with BatchNorm */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.7 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Settings className="w-8 h-8 text-accent-purple mr-3" />
            Deep Networks with Batch Normalization
          </h2>
          
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8 mb-8">
            <p className="text-text-secondary text-lg mb-6">
              DNet models enhance the standard architecture with Batch Normalization, enabling 
              deeper networks with improved training stability and performance.
            </p>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
              <div>
                <h3 className="text-xl font-semibold text-text-primary mb-4">Standard NN Architecture:</h3>
                <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 font-mono text-sm">
                  Input → Linear → ReLU → Dropout → Output
                </div>
              </div>
              <div>
                <h3 className="text-xl font-semibold text-text-primary mb-4">DNN Architecture:</h3>
                <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 font-mono text-sm">
                  Input → Linear → <span className="text-accent-purple">BatchNorm</span> → ReLU → Dropout → Output
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <h4 className="font-semibold text-accent-purple">Benefits of BatchNorm:</h4>
                <div className="space-y-3 text-sm">
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-purple mt-2"></div>
                    <span className="text-text-secondary"><strong>Training Stability:</strong> Prevents internal covariate shift</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-purple mt-2"></div>
                    <span className="text-text-secondary"><strong>Faster Convergence:</strong> Enables higher learning rates</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-purple mt-2"></div>
                    <span className="text-text-secondary"><strong>Better Generalization:</strong> Acts as implicit regularization</span>
                  </div>
                </div>
              </div>
              <div className="space-y-4">
                <h4 className="font-semibold text-accent-blue">DNet Model Specifications:</h4>
                <div className="space-y-2 text-sm">
                  <div><strong>DNet1:</strong> 4 layers, 64-384 to 16-128 neurons</div>
                  <div><strong>DNet2:</strong> 5 layers, 64-384 to 12-64 neurons</div>
                  <div><strong>DNet3:</strong> 5 layers, 128-512 to 16-128 neurons</div>
                </div>
              </div>
            </div>
          </div>

          <CodeBlock
            language="python"
            title="DBlock Implementation with BatchNorm"
            code={`# From src/models/nns.py - Enhanced building block with BatchNorm

class DBlock(nn.Module):
    def __init__(self, n_in, n_out, activation_fn_name="relu", dropout_rate=0.0, use_batch_norm=True):
        super().__init__()
        self.linear = nn.Linear(n_in, n_out)
        
        # Activation function selection
        if activation_fn_name.lower() == "relu":
            self.activation_fn = nn.ReLU()
        elif activation_fn_name.lower() == "leaky_relu":
            self.activation_fn = nn.LeakyReLU()
        
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.bn = nn.BatchNorm1d(n_out) if use_batch_norm else None

    def forward(self, x):
        # Step 1: Linear transformation
        x = self.linear(x)  # Wx + b
        
        # Step 2: Batch Normalization (before activation)
        if self.bn:
            # Handle edge case: batch_size = 1 during training
            if x.size(0) > 1 or not self.training:
                x = self.bn(x)
            # Skip BatchNorm for single samples during training to avoid error
        
        # Step 3: Activation function
        x = self.activation_fn(x)
        
        # Step 4: Dropout (if specified)
        if self.dropout:
            x = self.dropout(x)
            
        return x

# DNet1 Implementation
class DNet1(nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_hidden3, n_hidden4, 
                 n_output, dropout=0.1, activation_fn="relu", **kw):
        super().__init__()
        self.blocks = nn.Sequential(
            DBlock(n_feature, n_hidden1, activation_fn, dropout),    # Input → Hidden₁
            DBlock(n_hidden1, n_hidden2, activation_fn, dropout),    # Hidden₁ → Hidden₂
            DBlock(n_hidden2, n_hidden3, activation_fn, dropout),    # Hidden₂ → Hidden₃
            DBlock(n_hidden3, n_hidden4, activation_fn, dropout),    # Hidden₃ → Hidden₄
        )
        self.out = nn.Linear(n_hidden4, n_output)  # Final output layer (no BatchNorm)
    
    def forward(self, x):
        return self.out(self.blocks(x))`}
          />
        </motion.section>

        {/* Architecture Comparison */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.8 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Play className="w-8 h-8 text-accent-green mr-3" />
            Architecture Comparison & Selection
          </h2>
          
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-bg-tertiary">
                    <th className="text-left py-3 text-text-primary">Aspect</th>
                    <th className="text-left py-3 text-text-primary">NN Models (Net1-5)</th>
                    <th className="text-left py-3 text-text-primary">DNN Models (DNet1-3)</th>
                  </tr>
                </thead>
                <tbody className="space-y-2">
                  <tr className="border-b border-bg-tertiary">
                    <td className="py-3 font-medium text-text-primary">Normalization</td>
                    <td className="py-3 text-text-secondary">None</td>
                    <td className="py-3 text-text-secondary">BatchNorm after each linear layer</td>
                  </tr>
                  <tr className="border-b border-bg-tertiary">
                    <td className="py-3 font-medium text-text-primary">Architecture</td>
                    <td className="py-3 text-text-secondary">1-5 layers, simpler structure</td>
                    <td className="py-3 text-text-secondary">4-5 layers with enhanced building blocks</td>
                  </tr>
                  <tr className="border-b border-bg-tertiary">
                    <td className="py-3 font-medium text-text-primary">Neuron Counts</td>
                    <td className="py-3 text-text-secondary">4-256 per layer</td>
                    <td className="py-3 text-text-secondary">12-512 per layer</td>
                  </tr>
                  <tr className="border-b border-bg-tertiary">
                    <td className="py-3 font-medium text-text-primary">Training Stability</td>
                    <td className="py-3 text-text-secondary">Standard gradient flow</td>
                    <td className="py-3 text-text-secondary">Enhanced stability via BatchNorm</td>
                  </tr>
                  <tr className="border-b border-bg-tertiary">
                    <td className="py-3 font-medium text-text-primary">Computational Cost</td>
                    <td className="py-3 text-text-secondary">Lower (simpler operations)</td>
                    <td className="py-3 text-text-secondary">Higher (additional normalization overhead)</td>
                  </tr>
                  <tr>
                    <td className="py-3 font-medium text-text-primary">Best Use Case</td>
                    <td className="py-3 text-text-secondary">Moderate complexity patterns</td>
                    <td className="py-3 text-text-secondary">Complex, noisy financial relationships</td>
                  </tr>
                </tbody>
              </table>
            </div>

          </div>
        </motion.section>

        {/* Implementation Details */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.8 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Code2 className="w-8 h-8 text-accent-green mr-3" />
            Actual Implementation
          </h2>
          
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8 mb-8">
            <p className="text-text-secondary text-lg mb-8">
              Here's how the neural network architectures are actually implemented in the codebase, 
              showing the modular design that supports all 8 model variants efficiently.
            </p>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
              {/* Standard NN Implementation */}
              <div>
                <h3 className="text-xl font-semibold text-text-primary mb-4">Standard Networks (Net1-Net5)</h3>
                <CodeBlock
                  language="python"
                  title="Base Class for Standard Networks"
                  codeType="actual"
                  code={`# From src/models/nns.py

class _Base(nn.Module):
    def __init__(self, layers, dropout=0.0, act="relu"):
        super().__init__()
        act_fn = _ACT[act.lower()]
        seq = []
        for i in range(len(layers) - 2):
            seq.extend([
                nn.Linear(layers[i], layers[i+1]), 
                act_fn, 
                nn.Dropout(dropout)
            ])
        seq.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*seq)

    def forward(self, x):
        return self.net(x)

# Example: Net3 inherits from _Base
class Net3(_Base):
    def __init__(self, n_feature, n_hidden1, n_hidden2, 
                 n_hidden3, n_output, dropout=0.0, 
                 activation_hidden="relu", **kw):
        super().__init__(
            [n_feature, n_hidden1, n_hidden2, n_hidden3, n_output], 
            dropout, activation_hidden
        )`}
                />
              </div>

              {/* Deep NN Implementation */}
              <div>
                <h3 className="text-xl font-semibold text-text-primary mb-4">Deep Networks (DNet1-DNet3)</h3>
                <CodeBlock
                  language="python"
                  title="DBlock with BatchNorm"
                  codeType="actual"
                  code={`# From src/models/nns.py

class DBlock(nn.Module):
    def __init__(self, n_in, n_out, activation_fn_name="relu", 
                 dropout_rate=0.0, use_batch_norm=True):
        super().__init__()
        self.linear = nn.Linear(n_in, n_out)
        
        # Activation function
        if activation_fn_name.lower() == "relu":
            self.activation_fn = nn.ReLU()
        # ... other activations
        
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.bn = nn.BatchNorm1d(n_out) if use_batch_norm else None

    def forward(self, x):
        x = self.linear(x)
        
        # Apply BatchNorm before activation
        if self.bn:
            if x.size(0) > 1 or not self.training:
                x = self.bn(x)
        
        x = self.activation_fn(x)
        
        if self.dropout:
            x = self.dropout(x)
        return x`}
                />
              </div>
            </div>

            {/* All Model Definitions */}
            <div className="bg-bg-primary border border-bg-tertiary rounded-xl p-6">
              <h3 className="text-xl font-semibold text-text-primary mb-4">Complete Model Definitions</h3>
              <CodeBlock
                language="python"
                title="All 8 Neural Network Models"
                codeType="actual"
                code={`# From src/models/nns.py - All model exports

__all__ = [
    "Net1", "Net2", "Net3", "Net4", "Net5",
    "DNet1", "DNet2", "DNet3"
]

# Standard networks (1-5 hidden layers)
class Net1(_Base):
    def __init__(self, n_feature, n_output=1, n_hidden1=64, 
                 activation_hidden='relu', dropout=0.1):
        super().__init__([n_feature, n_hidden1, n_output], dropout, activation_hidden)

class Net2(_Base):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output, 
                 dropout=0.0, activation_hidden="relu", **kw):
        super().__init__([n_feature, n_hidden1, n_hidden2, n_output], dropout, activation_hidden)

# ... Net3, Net4, Net5 follow similar pattern

# Deep networks with BatchNorm
class DNet1(nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_hidden3, n_hidden4, 
                 n_output, dropout=0.1, activation_fn="relu", **kw):
        super().__init__()
        self.blocks = nn.Sequential(
            DBlock(n_feature, n_hidden1, activation_fn, dropout),
            DBlock(n_hidden1, n_hidden2, activation_fn, dropout),
            DBlock(n_hidden2, n_hidden3, activation_fn, dropout),
            DBlock(n_hidden3, n_hidden4, activation_fn, dropout),
        )
        self.out = nn.Linear(n_hidden4, n_output)
    
    def forward(self, x):
        return self.out(self.blocks(x))

# DNet2 and DNet3 follow similar pattern with 5 layers`}
              />
              
              <div className="mt-6 bg-accent-green/10 border border-accent-green/20 rounded-lg p-4">
                <h4 className="font-semibold text-accent-green mb-3">Architecture Design Benefits:</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                  <div>
                    <h5 className="font-medium text-text-primary mb-2">Code Reusability:</h5>
                    <ul className="space-y-1 text-text-secondary">
                      <li>• Single _Base class for Net1-Net5</li>
                      <li>• Reusable DBlock for all deep networks</li>
                      <li>• Consistent parameter interfaces</li>
                    </ul>
                  </div>
                  <div>
                    <h5 className="font-medium text-text-primary mb-2">Hyperparameter Optimization:</h5>
                    <ul className="space-y-1 text-text-secondary">
                      <li>• Easy to swap between architectures</li>
                      <li>• Consistent naming conventions</li>
                      <li>• Supports automated model selection</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </motion.section>

        {/* Navigation */}
        <NavigationButtons 
          prevHref="/preprocessing"
          prevLabel="Data Preprocessing"
          nextHref="/forward-pass"
          nextLabel="Forward Pass"
        />
      </div>
    </div>
  )
}