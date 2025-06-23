'use client'

import { motion } from 'framer-motion'
import { Layers, Zap, BarChart3 } from 'lucide-react'
import Link from 'next/link'

const models = [
  {
    name: "Net1",
    type: "Standard",
    layers: 1,
    description: "Single hidden layer architecture for baseline comparison",
    params: "n_hidden1: 16-256",
    color: "accent-blue",
    use_case: "Simple patterns, fast training"
  },
  {
    name: "Net2", 
    type: "Standard",
    layers: 2,
    description: "Two hidden layers for moderate complexity",
    params: "n_hidden1: 16-192, n_hidden2: 8-128",
    color: "accent-blue",
    use_case: "Balanced complexity and performance"
  },
  {
    name: "Net3",
    type: "Standard", 
    layers: 3,
    description: "Three hidden layers, often optimal for financial data",
    params: "n_hidden1: 16-128, n_hidden2: 8-96, n_hidden3: 4-64",
    color: "accent-blue",
    use_case: "Sweet spot for equity premium prediction"
  },
  {
    name: "Net4",
    type: "Standard",
    layers: 4, 
    description: "Four hidden layers for complex pattern recognition",
    params: "n_hidden1: 32-192, n_hidden2: 16-128, n_hidden3: 8-96, n_hidden4: 4-64",
    color: "accent-blue",
    use_case: "Complex non-linear relationships"
  },
  {
    name: "Net5",
    type: "Standard",
    layers: 5,
    description: "Five hidden layers, maximum depth for standard models",
    params: "n_hidden1: 32-256, n_hidden2: 16-192, n_hidden3: 8-128, n_hidden4: 8-96, n_hidden5: 4-64", 
    color: "accent-blue",
    use_case: "Highly complex pattern capture"
  },
  {
    name: "DNet1",
    type: "Enhanced",
    layers: 4,
    description: "Four DBlocks with batch normalization and ReLU",
    params: "n_hidden1: 64-384, n_hidden2: 32-256, n_hidden3: 16-192, n_hidden4: 16-128",
    color: "accent-purple",
    use_case: "Stable training with normalization"
  },
  {
    name: "DNet2", 
    type: "Enhanced",
    layers: 5,
    description: "Five DBlocks with enhanced regularization",
    params: "n_hidden1: 64-512, n_hidden2: 32-384, n_hidden3: 16-256, n_hidden4: 16-192, n_hidden5: 8-128",
    color: "accent-purple", 
    use_case: "Deep learning with batch normalization"
  },
  {
    name: "DNet3",
    type: "Enhanced",
    layers: 5,
    description: "Five DBlocks, maximum capacity with normalization",
    params: "n_hidden1: 64-512, n_hidden2: 32-384, n_hidden3: 16-256, n_hidden4: 16-192, n_hidden5: 8-128",
    color: "accent-purple",
    use_case: "Most sophisticated architecture"
  }
]

export default function ModelArchitectures() {
  return (
    <section className="py-24 bg-bg-primary">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold mb-6">
            <span className="text-text-primary">8 Neural Network</span>
            <br />
            <span className="gradient-text">Architectures</span>
          </h2>
          <p className="max-w-3xl mx-auto text-xl text-text-secondary leading-relaxed">
            From simple single-layer networks to sophisticated deep architectures with batch normalization.
            All models are implemented in <code className="text-accent-green bg-code-bg px-2 py-1 rounded">src/models/nns.py</code> and optimized for financial time series.
          </p>
        </motion.div>

        {/* Architecture Comparison */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          viewport={{ once: true }}
          className="mb-16"
        >
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-6">
            <h3 className="text-xl font-bold mb-4 text-center text-text-primary">
              Architecture Types
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-bg-primary border border-accent-blue/20 rounded-lg p-4">
                <div className="flex items-center space-x-2 mb-3">
                  <Layers className="w-5 h-5 text-accent-blue" />
                  <span className="font-semibold text-accent-blue">Standard Networks (Net1-Net5)</span>
                </div>
                <ul className="space-y-2 text-text-secondary text-sm">
                  <li>• Simple linear layers with ReLU activation</li>
                  <li>• Dropout regularization only</li>
                  <li>• Fast training and inference</li>
                  <li>• Good baseline performance</li>
                </ul>
              </div>
              
              <div className="bg-bg-primary border border-accent-purple/20 rounded-lg p-4">
                <div className="flex items-center space-x-2 mb-3">
                  <Zap className="w-5 h-5 text-accent-purple" />
                  <span className="font-semibold text-accent-purple">Enhanced Networks (DNet1-DNet3)</span>
                </div>
                <ul className="space-y-2 text-text-secondary text-sm">
                  <li>• DBlock components with batch normalization</li>
                  <li>• Improved gradient flow and stability</li>
                  <li>• Better handling of financial data noise</li>
                  <li>• Enhanced regularization capabilities</li>
                </ul>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Models Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {models.map((model, index) => (
            <motion.div
              key={model.name}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              viewport={{ once: true }}
              className="group"
            >
              <div className="card-hover bg-bg-secondary border border-bg-tertiary rounded-xl p-6 h-full flex flex-col">
                {/* Model Header */}
                <div className="flex items-center justify-between mb-4">
                  <div className={`inline-flex items-center justify-center w-10 h-10 rounded-lg bg-${model.color}/10 border border-${model.color}/20`}>
                    <span className={`font-bold text-${model.color}`}>{model.name}</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    {Array.from({ length: model.layers }).map((_, i) => (
                      <div key={i} className={`w-2 h-6 bg-${model.color}/40 rounded-sm`} />
                    ))}
                  </div>
                </div>

                {/* Model Type Badge */}
                <div className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium mb-3 ${
                  model.type === 'Standard' 
                    ? 'bg-accent-blue/10 text-accent-blue border border-accent-blue/20'
                    : 'bg-accent-purple/10 text-accent-purple border border-accent-purple/20'
                }`}>
                  {model.type} • {model.layers} Layer{model.layers > 1 ? 's' : ''}
                </div>

                {/* Description */}
                <p className="text-text-secondary text-sm mb-4 flex-grow">
                  {model.description}
                </p>

                {/* Parameters */}
                <div className="bg-code-bg border border-code-border rounded-lg p-3 mb-4">
                  <div className="text-xs text-text-muted mb-1">Parameters:</div>
                  <code className="text-xs text-accent-green font-mono break-all">
                    {model.params}
                  </code>
                </div>

                {/* Use Case */}
                <div className="mb-4">
                  <div className="text-xs text-text-muted mb-1">Best for:</div>
                  <div className="text-sm text-text-secondary">
                    {model.use_case}
                  </div>
                </div>

                {/* Learn More Button */}
                <Link
                  href="/architecture"
                  className={`inline-flex items-center justify-center space-x-2 w-full py-2 px-4 rounded-lg border transition-all text-sm font-medium ${
                    model.type === 'Standard'
                      ? 'border-accent-blue/20 text-accent-blue hover:bg-accent-blue/10'
                      : 'border-accent-purple/20 text-accent-purple hover:bg-accent-purple/10'
                  }`}
                >
                  <BarChart3 className="w-4 h-4" />
                  <span>View Details</span>
                </Link>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Interactive Architecture Link */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          viewport={{ once: true }}
          className="mt-16 text-center"
        >
          <Link
            href="/interactive-architecture"
            className="inline-flex items-center space-x-3 bg-gradient-to-r from-accent-blue to-accent-purple text-white px-8 py-4 rounded-lg font-semibold text-lg hover:scale-105 transition-all duration-200 shadow-lg hover:shadow-xl"
          >
            <Layers className="w-6 h-6" />
            <span>Explore Interactive Architecture Viewer</span>
          </Link>
        </motion.div>
      </div>
    </section>
  )
}