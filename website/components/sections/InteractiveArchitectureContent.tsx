'use client'

import { motion } from 'framer-motion'
import { useState, useEffect } from 'react'
import { Brain, Layers, Eye, Settings, Play, Zap, ArrowRight, Target, BarChart3, Maximize2 } from 'lucide-react'
import NavigationButtons from '@/components/shared/NavigationButtons'

const architectures = [
  {
    name: 'Net1',
    title: 'Single Layer Network',
    layers: [32, 64, 1],
    description: 'Simplest architecture with one hidden layer',
    complexity: 'Low',
    parameters: '2,177',
    useCase: 'Baseline comparisons and quick prototyping',
    color: 'from-blue-400 to-blue-600'
  },
  {
    name: 'Net2',
    title: 'Two Layer Network', 
    layers: [32, 64, 32, 1],
    description: 'Two hidden layers for moderate complexity',
    complexity: 'Medium',
    parameters: '4,225',
    useCase: 'Balanced performance and interpretability',
    color: 'from-green-400 to-green-600'
  },
  {
    name: 'Net3',
    title: 'Three Layer Network',
    layers: [32, 64, 32, 16, 1],
    description: 'Standard architecture for financial modeling',
    complexity: 'Medium',
    parameters: '4,737',
    useCase: 'Primary model for equity premium prediction',
    color: 'from-purple-400 to-purple-600'
  },
  {
    name: 'Net4',
    title: 'Four Layer Network',
    layers: [32, 64, 32, 16, 8, 1],
    description: 'Deeper network for complex pattern recognition',
    complexity: 'High',
    parameters: '4,865',
    useCase: 'Complex financial relationships',
    color: 'from-orange-400 to-orange-600'
  },
  {
    name: 'Net5',
    title: 'Five Layer Network',
    layers: [32, 64, 32, 16, 8, 4, 1],
    description: 'Deepest standard architecture',
    complexity: 'High',
    parameters: '4,901',
    useCase: 'Maximum capacity for pattern learning',
    color: 'from-red-400 to-red-600'
  },
  {
    name: 'DNet1',
    title: 'Deep Network 1',
    layers: [32, 128, 64, 32, 16, 1],
    description: 'BatchNorm-enhanced 4-layer deep network',
    complexity: 'High',
    parameters: '11,089',
    useCase: 'BREAKTHROUGH MODEL: +0.75% R², 26.45% return',
    color: 'from-indigo-400 to-indigo-600',
    hasBatchNorm: true
  },
  {
    name: 'DNet2', 
    title: 'Deep Network 2',
    layers: [32, 128, 96, 64, 32, 16, 1],
    description: 'Five layers with batch normalization',
    complexity: 'Very High',
    parameters: '18,065',
    useCase: 'Advanced deep architecture with normalization',
    color: 'from-pink-400 to-pink-600',
    hasBatchNorm: true
  },
  {
    name: 'DNet3',
    title: 'Deep Network 3',
    layers: [32, 256, 128, 64, 32, 16, 1],
    description: 'Largest architecture for maximum capacity',
    complexity: 'Very High',
    parameters: '43,313',
    useCase: 'Maximum model capacity and complexity',
    color: 'from-teal-400 to-teal-600',
    hasBatchNorm: true
  }
]

const visualizationModes = [
  {
    mode: 'Network Diagram',
    description: 'Interactive network topology visualization',
    icon: Brain,
    features: ['Node connections', 'Layer information', 'Parameter counts', 'Activation flow']
  },
  {
    mode: 'Architecture Comparison',
    description: 'Side-by-side architecture analysis',
    icon: BarChart3,
    features: ['Size comparison', 'Complexity metrics', 'Performance overlay', 'Parameter efficiency']
  },
  {
    mode: 'Data Flow Animation',
    description: 'Animated forward pass visualization',
    icon: Play,
    features: ['Forward propagation', 'Activation visualization', 'Dimension changes', 'Processing flow']
  },
  {
    mode: 'Interactive Explorer',
    description: 'Deep dive into model components',
    icon: Eye,
    features: ['Layer details', 'Weight matrices', 'Activation functions', 'Dropout effects']
  }
]

export default function InteractiveArchitectureContent() {
  const [selectedArchitecture, setSelectedArchitecture] = useState(0)
  const [visualizationMode, setVisualizationMode] = useState(0)
  const [isAnimating, setIsAnimating] = useState(false)
  const [hoveredLayer, setHoveredLayer] = useState<number | null>(null)

  const currentArch = architectures[selectedArchitecture]

  // Animation control
  useEffect(() => {
    if (isAnimating) {
      const timer = setTimeout(() => setIsAnimating(false), 3000)
      return () => clearTimeout(timer)
    }
  }, [isAnimating])

  const NetworkVisualization = () => {
    const layers = currentArch.layers
    const maxNodes = Math.max(...layers)
    
    return (
      <div className="relative w-full h-96 bg-gradient-to-br from-slate-50 via-blue-50 to-purple-50 border border-slate-200 rounded-xl p-8 overflow-hidden shadow-lg">
        <div className="flex items-center justify-between h-full">
          {layers.map((nodeCount, layerIndex) => (
            <div key={layerIndex} className="flex flex-col items-center space-y-2 relative">
              {/* Layer Label */}
              <div className="text-xs font-semibold text-slate-700 mb-2 bg-white/70 px-2 py-1 rounded-md">
                {layerIndex === 0 ? 'Input (32)' : 
                 layerIndex === layers.length - 1 ? 'Output (1)' : 
                 `Hidden ${layerIndex}`}
              </div>
              
              {/* Nodes */}
              <div className="flex flex-col items-center space-y-1 relative">
                {Array.from({length: Math.min(nodeCount, 8)}, (_, nodeIndex) => (
                  <motion.div
                    key={nodeIndex}
                    className={`w-4 h-4 rounded-full border-2 shadow-sm transition-all duration-200 ${
                      hoveredLayer === layerIndex 
                        ? 'border-blue-500 bg-blue-200 shadow-blue-200 scale-110' 
                        : layerIndex === 0 
                          ? 'border-green-500 bg-green-100' 
                          : layerIndex === layers.length - 1
                            ? 'border-red-500 bg-red-100'
                            : 'border-purple-500 bg-purple-100'
                    }`}
                    initial={{ scale: 0, opacity: 0 }}
                    animate={{ 
                      scale: isAnimating ? [1, 1.2, 1] : 1, 
                      opacity: 1,
                      backgroundColor: isAnimating ? ['#ffffff', '#3b82f6', '#ffffff'] : '#ffffff'
                    }}
                    transition={{ 
                      duration: isAnimating ? 0.5 : 0.3,
                      delay: isAnimating ? layerIndex * 0.3 + nodeIndex * 0.05 : 0
                    }}
                    onMouseEnter={() => setHoveredLayer(layerIndex)}
                    onMouseLeave={() => setHoveredLayer(null)}
                  />
                ))}
                {nodeCount > 8 && (
                  <div className="text-xs text-gray-500">+{nodeCount - 8}</div>
                )}
                
              </div>
              
              {/* Node Count */}
              <div className="text-xs font-mono text-slate-700 bg-white/80 px-2 py-1 rounded-md font-semibold">
                {nodeCount} nodes
              </div>
              
              {/* Connections to next layer */}
              {layerIndex < layers.length - 1 && (
                <div className="absolute left-1/2 top-1/2 transform -translate-y-1/2 pointer-events-none">
                  <svg width="150" height="200" className="overflow-visible" style={{transform: 'translateX(20px)'}}>
                    {(() => {
                      const currentLayerNodes = Math.min(nodeCount, 8)
                      const nextLayerNodes = Math.min(layers[layerIndex + 1], 8)
                      const nodeSize = 16
                      const nodeSpacing = 20
                      
                      // Simple centering - nodes are centered in their own container
                      const svgCenter = 100 // Half of svg height
                      
                      // Calculate node positions relative to center
                      const currentNodesHeight = (currentLayerNodes - 1) * nodeSpacing
                      const nextNodesHeight = (nextLayerNodes - 1) * nodeSpacing
                      
                      const currentStartY = svgCenter - (currentNodesHeight / 2)
                      const nextStartY = svgCenter - (nextNodesHeight / 2)
                      
                      return Array.from({length: currentLayerNodes}, (_, i) => 
                        Array.from({length: nextLayerNodes}, (_, j) => (
                          <motion.line
                            key={`${i}-${j}`}
                            x1="0"
                            y1={currentStartY + (i * nodeSpacing) + (nodeSize / 2)}
                            x2="130"
                            y2={nextStartY + (j * nodeSpacing) + (nodeSize / 2)}
                            stroke="#8b5cf6"
                            strokeWidth={isAnimating ? "2" : "1"}
                            opacity={isAnimating ? 0.7 : 0.3}
                            animate={{
                              stroke: isAnimating ? ['#8b5cf6', '#3b82f6', '#8b5cf6'] : '#8b5cf6'
                            }}
                            transition={{
                              duration: isAnimating ? 0.8 : 0,
                              delay: isAnimating ? layerIndex * 0.3 : 0
                            }}
                          />
                        ))
                      )
                    })()}
                  </svg>
                </div>
              )}
            </div>
          ))}
        </div>
        
        {/* Activation Flow Indicator */}
        {isAnimating && (
          <motion.div
            className="absolute top-4 left-8 right-8 h-1 bg-gradient-to-r from-blue-400 to-purple-400 rounded-full"
            initial={{ scaleX: 0 }}
            animate={{ scaleX: 1 }}
            transition={{ duration: 2.5, ease: "easeInOut" }}
            style={{ transformOrigin: 'left' }}
          />
        )}
        
        {/* BatchNorm Indicator */}
        {currentArch.hasBatchNorm && (
          <div className="absolute top-2 right-2 px-2 py-1 bg-purple-100 text-purple-700 text-xs rounded-full">
            BatchNorm
          </div>
        )}
      </div>
    )
  }

  const ArchitectureComparison = () => {
    const maxParams = Math.max(...architectures.map(arch => parseInt(arch.parameters.replace(',', ''))))
    
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {architectures.map((arch, index) => {
          const paramCount = parseInt(arch.parameters.replace(',', ''))
          const relativeSize = (paramCount / maxParams) * 100
          
          return (
            <motion.div
              key={arch.name}
              className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                selectedArchitecture === index 
                  ? 'border-blue-500 bg-blue-50' 
                  : 'border-gray-200 bg-white hover:border-gray-300'
              }`}
              onClick={() => setSelectedArchitecture(index)}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-bold text-gray-900">{arch.name}</h3>
                <div className={`px-2 py-1 rounded text-xs font-medium ${
                  arch.complexity === 'Low' ? 'bg-green-100 text-green-700' :
                  arch.complexity === 'Medium' ? 'bg-yellow-100 text-yellow-700' :
                  arch.complexity === 'High' ? 'bg-orange-100 text-orange-700' :
                  'bg-red-100 text-red-700'
                }`}>
                  {arch.complexity}
                </div>
              </div>
              
              <div className="space-y-2 text-sm">
                <div className="text-gray-600">{arch.description}</div>
                <div className="font-mono text-blue-600">{arch.parameters} params</div>
                
                {/* Parameter size visualization */}
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className={`bg-gradient-to-r ${arch.color} h-2 rounded-full transition-all duration-500`}
                    style={{ width: `${relativeSize}%` }}
                  />
                </div>
                
                <div className="text-xs text-gray-500">{arch.useCase}</div>
              </div>
              
              {arch.hasBatchNorm && (
                <div className="mt-2 text-xs bg-purple-100 text-purple-700 px-2 py-1 rounded">
                  Batch Normalization
                </div>
              )}
            </motion.div>
          )
        })}
      </div>
    )
  }

  const DataFlowAnimation = () => (
    <div className="space-y-6">
      <div className="text-center">
        <button
          onClick={() => setIsAnimating(true)}
          className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center space-x-2 mx-auto"
        >
          <Play className="w-5 h-5" />
          <span>Animate Forward Pass</span>
        </button>
      </div>
      
      <NetworkVisualization />
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
        <div className="bg-blue-50 p-4 rounded-lg">
          <h4 className="font-semibold text-blue-900 mb-2">Input Processing</h4>
          <p className="text-blue-700">32 financial indicators are standardized and fed into the network simultaneously.</p>
        </div>
        <div className="bg-green-50 p-4 rounded-lg">
          <h4 className="font-semibold text-green-900 mb-2">Hidden Layers</h4>
          <p className="text-green-700">Each layer applies linear transformation, ReLU activation, and dropout regularization.</p>
        </div>
        <div className="bg-purple-50 p-4 rounded-lg">
          <h4 className="font-semibold text-purple-900 mb-2">Output Generation</h4>
          <p className="text-purple-700">Final layer produces a single equity premium prediction value.</p>
        </div>
      </div>
    </div>
  )

  const InteractiveExplorer = () => (
    <div className="space-y-6">
      <NetworkVisualization />
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-4">
          <h3 className="font-semibold text-text-primary">Architecture Details</h3>
          <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 space-y-3">
            <div className="flex justify-between">
              <span className="text-text-muted">Model:</span>
              <span className="font-mono font-bold text-text-primary">{currentArch.name}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-text-muted">Layers:</span>
              <span className="font-mono text-text-secondary">{currentArch.layers.length - 2} hidden + 1 output</span>
            </div>
            <div className="flex justify-between">
              <span className="text-text-muted">Parameters:</span>
              <span className="font-mono text-text-secondary">{currentArch.parameters}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-text-muted">Complexity:</span>
              <span className={`px-2 py-1 rounded text-xs ${
                currentArch.complexity === 'Low' ? 'bg-accent-green/20 text-accent-green' :
                currentArch.complexity === 'Medium' ? 'bg-accent-orange/20 text-accent-orange' :
                currentArch.complexity === 'High' ? 'bg-accent-red/20 text-accent-red' :
                'bg-accent-purple/20 text-accent-purple'
              }`}>
                {currentArch.complexity}
              </span>
            </div>
            {currentArch.hasBatchNorm && (
              <div className="flex justify-between">
                <span className="text-text-muted">Normalization:</span>
                <span className="text-accent-purple font-medium">Batch Normalization</span>
              </div>
            )}
          </div>
        </div>
        
        <div className="space-y-4">
          <h3 className="font-semibold text-text-primary">Layer Information</h3>
          <div className="space-y-2">
            {currentArch.layers.map((nodeCount, index) => (
              <div
                key={index}
                className={`p-3 rounded-lg border transition-all cursor-pointer ${
                  hoveredLayer === index ? 'border-accent-blue bg-accent-blue/10' : 'border-bg-tertiary bg-bg-primary'
                }`}
                onMouseEnter={() => setHoveredLayer(index)}
                onMouseLeave={() => setHoveredLayer(null)}
              >
                <div className="flex justify-between items-center">
                  <span className="font-medium text-text-primary">
                    {index === 0 ? 'Input Layer' : 
                     index === currentArch.layers.length - 1 ? 'Output Layer' : 
                     `Hidden Layer ${index}`}
                  </span>
                  <span className="font-mono text-accent-blue">{nodeCount} nodes</span>
                </div>
                {index > 0 && (
                  <div className="text-sm text-text-muted mt-1">
                    ReLU activation {currentArch.hasBatchNorm && index < currentArch.layers.length - 1 && '+ BatchNorm'} + Dropout
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )

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
            <span className="gradient-text">Interactive Architecture</span>
          </h1>
          <p className="max-w-4xl mx-auto text-xl text-text-secondary leading-relaxed">
            Visualize and explore all 8 neural network architectures: from simple single-layer 
            networks to complex deep architectures with batch normalization.
          </p>
        </motion.div>

        {/* Visualization Mode Selector */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="mb-12"
        >
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {visualizationModes.map((mode, index) => {
              const Icon = mode.icon
              const isActive = visualizationMode === index
              return (
                <button
                  key={mode.mode}
                  onClick={() => setVisualizationMode(index)}
                  className={`p-6 rounded-xl border-2 transition-all text-left ${
                    isActive 
                      ? 'border-accent-blue bg-accent-blue/10' 
                      : 'border-bg-tertiary bg-bg-secondary hover:border-accent-blue/50'
                  }`}
                >
                  <div className="flex items-center space-x-3 mb-3">
                    <div className={`inline-flex items-center justify-center w-10 h-10 rounded-lg ${
                      isActive ? 'bg-accent-blue/20' : 'bg-bg-tertiary'
                    }`}>
                      <Icon className={`w-5 h-5 ${isActive ? 'text-accent-blue' : 'text-text-muted'}`} />
                    </div>
                    <h3 className="font-semibold text-text-primary">{mode.mode}</h3>
                  </div>
                  <p className="text-text-secondary text-sm mb-3">{mode.description}</p>
                  <div className="space-y-1">
                    {mode.features.map((feature, idx) => (
                      <div key={idx} className="flex items-center space-x-2 text-xs text-text-muted">
                        <div className="w-1 h-1 rounded-full bg-accent-blue"></div>
                        <span>{feature}</span>
                      </div>
                    ))}
                  </div>
                </button>
              )
            })}
          </div>
        </motion.section>

        {/* Main Visualization Area */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="mb-16"
        >
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8">
            {visualizationMode === 0 && (
              <div className="space-y-8">
                <div className="flex items-center justify-between">
                  <h2 className="text-2xl font-bold text-text-primary">Network Diagram</h2>
                  <select
                    value={selectedArchitecture}
                    onChange={(e) => setSelectedArchitecture(parseInt(e.target.value))}
                    className="px-4 py-2 border border-bg-tertiary rounded-lg bg-bg-primary text-text-primary"
                  >
                    {architectures.map((arch, index) => (
                      <option key={arch.name} value={index}>
                        {arch.name} - {arch.title}
                      </option>
                    ))}
                  </select>
                </div>
                <NetworkVisualization />
                <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-6">
                  <h3 className="font-semibold text-text-primary mb-2">{currentArch.title}</h3>
                  <p className="text-text-secondary text-sm mb-4">{currentArch.description}</p>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div>
                      <span className="text-text-muted">Parameters:</span>
                      <div className="font-mono text-accent-blue">{currentArch.parameters}</div>
                    </div>
                    <div>
                      <span className="text-text-muted">Complexity:</span>
                      <div className="font-medium">{currentArch.complexity}</div>
                    </div>
                    <div>
                      <span className="text-text-muted">Layers:</span>
                      <div className="font-mono">{currentArch.layers.length - 2} hidden</div>
                    </div>
                    <div>
                      <span className="text-text-muted">Use Case:</span>
                      <div className="text-text-secondary">{currentArch.useCase}</div>
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            {visualizationMode === 1 && (
              <div className="space-y-8">
                <h2 className="text-2xl font-bold text-text-primary">Architecture Comparison</h2>
                <ArchitectureComparison />
              </div>
            )}
            
            {visualizationMode === 2 && (
              <div className="space-y-8">
                <h2 className="text-2xl font-bold text-text-primary">Data Flow Animation</h2>
                <DataFlowAnimation />
              </div>
            )}
            
            {visualizationMode === 3 && (
              <div className="space-y-8">
                <h2 className="text-2xl font-bold text-text-primary">Interactive Explorer</h2>
                <InteractiveExplorer />
              </div>
            )}
          </div>
        </motion.section>

        {/* External Visualization Link */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="mb-20"
        >
          <div className="bg-gradient-to-br from-accent-blue/5 to-accent-purple/5 border border-accent-blue/20 rounded-xl p-8">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-xl font-bold text-text-primary mb-2">Advanced Visualization</h3>
                <p className="text-text-secondary">
                  Explore the complete Net5 architecture in an enhanced interactive visualization 
                  with detailed Mermaid.js diagrams and advanced controls.
                </p>
              </div>
              <a
                href="/visualizations/net5_architecture_horizontal.html"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center space-x-2 px-6 py-3 bg-accent-blue text-white rounded-lg hover:bg-accent-blue/90 transition-colors"
              >
                <Maximize2 className="w-5 h-5" />
                <span>Open Full Visualization</span>
              </a>
            </div>
          </div>
        </motion.section>

        {/* Architecture Summary */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.8 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Layers className="w-8 h-8 text-accent-green mr-3" />
            Architecture Summary
          </h2>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-6">
              <h3 className="text-xl font-semibold text-text-primary mb-4">Standard Networks (Net1-5)</h3>
              <div className="space-y-3 text-sm">
                <div className="flex justify-between">
                  <span className="text-text-muted">Architecture:</span>
                  <span className="text-text-secondary">Feedforward with ReLU + Dropout</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-text-muted">Layers:</span>
                  <span className="text-text-secondary">1-5 hidden layers</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-text-muted">Parameters:</span>
                  <span className="text-text-secondary">1,985 - 4,765</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-text-muted">Best Use:</span>
                  <span className="text-text-secondary">Baseline to moderate complexity</span>
                </div>
              </div>
            </div>
            
            <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-6">
              <h3 className="text-xl font-semibold text-text-primary mb-4">Deep Networks (DNet1-3)</h3>
              <div className="space-y-3 text-sm">
                <div className="flex justify-between">
                  <span className="text-text-muted">Architecture:</span>
                  <span className="text-text-secondary">Enhanced with Batch Normalization</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-text-muted">Layers:</span>
                  <span className="text-text-secondary">4-5 hidden layers</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-text-muted">Parameters:</span>
                  <span className="text-text-secondary">10,833 - 42,801</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-text-muted">Best Use:</span>
                  <span className="text-text-secondary">Complex financial relationships</span>
                </div>
              </div>
            </div>
          </div>
          
          <div className="mt-8 bg-accent-green/5 border border-accent-green/20 rounded-lg p-6">
            <h4 className="font-semibold text-accent-green mb-3">Key Design Insights</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm text-text-secondary">
              <div>
                <h5 className="font-semibold text-text-primary mb-2">Progressive Complexity:</h5>
                <p>Models increase in depth and width systematically, allowing comparison of architectural choices on prediction performance.</p>
              </div>
              <div>
                <h5 className="font-semibold text-text-primary mb-2">Batch Normalization:</h5>
                <p>DNet models include batch normalization for training stability and improved convergence in deeper architectures.</p>
              </div>
            </div>
          </div>
        </motion.section>

        {/* Navigation */}
        <NavigationButtons 
          prevHref="/complete-pipeline"
          prevLabel="Complete Pipeline"
          nextHref="/multithreading"
          nextLabel="128-Core Optimization Plan"
        />
      </div>
    </div>
  )
}