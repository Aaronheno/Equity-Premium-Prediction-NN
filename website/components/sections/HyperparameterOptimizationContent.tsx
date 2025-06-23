'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { BarChart3, Search, Shuffle, Brain, Settings, Target, TrendingUp, Clock } from 'lucide-react'
import CodeBlock from '@/components/shared/CodeBlock'
import MathFormula from '@/components/shared/MathFormula'
import NavigationButtons from '@/components/shared/NavigationButtons'

const searchMethods = [
  {
    name: 'Bayesian Optimization',
    description: 'Uses Tree-structured Parzen Estimator (TPE) to model p(θ|y) and select promising hyperparameters',
    algorithm: 'TPE with Expected Improvement acquisition function',
    advantages: [
      'Models hyperparameter distributions directly from trial history',
      'Handles mixed parameter types (continuous, discrete, categorical)', 
      'No gradient computation required for black-box optimization',
      'Built-in trial pruning based on intermediate results',
      'Parallelizable across multiple studies'
    ],
    disadvantages: ['Requires sufficient startup trials for model building', 'TPE assumptions may not hold for all parameter spaces'],
    bestFor: 'Expensive function evaluations where each trial requires full model training',
    icon: Brain,
    color: 'blue',
    efficiency: 'High',
    trials: '50-200'
  },
  {
    name: 'Grid Search',
    description: 'Exhaustive search through predefined parameter grid',
    algorithm: 'Systematic evaluation of all combinations',
    advantages: ['Simple and reliable', 'Guaranteed to find best in grid', 'Parallelizable'],
    disadvantages: ['Computationally expensive', 'Curse of dimensionality', 'May miss optimal values'],
    bestFor: 'Small parameter spaces or when computational resources are abundant',
    icon: Search,
    color: 'orange',
    efficiency: 'Low',
    trials: '100-10000'
  },
  {
    name: 'Random Search',
    description: 'Random sampling from parameter distributions',
    algorithm: 'Independent random draws from distributions',
    advantages: ['Simple implementation', 'Effective for many parameters', 'Unbiased exploration'],
    disadvantages: ['No learning from previous trials', 'May require many samples'],
    bestFor: 'High-dimensional spaces or initial exploration',
    icon: Shuffle,
    color: 'purple',
    efficiency: 'Medium',
    trials: '100-500'
  }
]

const hyperparameters = [
  {
    category: 'Architecture',
    params: [
      { name: 'Hidden Units', range: '4-512', description: 'Number of neurons per layer', sensitivity: 'High' },
      { name: 'Number of Layers', range: '1-5', description: 'Network depth', sensitivity: 'High' },
      { name: 'Activation Function', options: ['ReLU', 'LeakyReLU', 'ELU'], description: 'Non-linearity type', sensitivity: 'Medium' }
    ]
  },
  {
    category: 'Regularization',
    params: [
      { name: 'Dropout Rate', range: '0.0-0.5', description: 'Neuron deactivation probability', sensitivity: 'High' },
      { name: 'L1 Lambda', range: '1e-6 to 1e-2', description: 'Sparsity penalty strength', sensitivity: 'Medium' },
      { name: 'L2 Lambda', range: '1e-6 to 1e-2', description: 'Weight decay strength', sensitivity: 'Medium' }
    ]
  },
  {
    category: 'Optimization',
    params: [
      { name: 'Learning Rate', range: '1e-5 to 1e-1', description: 'Step size for weight updates', sensitivity: 'Very High' },
      { name: 'Batch Size', range: '16-512', description: 'Samples per gradient update', sensitivity: 'Medium' },
      { name: 'Optimizer', options: ['Adam', 'RMSprop', 'SGD'], description: 'Optimization algorithm', sensitivity: 'Medium' }
    ]
  },
  {
    category: 'Training',
    params: [
      { name: 'Epochs', range: '50-500', description: 'Training iterations', sensitivity: 'Medium' },
      { name: 'Early Stopping Patience', range: '5-50', description: 'Epochs without improvement', sensitivity: 'Low' },
      { name: 'Learning Rate Schedule', options: ['Step', 'Exponential', 'Plateau'], description: 'LR reduction strategy', sensitivity: 'Low' }
    ]
  }
]

const HyperparameterOptimizationContent = () => {
  const [activeMethod, setActiveMethod] = useState(0)
  const [selectedCategory, setSelectedCategory] = useState(0)

  const getMethodClasses = (method: { color: string }, isActive: boolean): string => {
    if (isActive) {
      if (method.color === 'blue') return 'bg-accent-blue/10 border-accent-blue/30'
      if (method.color === 'orange') return 'bg-accent-orange/10 border-accent-orange/30'
      if (method.color === 'purple') return 'bg-accent-purple/10 border-accent-purple/30'
    }
    return 'bg-bg-primary border-bg-tertiary hover:border-accent-blue/30'
  }

  const getIconContainerClasses = (method: { color: string }): string => {
    if (method.color === 'blue') return 'bg-accent-blue/10 border-accent-blue/20'
    if (method.color === 'orange') return 'bg-accent-orange/10 border-accent-orange/20'
    if (method.color === 'purple') return 'bg-accent-purple/10 border-accent-purple/20'
    return ''
  }

  const getIconClasses = (method: { color: string }): string => {
    if (method.color === 'blue') return 'text-accent-blue'
    if (method.color === 'orange') return 'text-accent-orange'
    if (method.color === 'purple') return 'text-accent-purple'
    return ''
  }

  const getSensitivityClasses = (sensitivity: string): string => {
    if (sensitivity === 'Very High') return 'bg-accent-red/20 text-accent-red'
    if (sensitivity === 'High') return 'bg-accent-orange/20 text-accent-orange'
    if (sensitivity === 'Medium') return 'bg-accent-blue/20 text-accent-blue'
    return 'bg-accent-green/20 text-accent-green'
  }

  return (
    <div className="min-h-screen bg-bg-primary pt-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-16"
        >
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold mb-6">
            <span className="gradient-text">Hyperparameter Optimization</span>
          </h1>
          <p className="max-w-4xl mx-auto text-xl text-text-secondary leading-relaxed">
            Advanced techniques for finding optimal neural network configurations: Bayesian optimization, 
            grid search, and random search strategies for financial prediction models.
          </p>
        </motion.div>

        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Settings className="w-8 h-8 text-accent-blue mr-3" />
            The Hyperparameter Challenge
          </h2>
          
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8 mb-8">
            <p className="text-text-secondary text-lg mb-6 leading-relaxed">
              Hyperparameters control the learning process and architecture of neural networks. Unlike 
              regular parameters (weights), hyperparameters cannot be learned directly from data and must 
              be set before training. Finding optimal values is crucial for model performance.
            </p>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div>
                <h3 className="text-xl font-semibold text-text-primary mb-4">Why Hyperparameter Optimization Matters:</h3>
                <div className="space-y-3 text-sm">
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-blue mt-2"></div>
                    <span className="text-text-secondary"><strong>Performance Impact:</strong> Can improve accuracy by 10-30%</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-blue mt-2"></div>
                    <span className="text-text-secondary"><strong>Generalization:</strong> Prevents overfitting through rigorous regularization</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-blue mt-2"></div>
                    <span className="text-text-secondary"><strong>Training Efficiency:</strong> Optimal settings reduce training time</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-blue mt-2"></div>
                    <span className="text-text-secondary"><strong>Robustness:</strong> Well-tuned models handle market changes better</span>
                  </div>
                </div>
              </div>
              
              <div>
                <h3 className="text-xl font-semibold text-text-primary mb-4">Financial Model Considerations:</h3>
                <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 space-y-3 text-sm">
                  <div className="text-accent-blue font-medium">Temporal Validation is Critical</div>
                  <div className="text-text-secondary">• Use walk-forward validation for time series</div>
                  <div className="text-text-secondary">• Never use future data for validation</div>
                  <div className="text-text-secondary">• Account for regime changes in markets</div>
                  <div className="text-text-secondary">• Consider transaction costs in objective</div>
                </div>
              </div>
            </div>
          </div>
        </motion.section>

        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <BarChart3 className="w-8 h-8 text-accent-orange mr-3" />
            Optimization Methods
          </h2>
          
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8 mb-8">
            <p className="text-text-secondary text-lg mb-8">
              Each optimization method has unique characteristics suited for different scenarios. 
              Choose based on computational budget, parameter space size, and required accuracy.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
              {searchMethods.map((method, index) => {
                const Icon = method.icon
                const isActive = activeMethod === index
                return (
                  <button
                    key={method.name}
                    onClick={() => setActiveMethod(index)}
                    className={`p-6 rounded-lg border transition-all text-left ${getMethodClasses(method, isActive)}`}
                  >
                    <div className="flex items-center space-x-3 mb-4">
                      <div className={`inline-flex items-center justify-center w-10 h-10 rounded-lg ${getIconContainerClasses(method)}`}>
                        <Icon className={`w-5 h-5 ${getIconClasses(method)}`} />
                      </div>
                      <div>
                        <h3 className="font-bold text-text-primary">{method.name}</h3>
                        <div className="flex items-center space-x-2 text-xs">
                          <span className="text-text-muted">Efficiency: {method.efficiency}</span>
                          <span className="text-text-muted">•</span>
                          <span className="text-text-muted">Trials: {method.trials}</span>
                        </div>
                      </div>
                    </div>
                    <p className="text-text-secondary text-sm">{method.description}</p>
                  </button>
                )
              })}
            </div>

            {searchMethods.map((method, index) => {
              const Icon = method.icon
              if (activeMethod !== index) return null
              
              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                  className={`${getMethodClasses(method, true)} rounded-xl p-8`}
                >
                  <div className="flex items-center space-x-3 mb-6">
                    <div className={`inline-flex items-center justify-center w-12 h-12 rounded-lg ${getIconContainerClasses(method)}`}>
                      <Icon className={`w-6 h-6 ${getIconClasses(method)}`} />
                    </div>
                    <div>
                      <h3 className="text-2xl font-bold text-text-primary">{method.name}</h3>
                      <p className="text-text-secondary">{method.algorithm}</p>
                    </div>
                  </div>

                  <div className="mb-6">
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                      <div>
                        <h4 className="font-semibold text-text-primary mb-3">Advantages:</h4>
                        <div className="space-y-2 text-sm mb-6">
                          {method.advantages.map((advantage, idx) => (
                            <div key={idx} className="flex items-start space-x-3">
                              <div className="w-2 h-2 rounded-full bg-accent-green mt-2"></div>
                              <span className="text-text-secondary">{advantage}</span>
                            </div>
                          ))}
                        </div>
                        
                        <h4 className="font-semibold text-text-primary mb-3">Disadvantages:</h4>
                        <div className="space-y-2 text-sm">
                          {method.disadvantages.map((disadvantage, idx) => (
                            <div key={idx} className="flex items-start space-x-3">
                              <div className="w-2 h-2 rounded-full bg-accent-red mt-2"></div>
                              <span className="text-text-secondary">{disadvantage}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                      
                      <div>
                        <h4 className="font-semibold text-text-primary mb-3">Best Use Case:</h4>
                        <div className={`${getMethodClasses(method, true)} rounded-lg p-4 mb-4`}>
                          <p className="text-text-secondary text-sm">{method.bestFor}</p>
                        </div>
                      </div>
                    </div>
                  </div>
                      
                  {method.name === 'Bayesian Optimization' && (
                    <div className="space-y-6">
                      <div>
                        <h4 className="font-semibold text-text-primary mb-3">Tree-structured Parzen Estimator (TPE):</h4>
                        
                        <div className="mb-6">
                          <h5 className="font-semibold text-text-primary mb-2">Core Mathematical Formulation:</h5>
                          <div className="text-sm text-text-secondary mb-2">TPE models conditional distributions directly:</div>
                          <MathFormula 
                            latex={String.raw`p(\theta | y) = \begin{cases} l(\theta) & \text{if } y < y^* \\ g(\theta) & \text{if } y \geq y^* \end{cases}`} 
                            block={true} 
                            explanation="This formula divides hyperparameter space based on performance. 'Good' hyperparameters (low loss) follow l(θ), while 'bad' ones follow g(θ). This separation allows TPE to focus sampling on promising regions."
                          />
                          <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 mt-4 text-sm">
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                              <div>
                                <h6 className="font-semibold text-text-primary mb-2">Formula Components:</h6>
                                <div className="space-y-2">
                                  <div><span className="font-mono text-accent-blue">θ</span> = hyperparameter vector (learning rate, hidden units, etc.)</div>
                                  <div><span className="font-mono text-accent-blue">y</span> = validation loss for a given hyperparameter setting</div>
                                  <div><span className="font-mono text-accent-blue">y*</span> = percentile threshold (e.g., 25th percentile of all trials)</div>
                                </div>
                              </div>
                              <div>
                                <h6 className="font-semibold text-text-primary mb-2">Distribution Functions:</h6>
                                <div className="space-y-2">
                                  <div><span className="font-mono text-accent-green">l(θ)</span> = density for "good" hyperparameters (y &lt; y*)</div>
                                  <div><span className="font-mono text-accent-orange">g(θ)</span> = density for "bad" hyperparameters (y ≥ y*)</div>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>

                        <div className="mb-6">
                          <h5 className="font-semibold text-text-primary mb-2">Expected Improvement Acquisition:</h5>
                          <div className="text-sm text-text-secondary mb-2">TPE maximizes expected improvement:</div>
                          <MathFormula 
                            latex={String.raw`EI(\theta) = \gamma \cdot l(\theta) - (1-\gamma) \cdot g(\theta)`} 
                            block={true} 
                            explanation="This acquisition function balances exploration vs exploitation. High values indicate promising hyperparameters. The formula rewards regions where good hyperparameters are likely (high l(θ)) while penalizing regions where bad hyperparameters cluster (high g(θ))."
                          />
                          <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 mt-4 text-sm">
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                              <div>
                                <h6 className="font-semibold text-text-primary mb-2">Parameters:</h6>
                                <div className="space-y-2">
                                  <div><span className="font-mono text-accent-blue">γ</span> = p(y &lt; y*) ≈ 0.25 (quantile parameter)</div>
                                  <div><span className="font-mono text-accent-green">l(θ)</span> = density of good hyperparameters at θ</div>
                                  <div><span className="font-mono text-accent-orange">g(θ)</span> = density of bad hyperparameters at θ</div>
                                </div>
                              </div>
                              <div>
                                <h6 className="font-semibold text-text-primary mb-2">Intuition:</h6>
                                <div className="space-y-2">
                                  <div><strong>High EI:</strong> More good than bad hyperparameters nearby</div>
                                  <div><strong>Low EI:</strong> Dominated by poor-performing hyperparameters</div>
                                  <div><strong>Balance:</strong> γ controls exploration appetite</div>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>

                        <div className="mb-6">
                          <h5 className="font-semibold text-text-primary mb-2">Parameter Selection Strategy:</h5>
                          <div className="text-sm text-text-secondary mb-2">Next trial selects:</div>
                          <MathFormula 
                            latex={String.raw`\theta_{\text{next}} = \arg\max_\theta EI(\theta)`} 
                            block={true} 
                            explanation="This optimization step finds the hyperparameter configuration that maximizes expected improvement. TPE intelligently samples from regions most likely to yield better performance than current best."
                          />
                          <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 mt-4 text-sm">
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                              <div>
                                <h6 className="font-semibold text-text-primary mb-2">Selection Process:</h6>
                                <div className="space-y-2">
                                  <div><strong>1. Evaluate EI(θ)</strong> across hyperparameter space</div>
                                  <div><strong>2. Find maximum</strong> of acquisition function</div>
                                  <div><strong>3. Sample θ_next</strong> from high-EI region</div>
                                  <div><strong>4. Train model</strong> with selected hyperparameters</div>
                                </div>
                              </div>
                              <div>
                                <h6 className="font-semibold text-text-primary mb-2">Key Benefits:</h6>
                                <div className="space-y-2">
                                  <div><strong>Informed sampling:</strong> Uses all previous trial history</div>
                                  <div><strong>Adaptive:</strong> Focuses on promising regions over time</div>
                                  <div><strong>Efficient:</strong> Avoids random or exhaustive search</div>
                                  <div><strong>Probabilistic:</strong> Accounts for uncertainty</div>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                      
                      <div>
                        <h4 className="font-semibold text-text-primary mb-3">Our Search Space Implementation:</h4>
                        
                        <div className="mb-4">
                          <div className="font-semibold text-sm mb-2">Learning Rate Distribution:</div>
                          <MathFormula 
                            latex={String.raw`lr \sim \text{LogUniform}(10^{-5}, 10^{-2})`} 
                            block={true} 
                            explanation="Learning rates span multiple orders of magnitude, making log-uniform sampling essential. This ensures equal probability across exponential ranges (e.g., 1e-5 to 1e-4 gets same weight as 1e-3 to 1e-2)."
                          />
                          <div className="text-xs text-text-muted mt-2">
                            <strong>Why log-scale:</strong> Learning rate sensitivity is exponential - small changes in log space create proportional effects
                          </div>
                        </div>
                        
                        <div className="mb-4">
                          <div className="font-semibold text-sm mb-2">Hidden Layer Sizes (Net3 example):</div>
                          <MathFormula 
                            latex={String.raw`h_1 \sim \text{Uniform}(16, 128), \quad h_2 \sim \text{Uniform}(8, 96), \quad h_3 \sim \text{Uniform}(4, 64)`} 
                            block={true} 
                            explanation="Uniform distributions for hidden layer sizes with decreasing ranges. This enforces architectural constraint that layers should generally decrease in size from input to output, creating an information bottleneck."
                          />
                          <div className="text-xs text-text-muted mt-2">
                            <strong>Architecture constraint:</strong> h₁ &gt; h₂ &gt; h₃ promotes hierarchical feature learning and prevents overfitting
                          </div>
                        </div>
                        
                        <div className="mb-4">
                          <div className="font-semibold text-sm mb-2">Regularization Parameters:</div>
                          <MathFormula 
                            latex={String.raw`\lambda_{L1}, \lambda_{L2} \sim \text{LogUniform}(10^{-7}, 10^{-2})`} 
                            block={true} 
                            explanation="Both L1 and L2 regularization strengths use log-uniform distributions. This covers the full spectrum from minimal regularization (1e-7) to strong regularization (1e-2), allowing TPE to find the optimal bias-variance tradeoff."
                          />
                          <div className="text-xs text-text-muted mt-2">
                            <strong>Regularization effects:</strong> L1 (λ₁) promotes sparsity, L2 (λ₂) prevents weight explosion - both crucial for financial data
                          </div>
                        </div>
                        
                        <div className="mb-4">
                          <div className="font-semibold text-sm mb-2">Categorical Parameters:</div>
                          <MathFormula 
                            latex={String.raw`\text{optimizer} \sim \text{Categorical}(\{\text{Adam}, \text{RMSprop}, \text{SGD}\})`} 
                            block={true} 
                            explanation="Categorical distribution for optimizer selection. TPE learns which optimizer works best for equity premium prediction by tracking performance across different algorithms and automatically favoring the most successful choices."
                          />
                          <div className="text-xs text-text-muted mt-2">
                            <strong>Optimizer strengths:</strong> Adam (adaptive), RMSprop (non-stationary), SGD (robust convergence) - TPE discovers best fit
                          </div>
                        </div>
                      </div>

                      <div>
                        <h4 className="font-semibold text-text-primary mb-3">Objective Function:</h4>
                        <div className="text-sm text-text-secondary mb-2">For each trial with hyperparameters θ:</div>
                        <MathFormula 
                          latex={String.raw`f(\theta) = \text{ValidationLoss}\left(\text{Train}\left(\text{Model}(\theta), \mathcal{D}_{\text{train}}\right), \mathcal{D}_{\text{val}}\right)`} 
                          block={true} 
                          explanation="This is the black-box function TPE optimizes. For each hyperparameter configuration θ, we train a complete neural network and measure its validation performance. This expensive evaluation is what makes Bayesian optimization so valuable."
                        />
                        <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 mt-4 text-sm">
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div>
                              <h6 className="font-semibold text-text-primary mb-2">Function Components:</h6>
                              <div className="space-y-2">
                                <div><span className="font-mono text-accent-blue">Model(θ)</span> = Neural network with hyperparameters θ</div>
                                <div><span className="font-mono text-accent-blue">Train()</span> = Full training with early stopping (patience=10)</div>
                                <div><span className="font-mono text-accent-blue">ValidationLoss()</span> = MSE on held-out validation set</div>
                              </div>
                            </div>
                            <div>
                              <h6 className="font-semibold text-text-primary mb-2">Why This Matters:</h6>
                              <div className="space-y-2">
                                <div><strong>Expensive:</strong> Each evaluation takes minutes/hours</div>
                                <div><strong>Noisy:</strong> Random initialization affects results</div>
                                <div><strong>Critical:</strong> Determines model's real-world performance</div>
                              </div>
                            </div>
                          </div>
                        </div>
                        <div className="mt-4">
                          <div className="text-sm text-text-secondary mb-2">TPE seeks to minimize:</div>
                          <MathFormula 
                            latex={String.raw`\theta^* = \arg\min_\theta f(\theta)`} 
                            block={true} 
                            explanation="The ultimate goal: find hyperparameters that minimize validation loss. TPE builds a probabilistic model of f(θ) to guide search toward optimal configurations while minimizing expensive function evaluations."
                          />
                        </div>
                      </div>

                      <div>
                        <h4 className="font-semibold text-text-primary mb-3">Pruning Strategy:</h4>
                        <div className="text-sm text-text-secondary mb-2">MedianPruner eliminates trial if at epoch e:</div>
                        <MathFormula 
                          latex={String.raw`\text{loss}_e > \text{median}\left(\{\text{loss}_e^{(i)} : i \in \text{completed trials}\}\right)`} 
                          block={true} 
                          explanation="Early stopping mechanism that terminates unpromising trials. If current trial's loss at epoch e exceeds the median loss of all completed trials at the same epoch, the trial is pruned. This saves computational resources for better hyperparameter configurations."
                        />
                        <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 mt-4 text-sm">
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div>
                              <h6 className="font-semibold text-text-primary mb-2">Pruning Logic:</h6>
                              <div className="space-y-2">
                                <div><span className="font-mono text-accent-blue">loss_e</span> = current trial's validation loss at epoch e</div>
                                <div><span className="font-mono text-accent-blue">loss_e^(i)</span> = loss of completed trial i at epoch e</div>
                                <div><span className="font-mono text-accent-blue">median()</span> = 50th percentile of historical performance</div>
                              </div>
                            </div>
                            <div>
                              <h6 className="font-semibold text-text-primary mb-2">Benefits:</h6>
                              <div className="space-y-2">
                                <div><strong>Resource savings:</strong> Stop bad trials early</div>
                                <div><strong>Faster optimization:</strong> More trials in same time</div>
                                <div><strong>Conservative:</strong> Only prunes clearly poor performers</div>
                                <div><strong>Adaptive:</strong> Threshold improves with better trials</div>
                              </div>
                            </div>
                          </div>
                        </div>
                        <div className="text-xs text-text-muted mt-2">
                          <strong>Warmup period:</strong> First 10 epochs allow models to stabilize before pruning decisions
                        </div>
                      </div>
                    </div>
                  )}
                  
                  {method.name === 'Grid Search' && (
                    <div>
                      <h4 className="font-semibold text-text-primary mb-3">Example Grid:</h4>
                      <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 text-sm font-mono">
                        <div>learning_rate: [0.001, 0.01, 0.1]</div>
                        <div>hidden_units: [32, 64, 128]</div>
                        <div>dropout: [0.2, 0.3, 0.4]</div>
                        <div className="text-accent-orange mt-2">Total combinations: 3 × 3 × 3 = 27</div>
                      </div>
                    </div>
                  )}
                  
                  {method.name === 'Random Search' && (
                    <div>
                      <h4 className="font-semibold text-text-primary mb-3">Distribution Examples:</h4>
                      <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 space-y-1 text-sm font-mono">
                        <div>lr ~ LogUniform(1e-5, 1e-2)</div>
                        <div>hidden ~ Uniform(16, 256)</div>
                        <div>dropout ~ Uniform(0.0, 0.5)</div>
                        <div>optimizer ~ Choice(['adam', 'rmsprop'])</div>
                      </div>
                    </div>
                  )}
                </motion.div>
              )
            })}
          </div>
        </motion.section>

        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Target className="w-8 h-8 text-accent-green mr-3" />
            Key Hyperparameters for Financial Neural Networks
          </h2>
          
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8 mb-8">
            <p className="text-text-secondary text-lg mb-8">
              Understanding which hyperparameters to tune and their typical ranges is crucial 
              for efficient optimization. Focus on high-sensitivity parameters first.
            </p>

            <div className="flex flex-wrap gap-2 mb-8">
              {hyperparameters.map((category, index) => (
                <button
                  key={category.category}
                  onClick={() => setSelectedCategory(index)}
                  className={`px-4 py-2 rounded-lg font-medium transition-all ${
                    selectedCategory === index
                      ? 'bg-accent-green text-white'
                      : 'bg-bg-primary text-text-secondary hover:text-accent-green'
                  }`}
                >
                  {category.category}
                </button>
              ))}
            </div>

            <motion.div
              key={selectedCategory}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              className="space-y-4"
            >
              {hyperparameters[selectedCategory].params.map((param) => (
                <div key={param.name} className="bg-bg-primary border border-bg-tertiary rounded-lg p-6">
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <h3 className="font-semibold text-text-primary text-lg">{param.name}</h3>
                      <p className="text-text-secondary text-sm">{param.description}</p>
                    </div>
                    <div className={`px-3 py-1 rounded-full text-xs font-medium ${getSensitivityClasses(param.sensitivity)}`}>
                      {param.sensitivity} Sensitivity
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-text-muted">
                        {param.range ? 'Range: ' : 'Options: '}
                      </span>
                      <span className="font-mono text-accent-blue">
                        {param.range || param.options?.join(', ')}
                      </span>
                    </div>
                    <div className="text-text-secondary">
                      {param.name === 'Hidden Units' && 'More units = more capacity but risk overfitting'}
                      {param.name === 'Learning Rate' && 'Most critical parameter - use log scale'}
                      {param.name === 'Dropout Rate' && '0.2-0.3 typical for financial data'}
                      {param.name === 'Batch Size' && 'Larger = more stable, smaller = faster convergence'}
                    </div>
                  </div>
                </div>
              ))}
            </motion.div>
          </div>
        </motion.section>

        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.5 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <TrendingUp className="w-8 h-8 text-accent-purple mr-3" />
            Search Space Configuration
          </h2>
          
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8 mb-8">
            <p className="text-text-secondary text-lg mb-8">
              Defining appropriate search spaces is critical for optimization success. Here are 
              actual search spaces used for equity premium prediction models.
            </p>

            <div className="space-y-8">
              <div>
                <h3 className="text-xl font-semibold text-text-primary mb-4">Actual Bayesian Search Spaces</h3>
                <CodeBlock
                  language="python"
                  title="From src/configs/search_spaces.py"
                  code={`# Bayesian search spaces for different model complexities

BAYES = {
    "Net1": {
        "hpo_config_fn": _create_optuna_hpo_config_fn(
            "Net1", 
            {"n_hidden1": IntDistribution(16, 256)}
        )
    },
    "Net3": {
        "hpo_config_fn": _create_optuna_hpo_config_fn(
            "Net3", 
            {
                "n_hidden1": IntDistribution(16, 128),
                "n_hidden2": IntDistribution(8, 96), 
                "n_hidden3": IntDistribution(4, 64)
            }
        )
    },
    "DNet1": {
        "hpo_config_fn": _create_optuna_hpo_config_fn(
            "DNet1", 
            {
                "n_hidden1": IntDistribution(64, 384),
                "n_hidden2": IntDistribution(32, 256),
                "n_hidden3": IntDistribution(16, 192),
                "n_hidden4": IntDistribution(16, 128)
            }, 
            is_dnn_model=True
        )
    }
}

# Base distributions shared across all models
_BASE_DISTRIBUTIONS = {
    "optimizer_choice": CategoricalDistribution(["Adam", "RMSprop", "SGD"]),
    "lr": FloatDistribution(1e-5, 1e-2, log=True),
    "weight_decay": FloatDistribution(1e-7, 1e-2, log=True),  # L2 regularization
    "l1_lambda": FloatDistribution(1e-7, 1e-2, log=True),     # L1 regularization
    "dropout": FloatDistribution(0.0, 0.6, step=0.05),
    "batch_size": CategoricalDistribution([64, 128, 256, 512, 1024]),
}`}
                />
              </div>

              <div>
                <h3 className="text-xl font-semibold text-text-primary mb-4">Grid Search Spaces (OOS)</h3>
                <CodeBlock
                  language="python"
                  title="Reduced Grid for Out-of-Sample Annual HPO"
                  code={`# Significantly reduced for computational feasibility
_BASE_GRID_PARAMS_OOS_REDUCED = { 
    "optimizer": [torch.optim.Adam, torch.optim.RMSprop],
    "lr": [5e-4],  # Single value for efficiency
    "optimizer__weight_decay": [1e-5, 1e-4],  # Two L2 options
    "l1_lambda": [0, 1e-4],  # No L1 and moderate L1
    "module__dropout": [0.1, 0.3],  # Low and moderate dropout
    "batch_size": [256],  # Standard batch size   
}

GRID_OOS = {
    "Net1": {
        **_BASE_GRID_PARAMS_OOS_REDUCED, 
        "module__n_hidden1": [64],  # Single neuron count
        "module__activation_hidden": ["relu"]
    },
    "Net3": {
        **_BASE_GRID_PARAMS_OOS_REDUCED,
        "module__n_hidden1": [96], 
        "module__n_hidden2": [48], 
        "module__n_hidden3": [24],
        "module__activation_hidden": ["relu"]
    }
}

# Warning: Annual HPO with Grid Search is computationally expensive
# These grids are intentionally minimal for feasibility`}
                />
              </div>
            </div>

            <div className="mt-8 bg-accent-purple/5 border border-accent-purple/20 rounded-lg p-6">
              <h3 className="text-xl font-semibold text-accent-purple mb-4">Search Space Design Tips</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm">
                <div>
                  <h4 className="font-semibold text-text-primary mb-2">Use Log Scale For:</h4>
                  <div className="space-y-1 text-text-secondary">
                    <div>• Learning rates (spans orders of magnitude)</div>
                    <div>• Regularization parameters (λ₁, λ₂)</div>
                    <div>• Any parameter varying by 10x or more</div>
                  </div>
                </div>
                <div>
                  <h4 className="font-semibold text-text-primary mb-2">Constraint Relationships:</h4>
                  <div className="space-y-1 text-text-secondary">
                    <div>• Layer sizes should decrease: h₁ &gt; h₂ &gt; h₃</div>
                    <div>• Batch size should divide dataset size</div>
                    <div>• Consider memory constraints for large models</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </motion.section>

        <NavigationButtons 
          prevHref="/optimization"
          prevLabel="Optimization"
          nextHref="/predictions"
          nextLabel="Making Predictions"
        />
      </div>
    </div>
  )
}

export default HyperparameterOptimizationContent