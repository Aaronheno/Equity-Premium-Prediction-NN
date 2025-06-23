'use client'

import { motion } from 'framer-motion'
import { Code, FileText, Layers, Settings, Database, BarChart3 } from 'lucide-react'

const implementationAreas = [
  {
    title: "Neural Network Architectures",
    file: "src/models/nns.py",
    icon: Layers,
    description: "8 neural network models: Net1-Net5 (standard) and DNet1-DNet3 (with batch normalization)",
    highlights: [
      "Modular _Base class for shared functionality",
      "DBlock components with BatchNorm and ReLU",
      "Configurable dropout, activation functions, L1 and L2 regularization",
      "Optimized for financial time series data"
    ],
    color: "accent-blue",
    models: [
      { name: "Net1", params: "1 hidden layer", neurons: "16-256" },
      { name: "Net2", params: "2 hidden layers", neurons: "16-192, 8-128" },
      { name: "Net3", params: "3 hidden layers", neurons: "16-128, 8-96, 4-64" },
      { name: "Net4", params: "4 hidden layers", neurons: "32-192, 16-128, 8-96, 4-64" },
      { name: "Net5", params: "5 hidden layers", neurons: "32-256, 16-192, 8-128, 8-96, 4-64" },
      { name: "DNet1", params: "4 DBlocks + BatchNorm", neurons: "64-384, 32-256, 16-192, 16-128" },
      { name: "DNet2", params: "5 DBlocks + BatchNorm", neurons: "64-512, 32-384, 16-256, 16-192, 8-128" },
      { name: "DNet3", params: "5 DBlocks + BatchNorm", neurons: "64-512, 32-384, 16-256, 16-192, 8-128" }
    ]
  },
  {
    title: "Hyperparameter Configuration",
    file: "src/configs/search_spaces.py",
    icon: Settings,
    description: "Comprehensive search spaces for Bayesian, grid, and random optimization methods",
    highlights: [
      "Financial domain-specific parameter ranges",
      "Separate in-sample and out-of-sample configurations",
      "Model-specific architecture constraints",
      "Optimizer and regularization parameters"
    ],
    color: "accent-purple",
    spaces: [
      { method: "Bayesian (BAYES)", description: "Optuna-based smart hyperparameter search" },
      { method: "Grid (GRID)", description: "Systematic parameter combination testing" },
      { method: "Random (RANDOM)", description: "Random sampling from parameter distributions" },
      { method: "OOS Variants", description: "Constrained spaces for annual reoptimization" }
    ]
  },
  {
    title: "Experiment Pipeline",
    file: "src/experiments/",
    icon: BarChart3,
    description: "25+ experiment scripts covering in-sample optimization, out-of-sample validation, and analysis",
    highlights: [
      "Automated experiment execution and logging",
      "Temporal validation with expanding windows",
      "Statistical significance testing",
      "Economic value and market timing analysis"
    ],
    color: "accent-green",
    experiments: [
      { script: "bayes_oos_1.py", purpose: "Main out-of-sample Bayesian optimization" },
      { script: "grid_is_0.py", purpose: "In-sample grid search baseline" },
      { script: "economic_value_2.py", purpose: "Market timing and economic significance" },
      { script: "expanding_window_4.py", purpose: "Temporal validation methodology" }
    ]
  },
  {
    title: "Training & Utilities",
    file: "src/utils/",
    icon: Code,
    description: "Core training loops, evaluation metrics, and statistical testing functions",
    highlights: [
      "Optuna, grid, and random search implementations",
      "Unified metrics calculation and reporting",
      "Clark-West and Pesaran-Timmermann tests",
      "Out-of-sample evaluation framework"
    ],
    color: "accent-orange",
    utilities: [
      { module: "training_optuna.py", function: "Bayesian optimization with Optuna" },
      { module: "training_grid.py", function: "Systematic grid search" },
      { module: "training_random.py", function: "Random hyperparameter sampling" },
      { module: "oos_common.py", function: "Out-of-sample evaluation pipeline" },
      { module: "statistical_tests.py", function: "Statistical significance testing" },
      { module: "metrics_unified.py", function: "Performance metrics calculation" }
    ]
  }
]

export default function ImplementationShowcase() {
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
            <span className="text-text-primary">Implementation</span>
            <br />
            <span className="gradient-text">Architecture</span>
          </h2>
          <p className="max-w-4xl mx-auto text-xl text-text-secondary leading-relaxed">
            A complete, production-ready implementation built with PyTorch, featuring modular design, 
            comprehensive testing, and reproducible experiments. All code is available and documented at the GitHub repository below.
          </p>
        </motion.div>

        {/* Implementation Areas */}
        <div className="space-y-16">
          {implementationAreas.map((area, index) => {
            const Icon = area.icon
            
            return (
              <motion.div
                key={area.title}
                initial={{ opacity: 0, x: index % 2 === 0 ? -50 : 50 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.8, delay: index * 0.1 }}
                viewport={{ once: true }}
                className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-start"
              >
                {/* Content */}
                <div className={index % 2 === 1 ? 'lg:order-2' : ''}>
                  <div className={`inline-flex items-center justify-center w-12 h-12 rounded-lg bg-${area.color}/10 border border-${area.color}/20 mb-6`}>
                    <Icon className={`w-6 h-6 text-${area.color}`} />
                  </div>
                  
                  <h3 className="text-2xl font-bold text-text-primary mb-3">
                    {area.title}
                  </h3>
                  
                  <div className="mb-4">
                    <code className={`text-${area.color} bg-code-bg px-3 py-1 rounded-md text-sm font-mono`}>
                      {area.file}
                    </code>
                  </div>
                  
                  <p className="text-text-secondary mb-6 leading-relaxed">
                    {area.description}
                  </p>
                  
                  <div className="space-y-3">
                    {area.highlights.map((highlight, i) => (
                      <div key={i} className="flex items-start space-x-3">
                        <div className={`w-1.5 h-1.5 rounded-full bg-${area.color} mt-2 flex-shrink-0`}></div>
                        <span className="text-text-secondary text-sm">{highlight}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Details Panel */}
                <div className={index % 2 === 1 ? 'lg:order-1' : ''}>
                  <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-6">
                    {/* Models */}
                    {area.models && (
                      <div>
                        <h4 className="font-semibold text-text-primary mb-4 text-center">
                          Neural Network Models
                        </h4>
                        <div className="grid grid-cols-2 gap-3">
                          {area.models.map((model, i) => (
                            <div key={i} className="bg-bg-primary border border-bg-tertiary rounded-lg p-3">
                              <div className={`text-${area.color} font-semibold text-sm`}>{model.name}</div>
                              <div className="text-text-muted text-xs">{model.params}</div>
                              <div className="text-text-secondary text-xs mt-1 font-mono">{model.neurons}</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Search Spaces */}
                    {area.spaces && (
                      <div>
                        <h4 className="font-semibold text-text-primary mb-4 text-center">
                          Search Methods
                        </h4>
                        <div className="space-y-3">
                          {area.spaces.map((space, i) => (
                            <div key={i} className="bg-bg-primary border border-bg-tertiary rounded-lg p-3">
                              <div className={`text-${area.color} font-semibold text-sm`}>{space.method}</div>
                              <div className="text-text-secondary text-xs">{space.description}</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Experiments */}
                    {area.experiments && (
                      <div>
                        <h4 className="font-semibold text-text-primary mb-4 text-center">
                          Key Experiment Examples from the Repo
                        </h4>
                        <div className="space-y-3">
                          {area.experiments.map((exp, i) => (
                            <div key={i} className="bg-bg-primary border border-bg-tertiary rounded-lg p-3">
                              <div className={`text-${area.color} font-semibold text-sm font-mono`}>{exp.script}</div>
                              <div className="text-text-secondary text-xs">{exp.purpose}</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Utilities */}
                    {area.utilities && (
                      <div>
                        <h4 className="font-semibold text-text-primary mb-4 text-center">
                          Core Utilities
                        </h4>
                        <div className="space-y-3">
                          {area.utilities.map((util, i) => (
                            <div key={i} className="bg-bg-primary border border-bg-tertiary rounded-lg p-3">
                              <div className={`text-${area.color} font-semibold text-sm font-mono`}>{util.module}</div>
                              <div className="text-text-secondary text-xs">{util.function}</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </motion.div>
            )
          })}
        </div>

        {/* Code Statistics */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          viewport={{ once: true }}
          className="mt-20 bg-bg-secondary border border-bg-tertiary rounded-2xl p-8"
        >
          <h3 className="text-2xl font-bold text-center mb-8 gradient-text">
            Implementation Statistics
          </h3>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            <div className="text-center">
              <div className="text-3xl font-bold text-accent-blue mb-2">25+</div>
              <div className="text-text-muted text-sm">Experiment Scripts</div>
            </div>
            
            <div className="text-center">
              <div className="text-3xl font-bold text-accent-purple mb-2">8</div>
              <div className="text-text-muted text-sm">Neural Architectures</div>
            </div>
            
            <div className="text-center">
              <div className="text-3xl font-bold text-accent-green mb-2">3</div>
              <div className="text-text-muted text-sm">HPO Methods</div>
            </div>
            
            <div className="text-center">
              <div className="text-3xl font-bold text-accent-orange mb-2">100%</div>
              <div className="text-text-muted text-sm">Reproducible</div>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  )
}