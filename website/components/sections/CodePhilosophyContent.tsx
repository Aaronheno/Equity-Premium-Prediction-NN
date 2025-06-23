'use client'

import { motion } from 'framer-motion'
import { Code2, Layers, Settings, Zap, GitBranch, Brain, Target, CheckCircle } from 'lucide-react'
import CodeBlock from '@/components/shared/CodeBlock'
import NavigationButtons from '@/components/shared/NavigationButtons'

const designPrinciples = [
  {
    title: 'Modularity',
    icon: Layers,
    color: 'accent-blue',
    description: 'Reusable components across 8 model architectures',
    benefits: [
      'Single _Base class supports Net1-Net5',
      'DBlock pattern for all deep networks',
      'Consistent parameter interfaces',
      'Easy architecture comparison'
    ],
    example: 'All standard networks inherit from _Base with just layer configuration'
  },
  {
    title: 'Maintainability',
    icon: Settings,
    color: 'accent-green',
    description: 'Single source of truth for each concept',
    benefits: [
      'Centralized activation function mapping',
      'Unified training loop implementations',
      'Consistent hyperparameter naming',
      'Shared validation and metrics'
    ],
    example: 'Change dropout behavior once in _Base, affects all standard networks'
  },
  {
    title: 'Experimentation',
    icon: Zap,
    color: 'accent-orange',
    description: 'Easy hyperparameter search across models',
    benefits: [
      'Automated model instantiation',
      'Consistent search space definitions',
      'Parallel model evaluation',
      'Reproducible experiment tracking'
    ],
    example: 'Search spaces in config files work with all 8 architectures'
  },
  {
    title: 'Performance',
    icon: Target,
    color: 'accent-purple',
    description: 'Optimized PyTorch Sequential modules',
    benefits: [
      'Efficient forward/backward passes',
      'Automatic GPU utilization',
      'Memory-optimized implementations',
      'Vectorized operations'
    ],
    example: 'Sequential modules enable PyTorch optimizations automatically'
  }
]

const codeExamples = [
  {
    title: 'Architecture Flexibility',
    description: 'How the same base class supports different complexities',
    code: `# Single class, multiple architectures
Net1: _Base([30, 64, 1])           # 1 hidden layer
Net2: _Base([30, 64, 32, 1])       # 2 hidden layers  
Net3: _Base([30, 64, 32, 16, 1])   # 3 hidden layers
Net5: _Base([30, 96, 64, 48, 32, 16, 1])  # 5 hidden layers

# Same interface, different capabilities
all_models = [Net1, Net2, Net3, Net4, Net5]
for ModelClass in all_models:
    model = ModelClass(n_feature=30, n_output=1, dropout=0.3)
    predictions = model(financial_data)`
  },
  {
    title: 'Hyperparameter Consistency',
    description: 'Unified parameter naming across all experiments',
    code: `# From src/configs/search_spaces.py
# Same parameter structure for all models

BAYES = {
    "Net1": {"hpo_config_fn": create_config("Net1", {"n_hidden1": IntDistribution(16, 256)})},
    "Net2": {"hpo_config_fn": create_config("Net2", {"n_hidden1": IntDistribution(16, 192), 
                                                     "n_hidden2": IntDistribution(8, 128)})},
    "Net3": {"hpo_config_fn": create_config("Net3", {"n_hidden1": IntDistribution(16, 128),
                                                     "n_hidden2": IntDistribution(8, 96),
                                                     "n_hidden3": IntDistribution(4, 64)})}
}

# Common parameters across all models:
# - optimizer, lr, weight_decay, l1_lambda, dropout, batch_size`
  },
  {
    title: 'Experiment Modularity',
    description: 'How the same training logic works for all optimization methods',
    code: `# From src/utils/oos_common.py
# Single function handles all HPO methods

def run_oos_experiment(nn_model_configs, hpo_general_config):
    for model_name, config in nn_model_configs.items():
        model_class = config['model_class']           # Net1, Net2, etc.
        hpo_function = config['hpo_function']         # grid, random, or bayes
        regressor_class = config['regressor_class']   # Skorch wrapper
        search_space = config['search_space_config_or_fn']
        
        # Same training logic regardless of HPO method
        best_params = hpo_function(model_class, search_space, train_data)
        final_model = regressor_class(model_class, **best_params)
        predictions = final_model.predict(test_data)`
  }
]

export default function CodePhilosophyContent() {
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
            <span className="gradient-text">Implementation Philosophy</span>
          </h1>
          <p className="max-w-4xl mx-auto text-xl text-text-secondary leading-relaxed">
            Understanding the architectural decisions behind the neural network implementation: 
            why the code is structured for modularity, maintainability, and experimental flexibility.
          </p>
        </motion.div>

        {/* Philosophy Overview */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Brain className="w-8 h-8 text-accent-blue mr-3" />
            Code vs. Learning Examples
          </h2>
          
          <div className="bg-accent-blue/5 border border-accent-blue/20 rounded-xl p-8 mb-8">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div>
                <h3 className="text-xl font-semibold text-text-primary mb-4">Why Two Types of Code Examples?</h3>
                <div className="space-y-4 text-text-secondary">
                  <div className="flex items-start space-x-3">
                    <CheckCircle className="w-5 h-5 text-accent-green mt-0.5 flex-shrink-0" />
                    <span><strong>Conceptual Examples:</strong> Step-by-step educational code that shows exactly what happens inside each operation</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <CheckCircle className="w-5 h-5 text-accent-blue mt-0.5 flex-shrink-0" />
                    <span><strong>Actual Implementation:</strong> Production-ready code optimized for research and experimentation</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <CheckCircle className="w-5 h-5 text-accent-purple mt-0.5 flex-shrink-0" />
                    <span><strong>Hybrid Approach:</strong> Learn concepts first, then see how they're efficiently implemented</span>
                  </div>
                </div>
              </div>
              
              <div>
                <h3 className="text-xl font-semibold text-text-primary mb-4">Production Code Priorities:</h3>
                <div className="bg-bg-secondary border border-bg-tertiary rounded-lg p-4 space-y-3 text-sm">
                  <div><strong>Research Efficiency:</strong> Run 8 models × 3 HPO methods × multiple experiments</div>
                  <div><strong>Code Reuse:</strong> Single implementation supports multiple use cases</div>
                  <div><strong>Maintainability:</strong> Changes propagate automatically across models</div>
                  <div><strong>Reproducibility:</strong> Consistent interfaces ensure reliable results</div>
                </div>
              </div>
            </div>
          </div>
        </motion.section>

        {/* Design Principles */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <GitBranch className="w-8 h-8 text-accent-green mr-3" />
            Core Design Principles
          </h2>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {designPrinciples.map((principle, index) => {
              const Icon = principle.icon
              return (
                <motion.div
                  key={principle.title}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: 0.4 + index * 0.1 }}
                  className={`bg-${principle.color}/5 border border-${principle.color}/20 rounded-xl p-6`}
                >
                  <div className="flex items-center space-x-3 mb-4">
                    <div className={`inline-flex items-center justify-center w-12 h-12 rounded-lg bg-${principle.color}/10 border border-${principle.color}/20`}>
                      <Icon className={`w-6 h-6 text-${principle.color}`} />
                    </div>
                    <div>
                      <h3 className="text-xl font-bold text-text-primary">{principle.title}</h3>
                      <p className="text-text-secondary">{principle.description}</p>
                    </div>
                  </div>

                  <div className="mb-4">
                    <h4 className="font-semibold text-text-primary mb-2">Key Benefits:</h4>
                    <ul className="space-y-1 text-sm text-text-secondary">
                      {principle.benefits.map((benefit, i) => (
                        <li key={i} className="flex items-start space-x-2">
                          <span className={`w-1 h-1 rounded-full bg-${principle.color} mt-2 flex-shrink-0`}></span>
                          <span>{benefit}</span>
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div className={`bg-${principle.color}/10 border border-${principle.color}/20 rounded p-3`}>
                    <div className={`text-${principle.color} text-xs font-medium mb-1`}>Example Impact:</div>
                    <div className="text-text-secondary text-xs">{principle.example}</div>
                  </div>
                </motion.div>
              )
            })}
          </div>
        </motion.section>

        {/* Code Examples */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Code2 className="w-8 h-8 text-accent-purple mr-3" />
            Design in Practice
          </h2>
          
          <div className="space-y-8">
            {codeExamples.map((example, index) => (
              <motion.div
                key={example.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.5 + index * 0.1 }}
                className="bg-bg-secondary border border-bg-tertiary rounded-xl p-6"
              >
                <h3 className="text-xl font-semibold text-text-primary mb-3">{example.title}</h3>
                <p className="text-text-secondary mb-6">{example.description}</p>
                
                <CodeBlock
                  language="python"
                  title={example.title}
                  codeType="actual"
                  code={example.code}
                />
              </motion.div>
            ))}
          </div>
        </motion.section>

        {/* Implementation Benefits */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.5 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8">Real-World Impact</h2>
          
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
              <div className="text-center">
                <div className="text-3xl font-bold text-accent-blue mb-2">8</div>
                <div className="text-sm text-text-muted">Neural Architectures</div>
                <div className="text-xs text-text-secondary mt-1">Single codebase</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-accent-green mb-2">3</div>
                <div className="text-sm text-text-muted">HPO Methods</div>
                <div className="text-xs text-text-secondary mt-1">Unified interface</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-accent-orange mb-2">25+</div>
                <div className="text-sm text-text-muted">Experiments</div>
                <div className="text-xs text-text-secondary mt-1">Shared utilities</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-accent-purple mb-2">1M+</div>
                <div className="text-sm text-text-muted">Model Configurations</div>
                <div className="text-xs text-text-secondary mt-1">Automated testing</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-accent-red mb-2">30</div>
                <div className="text-sm text-text-muted">Financial Indicators</div>
                <div className="text-xs text-text-secondary mt-1">Consistent processing</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-accent-cyan mb-2">100%</div>
                <div className="text-sm text-text-muted">Reproducible</div>
                <div className="text-xs text-text-secondary mt-1">Standardized pipeline</div>
              </div>
            </div>
            
            <div className="mt-8 text-center">
              <p className="text-text-secondary text-lg">
                This architectural approach enabled comprehensive research across multiple model types, 
                optimization methods, and validation schemes while maintaining code quality and reproducibility.
              </p>
            </div>
          </div>
        </motion.section>

        {/* Navigation */}
        <NavigationButtons 
          prevHref="/evaluation"
          prevLabel="Evaluation & Interpretation"
          nextHref="/complete-pipeline"
          nextLabel="Complete Pipeline"
        />
      </div>
    </div>
  )
}