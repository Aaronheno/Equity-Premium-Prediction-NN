'use client'

import { motion } from 'framer-motion'
import { Code, FileText, BarChart3, Zap, ArrowRight, Cpu } from 'lucide-react'
import Link from 'next/link'

const features = [
  {
    icon: Code,
    title: "Interactive Code Examples",
    description: "Copy-paste ready code blocks with syntax highlighting, extracted directly from the actual implementation",
    items: [
      "Model architecture definitions",
      "Training loop implementations", 
      "Hyperparameter optimization scripts",
      "Evaluation and testing functions"
    ],
    link: "/data-setup",
    color: "accent-blue"
  },
  {
    icon: FileText,
    title: "Mathematical Foundations",
    description: "Beautiful mathematical formulas with detailed explanations, rendered with KaTeX for crystal-clear presentation",
    items: [
      "Forward pass computations",
      "Backpropagation derivations",
      "Loss function mathematics",
      "Optimizer update rules"
    ],
    link: "/forward-pass",
    color: "accent-purple"
  },
  {
    icon: BarChart3,
    title: "Real Results & Analysis",
    description: "Actual performance metrics and statistical tests from completed experiments and out-of-sample validation",
    items: [
      "HPO trial results and convergence",
      "Out-of-sample performance metrics",
      "Economic value analysis",
      "Statistical significance testing"
    ],
    link: "/evaluation",
    color: "accent-green"
  },
  {
    icon: Zap,
    title: "Interactive Visualizations",
    description: "Dynamic diagrams and interactive components to understand neural network behavior and training dynamics",
    items: [
      "Architecture visualization tool",
      "Hyperparameter space exploration",
      "Training progress animations",
      "Performance comparison charts"
    ],
    link: "/interactive-architecture",
    color: "accent-orange"
  },
  {
    icon: Cpu,
    title: "128-Core Parallel Optimization",
    description: "Comprehensive multithreading implementation plan with documented speedup potential for all experiments",
    items: [
      "Perfect parallelization for Grid/Random Search",
      "Model-level parallelization for Bayesian HPO",
      "Time-series coordination optimization", 
      "20-256x speedup across experiment pipeline"
    ],
    link: "/multithreading",
    color: "accent-red"
  }
]

export default function KeyFeatures() {
  return (
    <section className="py-24 bg-bg-secondary">
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
            <span className="gradient-text">Key Features</span>
            <br />
            <span className="text-text-primary">& Capabilities</span>
          </h2>
          <p className="max-w-3xl mx-auto text-xl text-text-secondary leading-relaxed">
            Everything you need to understand, implement, and optimize neural networks for financial prediction,
            from mathematical foundations to interactive visualizations.
          </p>
        </motion.div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-16">
          {features.map((feature, index) => {
            const Icon = feature.icon
            
            return (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: index * 0.2 }}
                viewport={{ once: true }}
                className="group"
              >
                <div className="card-hover bg-bg-primary border border-bg-tertiary rounded-xl p-8 h-full">
                  {/* Icon and Title */}
                  <div className="flex items-start space-x-4 mb-6">
                    <div className={`flex-shrink-0 inline-flex items-center justify-center w-12 h-12 rounded-lg bg-${feature.color}/10 border border-${feature.color}/20`}>
                      <Icon className={`w-6 h-6 text-${feature.color}`} />
                    </div>
                    <div>
                      <h3 className="text-xl font-semibold text-text-primary mb-2 group-hover:text-accent-blue transition-colors">
                        {feature.title}
                      </h3>
                      <p className="text-text-secondary leading-relaxed">
                        {feature.description}
                      </p>
                    </div>
                  </div>

                  {/* Feature Items */}
                  <div className="space-y-3 mb-6">
                    {feature.items.map((item, itemIndex) => (
                      <div key={itemIndex} className="flex items-center space-x-3">
                        <div className={`w-1.5 h-1.5 rounded-full bg-${feature.color}`}></div>
                        <span className="text-text-secondary text-sm">{item}</span>
                      </div>
                    ))}
                  </div>

                  {/* Learn More Link */}
                  <Link
                    href={feature.link}
                    className={`inline-flex items-center space-x-2 text-${feature.color} hover:text-${feature.color} transition-colors group/link`}
                  >
                    <span className="font-medium">Learn More</span>
                    <ArrowRight className="w-4 h-4 group-hover/link:translate-x-1 transition-transform" />
                  </Link>
                </div>
              </motion.div>
            )
          })}
        </div>

        {/* Implementation Showcase */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          viewport={{ once: true }}
          className="bg-bg-primary border border-bg-tertiary rounded-2xl p-8"
        >
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-center">
            {/* Content */}
            <div>
              <h3 className="text-2xl font-bold mb-4">
                <span className="gradient-text">Complete Implementation</span>
              </h3>
              <p className="text-text-secondary mb-6 leading-relaxed">
                This documentation is based on a real, working implementation for equity premium prediction
                with comprehensive 128-core parallel optimization. Every code example, configuration, and 
                result shown is extracted from the actual codebase.
              </p>
              
              <div className="space-y-4">
                <div className="flex items-center space-x-3">
                  <div className="w-2 h-2 rounded-full bg-accent-green"></div>
                  <span className="text-text-secondary">
                    <code className="text-accent-green bg-code-bg px-2 py-1 rounded text-sm">src/models/nns.py</code> - 
                    All 8 neural network architectures
                  </span>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="w-2 h-2 rounded-full bg-accent-blue"></div>
                  <span className="text-text-secondary">
                    <code className="text-accent-blue bg-code-bg px-2 py-1 rounded text-sm">src/experiments/</code> - 
                    48 threading-optimized files with 20-256x speedup
                  </span>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="w-2 h-2 rounded-full bg-accent-purple"></div>
                  <span className="text-text-secondary">
                    <code className="text-accent-purple bg-code-bg px-2 py-1 rounded text-sm">runs/</code> - 
                    Thousands of HPO trials and performance metrics
                  </span>
                </div>
              </div>
            </div>

            {/* Code Preview */}
            <div>
              <div className="bg-code-bg border border-code-border rounded-lg overflow-hidden">
                <div className="bg-bg-tertiary px-4 py-2 border-b border-code-border">
                  <div className="flex items-center space-x-2">
                    <div className="w-3 h-3 rounded-full bg-accent-red"></div>
                    <div className="w-3 h-3 rounded-full bg-accent-orange"></div>
                    <div className="w-3 h-3 rounded-full bg-accent-green"></div>
                    <span className="ml-2 text-sm text-text-muted font-mono">src/experiments/bayes_oos_1.py</span>
                  </div>
                </div>
                <div className="p-4 font-mono text-sm">
                  <div className="text-accent-purple">from</div>
                  <span className="text-text-primary"> src.utils.oos_common </span>
                  <span className="text-accent-purple">import</span>
                  <span className="text-accent-green"> run_oos_experiment</span>
                  <br />
                  <div className="text-accent-purple">from</div>
                  <span className="text-text-primary"> src.configs.search_spaces </span>
                  <span className="text-accent-purple">import</span>
                  <span className="text-accent-green"> BAYES_OOS</span>
                  <br /><br />
                  <span className="text-text-muted"># Real out-of-sample experiment</span>
                  <br />
                  <span className="text-accent-blue">run_oos_experiment</span>
                  <span className="text-text-primary">(</span>
                  <br />
                  <span className="text-text-primary">    search_method=</span>
                  <span className="text-accent-orange">'bayes'</span>
                  <span className="text-text-primary">,</span>
                  <br />
                  <span className="text-text-primary">    config=</span>
                  <span className="text-accent-green">BAYES_OOS</span>
                  <br />
                  <span className="text-text-primary">)</span>
                </div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* CTA Section */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          viewport={{ once: true }}
          className="mt-16 text-center"
        >
          <h3 className="text-2xl font-bold mb-6 text-text-primary">
            Ready to Start Learning?
          </h3>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              href="/data-setup"
              className="inline-flex items-center space-x-2 bg-gradient-to-r from-accent-blue to-accent-purple text-white px-8 py-4 rounded-lg font-semibold hover:scale-105 transition-all duration-200 shadow-lg hover:shadow-xl"
            >
              <span>Begin with Data Setup</span>
              <ArrowRight className="w-5 h-5" />
            </Link>
            
            <Link
              href="/interactive-architecture"
              className="inline-flex items-center space-x-2 bg-transparent border-2 border-accent-green text-accent-green px-8 py-4 rounded-lg font-semibold hover:bg-accent-green hover:text-white transition-all duration-200"
            >
              <BarChart3 className="w-5 h-5" />
              <span>Explore Architectures</span>
            </Link>
          </div>
        </motion.div>
      </div>
    </section>
  )
}