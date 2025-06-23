'use client'

import { motion } from 'framer-motion'
import { Database, Brain, BarChart3, Zap, Target, TrendingUp } from 'lucide-react'

const features = [
  {
    icon: Database,
    title: "32 Financial Indicators",
    description: "Comprehensive dataset including valuation ratios, interest rates, credit measures, and technical indicators from ml_equity_premium_data.xlsx",
    color: "accent-blue"
  },
  {
    icon: Brain,
    title: "8 Neural Network Models",
    description: "Net1-Net5 standard architectures and DNet1-DNet3 with batch normalization, implemented in src/models/nns.py",
    color: "accent-purple"
  },
  {
    icon: BarChart3,
    title: "3 HPO Methods",
    description: "Bayesian optimization, grid search, and random search with domain-specific hyperparameter guidance",
    color: "accent-green"
  },
  {
    icon: Zap,
    title: "Real Implementation",
    description: "Actual code from src/experiments/, src/utils/, and src/configs/ with working examples and results",
    color: "accent-orange"
  },
  {
    icon: Target,
    title: "Out-of-Sample Evaluation",
    description: "Expanding window methodology with annual retraining and statistical significance testing",
    color: "accent-red"
  },
  {
    icon: TrendingUp,
    title: "Economic Value Analysis",
    description: "Market timing performance, Sharpe ratios, and economic significance beyond statistical metrics",
    color: "accent-blue"
  }
]

export default function OverviewSection() {
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
            <span className="gradient-text">Complete Neural Network</span>
            <br />
            <span className="text-text-primary">Implementation Guide</span>
          </h2>
          <p className="max-w-3xl mx-auto text-xl text-text-secondary leading-relaxed">
            This documentation covers the entire pipeline from data preprocessing to economic value analysis, 
            based on actual implementation for equity premium prediction using PyTorch and financial time series data.
          </p>
        </motion.div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => {
            const Icon = feature.icon
            
            return (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                viewport={{ once: true }}
                className="group"
              >
                <div className="card-hover bg-bg-primary border border-bg-tertiary rounded-xl p-6 h-full">
                  <div className={`inline-flex items-center justify-center w-12 h-12 rounded-lg bg-${feature.color}/10 border border-${feature.color}/20 mb-4`}>
                    <Icon className={`w-6 h-6 text-${feature.color}`} />
                  </div>
                  
                  <h3 className="text-xl font-semibold text-text-primary mb-3 group-hover:text-accent-blue transition-colors">
                    {feature.title}
                  </h3>
                  
                  <p className="text-text-secondary leading-relaxed">
                    {feature.description}
                  </p>
                </div>
              </motion.div>
            )
          })}
        </div>

        {/* Key Statistics */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          viewport={{ once: true }}
          className="mt-20 bg-bg-primary border border-bg-tertiary rounded-2xl p-8"
        >
          <h3 className="text-2xl font-bold text-center mb-8 gradient-text">
            Implementation Highlights
          </h3>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            <div className="text-center">
              <div className="text-3xl font-bold text-accent-blue mb-2">1,000+</div>
              <div className="text-text-muted">HPO Trials</div>
            </div>
            
            <div className="text-center">
              <div className="text-3xl font-bold text-accent-purple mb-2">25+</div>
              <div className="text-text-muted">Experiment Scripts</div>
            </div>
            
            <div className="text-center">
              <div className="text-3xl font-bold text-accent-green mb-2">70+</div>
              <div className="text-text-muted">Years of Data</div>
            </div>
            
            <div className="text-center">
              <div className="text-3xl font-bold text-accent-orange mb-2">3</div>
              <div className="text-text-muted">Validation Methods</div>
            </div>
          </div>
        </motion.div>

        {/* What You'll Learn */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          viewport={{ once: true }}
          className="mt-16 text-center"
        >
          <h3 className="text-2xl font-bold mb-8 text-text-primary">
            What You'll Learn
          </h3>
          
          <div className="max-w-4xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-6 text-left">
            <div className="space-y-3">
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 rounded-full bg-accent-blue mt-2"></div>
                <span className="text-text-secondary">Neural network architecture design for financial data</span>
              </div>
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 rounded-full bg-accent-blue mt-2"></div>
                <span className="text-text-secondary">Hyperparameter optimization strategies and implementation</span>
              </div>
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 rounded-full bg-accent-blue mt-2"></div>
                <span className="text-text-secondary">Temporal validation and out-of-sample evaluation</span>
              </div>
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 rounded-full bg-accent-blue mt-2"></div>
                <span className="text-text-secondary">Statistical significance testing in finance</span>
              </div>
            </div>
            
            <div className="space-y-3">
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 rounded-full bg-accent-purple mt-2"></div>
                <span className="text-text-secondary">Economic value analysis and market timing</span>
              </div>
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 rounded-full bg-accent-purple mt-2"></div>
                <span className="text-text-secondary">Financial domain-specific considerations</span>
              </div>
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 rounded-full bg-accent-purple mt-2"></div>
                <span className="text-text-secondary">Complete PyTorch implementation pipeline</span>
              </div>
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 rounded-full bg-accent-purple mt-2"></div>
                <span className="text-text-secondary">Reproducible research methodology</span>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  )
}