'use client'

import { motion } from 'framer-motion'
import { FileText, Database, Brain, BarChart3, TrendingUp, Target } from 'lucide-react'

export default function ResearchOverview() {
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
            <span className="gradient-text">Research Overview</span>
          </h2>
          <p className="max-w-4xl mx-auto text-xl text-text-secondary leading-relaxed">
            This research implements and evaluates neural network architectures for equity premium prediction 
            using 32 financial indicators across multiple decades of market data, with rigorous out-of-sample testing,
            statistical validation, and comprehensive parallel optimization.
          </p>
        </motion.div>

        {/* Research Methodology */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          viewport={{ once: true }}
          className="grid grid-cols-1 lg:grid-cols-2 gap-12 mb-20"
        >
          {/* Problem Statement */}
          <div className="bg-bg-primary border border-bg-tertiary rounded-xl p-8">
            <div className="flex items-center space-x-3 mb-6">
              <Target className="w-8 h-8 text-accent-blue" />
              <h3 className="text-2xl font-bold text-text-primary">Research Question</h3>
            </div>
            <p className="text-text-secondary mb-6 leading-relaxed">
              Can neural networks with rigorous hyperparameter optimization and temporal validation 
              significantly outperform traditional benchmark models in predicting equity premium returns?
            </p>
            <div className="space-y-3">
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 rounded-full bg-accent-blue mt-2"></div>
                <span className="text-text-secondary text-sm">Target: Log equity premium (market return - risk-free rate)</span>
              </div>
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 rounded-full bg-accent-blue mt-2"></div>
                <span className="text-text-secondary text-sm">Prediction horizon: Monthly out-of-sample forecasts</span>
              </div>
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 rounded-full bg-accent-blue mt-2"></div>
                <span className="text-text-secondary text-sm">Benchmarks: Historical Average (HA) and Combination Forecast (CF) models</span>
              </div>
            </div>
          </div>

          {/* Methodology */}
          <div className="bg-bg-primary border border-bg-tertiary rounded-xl p-8">
            <div className="flex items-center space-x-3 mb-6">
              <FileText className="w-8 h-8 text-accent-purple" />
              <h3 className="text-2xl font-bold text-text-primary">Methodology</h3>
            </div>
            <p className="text-text-secondary mb-6 leading-relaxed">
              Comprehensive experimental design with temporal validation, preventing look-ahead bias 
              through expanding window out-of-sample evaluation and annual hyperparameter reoptimization.
            </p>
            <div className="space-y-3">
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 rounded-full bg-accent-purple mt-2"></div>
                <span className="text-text-secondary text-sm">Expanding window validation (no future data leakage)</span>
              </div>
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 rounded-full bg-accent-purple mt-2"></div>
                <span className="text-text-secondary text-sm">Annual hyperparameter reoptimization</span>
              </div>
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 rounded-full bg-accent-purple mt-2"></div>
                <span className="text-text-secondary text-sm">Statistical significance testing (Clark-West, Pesaran-Timmermann)</span>
              </div>
              <div className="flex items-start space-x-3">
                <div className="w-2 h-2 rounded-full bg-accent-purple mt-2"></div>
                <span className="text-text-secondary text-sm">Comprehensive Evaluation: Statistical (OOS R², Success Ratio, MSFE-adjusted) and economic metrics (Average Returns, Sharpe Ratio, CER)</span>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Key Research Components */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            viewport={{ once: true }}
            className="bg-bg-primary border border-bg-tertiary rounded-xl p-6 text-center"
          >
            <Database className="w-12 h-12 text-accent-green mx-auto mb-4" />
            <h4 className="text-xl font-semibold text-text-primary mb-3">Data & Features</h4>
            <p className="text-text-secondary text-sm mb-4">
              32 financial indicators including valuation ratios, interest rates, credit spreads, 
              and technical indicators from decades of market data.
            </p>
            <div className="text-2xl font-bold text-accent-green">32</div>
            <div className="text-xs text-text-muted">Financial Predictors</div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            viewport={{ once: true }}
            className="bg-bg-primary border border-bg-tertiary rounded-xl p-6 text-center"
          >
            <Brain className="w-12 h-12 text-accent-blue mx-auto mb-4" />
            <h4 className="text-xl font-semibold text-text-primary mb-3">Neural Architectures</h4>
            <p className="text-text-secondary text-sm mb-4">
              8 neural network models from simple single-layer to sophisticated 5-layer architectures 
              with batch normalization and advanced regularization.
            </p>
            <div className="text-2xl font-bold text-accent-blue">8</div>
            <div className="text-xs text-text-muted">Model Architectures</div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            viewport={{ once: true }}
            className="bg-bg-primary border border-bg-tertiary rounded-xl p-6 text-center"
          >
            <BarChart3 className="w-12 h-12 text-accent-orange mx-auto mb-4" />
            <h4 className="text-xl font-semibold text-text-primary mb-3">Optimization</h4>
            <p className="text-text-secondary text-sm mb-4">
              Comprehensive hyperparameter optimization using Bayesian, grid, and random search 
              methods with parallel implementation and financial domain-specific parameter spaces.
            </p>
            <div className="text-2xl font-bold text-accent-orange">1000+</div>
            <div className="text-xs text-text-muted">HPO Trials</div>
          </motion.div>
        </div>

        {/* Research Timeline */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          viewport={{ once: true }}
          className="mt-20 bg-bg-primary border border-bg-tertiary rounded-2xl p-8"
        >
          <h3 className="text-2xl font-bold text-center mb-8 gradient-text">
            Implementation Pipeline
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="w-12 h-12 bg-accent-blue/20 border-2 border-accent-blue rounded-full flex items-center justify-center mx-auto mb-3">
                <span className="text-accent-blue font-bold">1</span>
              </div>
              <h4 className="font-semibold text-text-primary mb-2">Data Preparation</h4>
              <p className="text-text-muted text-sm">Feature engineering, scaling, temporal splits</p>
            </div>
            
            <div className="text-center">
              <div className="w-12 h-12 bg-accent-purple/20 border-2 border-accent-purple rounded-full flex items-center justify-center mx-auto mb-3">
                <span className="text-accent-purple font-bold">2</span>
              </div>
              <h4 className="font-semibold text-text-primary mb-2">Model Development</h4>
              <p className="text-text-muted text-sm">Architecture design, implementation, testing</p>
            </div>
            
            <div className="text-center">
              <div className="w-12 h-12 bg-accent-green/20 border-2 border-accent-green rounded-full flex items-center justify-center mx-auto mb-3">
                <span className="text-accent-green font-bold">3</span>
              </div>
              <h4 className="font-semibold text-text-primary mb-2">Optimization</h4>
              <p className="text-text-muted text-sm">Hyperparameter tuning, validation, selection</p>
            </div>
            
            <div className="text-center">
              <div className="w-12 h-12 bg-accent-orange/20 border-2 border-accent-orange rounded-full flex items-center justify-center mx-auto mb-3">
                <span className="text-accent-orange font-bold">4</span>
              </div>
              <h4 className="font-semibold text-text-primary mb-2">Evaluation</h4>
              <p className="text-text-muted text-sm">OOS testing, statistical significance, economic value</p>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  )
}