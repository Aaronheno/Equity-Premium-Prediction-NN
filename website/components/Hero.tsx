'use client'

import { motion } from 'framer-motion'
import { ArrowRight, Brain, TrendingUp, Zap } from 'lucide-react'
import Link from 'next/link'
import { memo } from 'react'

function Hero() {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden pt-20">
      {/* Background gradient */}
      <div className="absolute inset-0 bg-gradient-to-br from-bg-primary via-bg-secondary to-bg-primary" />
      
      {/* Static background elements for better performance */}
      <div className="absolute inset-0">
        <div className="absolute top-1/4 left-1/4 w-72 h-72 bg-accent-blue/5 rounded-full blur-3xl opacity-40" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-accent-purple/5 rounded-full blur-3xl opacity-50" />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className="space-y-8"
        >
          {/* Results Badge */}
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2, duration: 0.6 }}
            className="inline-flex items-center space-x-2 bg-accent-green/10 border border-accent-green/20 rounded-full px-4 py-2 text-sm font-medium text-accent-green"
          >
            <Zap className="w-4 h-4" />
            <span>Notable Results - Positive R² & Economic Outperformance Achieved</span>
          </motion.div>

          {/* Main heading */}
          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.8 }}
            className="text-4xl sm:text-5xl lg:text-7xl font-bold leading-tight"
          >
            <span className="gradient-text">Neural Networks</span>
            <br />
            <span className="text-text-primary">for Equity Premium</span>
            <br />
            <span className="text-text-secondary">Prediction</span>
          </motion.h1>

          {/* Subtitle */}
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5, duration: 0.8 }}
            className="max-w-3xl mx-auto text-xl sm:text-2xl text-text-secondary leading-relaxed"
          >
            <span className="text-accent-green font-bold">Key results:</span> DNN1 with intensive Bayesian optimization achieved
            <span className="text-accent-green font-semibold"> +0.75% out-of-sample R²</span> and 
            <span className="text-accent-blue font-semibold"> outperformed the HA benchmark</span> (26.45% vs 26.3% return, 0.86 vs 0.84 Sharpe).
            <span className="text-accent-purple font-semibold"> Computational intensity proved important</span> — 
            <span className="text-accent-orange font-semibold">1000+ optimization trials required</span>.
          </motion.p>

          {/* Breakthrough Stats */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7, duration: 0.8 }}
            className="flex flex-wrap justify-center gap-8 text-center"
          >
            <div className="flex items-center space-x-2 bg-accent-green/10 px-3 py-2 rounded-lg">
              <TrendingUp className="w-5 h-5 text-accent-green" />
              <span className="text-accent-green font-semibold">+0.75% R² (Positive!)</span>
            </div>
            <div className="flex items-center space-x-2 bg-accent-blue/10 px-3 py-2 rounded-lg">
              <Brain className="w-5 h-5 text-accent-blue" />
              <span className="text-accent-blue font-semibold">26.45% {'>'} 26.3% HA</span>
            </div>
            <div className="flex items-center space-x-2 bg-accent-purple/10 px-3 py-2 rounded-lg">
              <Zap className="w-5 h-5 text-accent-purple" />
              <span className="text-accent-purple font-semibold">1000+ Trials Critical</span>
            </div>
          </motion.div>

          {/* CTA Buttons */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.9, duration: 0.8 }}
            className="flex flex-col sm:flex-row gap-4 justify-center items-center"
          >
            <Link
              href="/data-setup"
              className="group inline-flex items-center space-x-2 bg-gradient-to-r from-accent-blue to-accent-purple text-white px-8 py-4 rounded-lg font-semibold text-lg hover:scale-105 transition-all duration-200 shadow-lg hover:shadow-xl"
            >
              <span>View Research</span>
              <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </Link>
            
            <Link
              href="/evaluation"
              className="group inline-flex items-center space-x-2 bg-transparent border-2 border-accent-green text-accent-green px-8 py-4 rounded-lg font-semibold text-lg hover:bg-accent-green hover:text-white transition-all duration-200"
            >
              <TrendingUp className="w-5 h-5" />
              <span>View Key Results</span>
            </Link>
          </motion.div>

          {/* Computational Requirements Insight */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.0, duration: 0.8 }}
            className="max-w-3xl mx-auto mt-8"
          >
            <div className="text-sm text-text-secondary bg-accent-purple/10 border border-accent-purple/20 rounded-lg p-4">
              <span className="text-accent-purple font-semibold">⚡ Computational Finding:</span> Results achieved through intensive computational resources. 
              DNN1 required 1000+ Bayesian optimization trials to achieve positive R² and economic outperformance. 
              <span className="text-accent-purple font-semibold">Computational constraints limited analysis to one model</span> — 
              adequate hardware important for optimizing neural network performance.
            </div>
          </motion.div>

          {/* Code snippet preview */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.1, duration: 0.8 }}
            className="max-w-2xl mx-auto mt-8"
          >
            <div className="bg-code-bg border border-code-border rounded-lg overflow-hidden">
              <div className="bg-bg-tertiary px-4 py-2 border-b border-code-border">
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 rounded-full bg-accent-red"></div>
                  <div className="w-3 h-3 rounded-full bg-accent-orange"></div>
                  <div className="w-3 h-3 rounded-full bg-accent-green"></div>
                  <span className="ml-2 text-sm text-text-muted font-mono">src/models/nns.py</span>
                </div>
              </div>
              <div className="p-4 font-mono text-sm">
                <div className="text-accent-purple">class</div>
                <span className="text-accent-blue"> DNet1</span>
                <span className="text-text-primary">(</span>
                <span className="text-accent-green">nn.Module</span>
                <span className="text-text-primary">):</span>
                <br />
                <span className="text-text-muted">    # Deep network with batch normalization</span>
                <br />
                <span className="text-text-muted">    # TOP PERFORMING MODEL: +0.75% R², 26.45% return</span>
                <br />
                <span className="text-text-muted">    # 4 hidden layers + BatchNorm blocks</span>
              </div>
            </div>
          </motion.div>
        </motion.div>
      </div>
    </section>
  )
}

export default memo(Hero)