'use client'

import { motion } from 'framer-motion'
import { BookOpen, Github, Mail, ExternalLink } from 'lucide-react'
import { memo } from 'react'

function Footer() {
  return (
    <footer className="bg-bg-secondary border-t border-bg-tertiary lg:ml-80">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          {/* Brand */}
          <div className="col-span-1 md:col-span-2">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              viewport={{ once: true }}
              className="space-y-4"
            >
              <div className="flex items-center space-x-2">
                <div className="w-8 h-8 bg-gradient-to-br from-accent-blue to-accent-purple rounded-lg flex items-center justify-center">
                  <BookOpen className="w-5 h-5 text-white" />
                </div>
                <span className="font-bold text-lg gradient-text">
                  Neural Networks for Equity Premium Prediction
                </span>
              </div>
              
              <p className="text-text-secondary max-w-md leading-relaxed">
                Comprehensive documentation and interactive guide for implementing neural network 
                architectures in financial prediction, featuring multi-core parallel optimization 
                and real experimental results.
              </p>
              
              <div className="flex items-center space-x-4 mt-4">
                <a 
                  href="https://github.com/Aaronheno/Equity-Premium-Prediction-NN" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="flex items-center space-x-2 text-text-muted hover:text-accent-blue transition-colors"
                >
                  <Github className="w-4 h-4" />
                  <span className="text-sm">View Source Code</span>
                </a>
                <a 
                  href="mailto:aaronheno@gmail.com" 
                  className="flex items-center space-x-2 text-text-muted hover:text-accent-green transition-colors"
                >
                  <Mail className="w-4 h-4" />
                  <span className="text-sm">Contact</span>
                </a>
              </div>
              
            </motion.div>
          </div>

          {/* Documentation Links */}
          <div>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.1 }}
              viewport={{ once: true }}
            >
              <h3 className="font-semibold text-text-primary mb-4">Documentation</h3>
              <ul className="space-y-2">
                <li>
                  <a href="/data-setup" className="text-text-muted hover:text-accent-blue transition-colors text-sm">
                    Data & Problem Setup
                  </a>
                </li>
                <li>
                  <a href="/architecture" className="text-text-muted hover:text-accent-blue transition-colors text-sm">
                    Model Architecture
                  </a>
                </li>
                <li>
                  <a href="/hyperparameter-optimization" className="text-text-muted hover:text-accent-blue transition-colors text-sm">
                    Hyperparameter Optimization
                  </a>
                </li>
                <li>
                  <a href="/evaluation" className="text-text-muted hover:text-accent-blue transition-colors text-sm">
                    Evaluation & Testing
                  </a>
                </li>
                <li>
                  <a href="/complete-pipeline" className="text-text-muted hover:text-accent-blue transition-colors text-sm">
                    Complete Pipeline
                  </a>
                </li>
                <li>
                  <a href="/multithreading" className="text-text-muted hover:text-accent-purple transition-colors text-sm">
                    128-Core Optimization Plan
                  </a>
                </li>
              </ul>
            </motion.div>
          </div>

          {/* Implementation Links */}
          <div>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              viewport={{ once: true }}
            >
              <h3 className="font-semibold text-text-primary mb-4">Implementation</h3>
              <ul className="space-y-2">
                <li>
                  <div className="text-text-muted text-sm">
                    <code className="text-accent-green bg-code-bg px-1 py-0.5 rounded text-xs">src/models/nns.py</code>
                    <span className="ml-2">Neural Networks</span>
                  </div>
                </li>
                <li>
                  <div className="text-text-muted text-sm">
                    <code className="text-accent-blue bg-code-bg px-1 py-0.5 rounded text-xs">src/experiments/</code>
                    <span className="ml-2">Experiment Scripts</span>
                  </div>
                </li>
                <li>
                  <div className="text-text-muted text-sm">
                    <code className="text-accent-purple bg-code-bg px-1 py-0.5 rounded text-xs">src/utils/</code>
                    <span className="ml-2">Utility Functions</span>
                  </div>
                </li>
                <li>
                  <div className="text-text-muted text-sm">
                    <code className="text-accent-orange bg-code-bg px-1 py-0.5 rounded text-xs">runs/</code>
                    <span className="ml-2">Experiment Results</span>
                  </div>
                </li>
                <li>
                  <div className="text-text-muted text-sm">
                    <code className="text-accent-red bg-code-bg px-1 py-0.5 rounded text-xs">src/configs/search_spaces.py</code>
                    <span className="ml-2">HPO Configurations</span>
                  </div>
                </li>
              </ul>
            </motion.div>
          </div>
        </div>

        {/* Project Stats */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
          viewport={{ once: true }}
          className="mt-8 pt-8 border-t border-bg-tertiary"
        >
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-accent-blue">8</div>
              <div className="text-xs text-text-muted">Neural Network Models</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-accent-purple">32</div>
              <div className="text-xs text-text-muted">Financial Indicators</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-accent-green">1000+</div>
              <div className="text-xs text-text-muted">HPO Trials</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-accent-orange">48</div>
              <div className="text-xs text-text-muted">Threading-Optimized Files</div>
            </div>
          </div>
        </motion.div>

        {/* Bottom Bar */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          viewport={{ once: true }}
          className="mt-8 pt-8 border-t border-bg-tertiary flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0"
        >
          <div className="text-text-muted text-sm">
            © {new Date().getFullYear()} Aaron Hennessy. Built with Next.js and Tailwind CSS.
          </div>
          
          <div className="flex items-center space-x-6 text-sm">
            <a 
              href="/interactive-architecture" 
              className="text-text-muted hover:text-accent-blue transition-colors flex items-center space-x-1"
            >
              <span>Interactive Architecture</span>
              <ExternalLink className="w-3 h-3" />
            </a>
            
            <span className="text-text-muted">
              Powered by PyTorch
            </span>
          </div>
        </motion.div>
      </div>
    </footer>
  )
}

export default memo(Footer)