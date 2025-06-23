'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { Menu, X, BookOpen, Code, BarChart3, Database, Layers, ArrowRight, Calculator, GitBranch, Zap, Settings, Target, Brain, TrendingUp, Workflow, Cpu, Mail, Github } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'

const navigationItems = [
  { 
    name: 'Introduction', 
    href: '/', 
    icon: BookOpen,
    description: 'Overview and project introduction'
  },
  { 
    name: 'Data & Setup', 
    href: '/data-setup', 
    icon: Database,
    description: '32 financial indicators and problem formulation'
  },
  { 
    name: 'Data Preprocessing', 
    href: '/preprocessing', 
    icon: Settings,
    description: 'Scaling, temporal splits, and feature engineering'
  },
  { 
    name: 'Model Architecture', 
    href: '/architecture', 
    icon: Layers,
    description: 'Net1-Net5 and DNet1-DNet3 models'
  },
  { 
    name: 'Forward Pass', 
    href: '/forward-pass', 
    icon: ArrowRight,
    description: 'Information flow through neural networks'
  },
  { 
    name: 'Loss Calculation', 
    href: '/loss-calculation', 
    icon: Calculator,
    description: 'MSE, L1, and L2 regularization'
  },
  { 
    name: 'Backpropagation', 
    href: '/backpropagation', 
    icon: GitBranch,
    description: 'Gradient computation and chain rule'
  },
  { 
    name: 'Optimization', 
    href: '/optimization', 
    icon: Zap,
    description: 'Adam, RMSprop, SGD optimizers'
  },
  { 
    name: 'Hyperparameter Optimization', 
    href: '/hyperparameter-optimization', 
    icon: Settings,
    description: 'Bayesian, grid, and random search'
  },
  { 
    name: 'Making Predictions', 
    href: '/predictions', 
    icon: Target,
    description: 'Model inference and output processing'
  },
  { 
    name: 'Evaluation', 
    href: '/evaluation', 
    icon: BarChart3,
    description: 'Performance metrics and statistical tests'
  },
  { 
    name: 'Complete Pipeline', 
    href: '/complete-pipeline', 
    icon: Workflow,
    description: 'End-to-end implementation'
  },
  { 
    name: 'Interactive Architecture', 
    href: '/interactive-architecture', 
    icon: Brain,
    description: 'Visualize all 8 model architectures'
  },
  { 
    name: '128-Core Optimization Plan', 
    href: '/multithreading', 
    icon: Cpu,
    description: 'Comprehensive parallel implementation plan with 20-256x speedup'
  },
]

export default function Navigation() {
  const [isOpen, setIsOpen] = useState(false)
  const [scrolled, setScrolled] = useState(false)
  const pathname = usePathname()

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50)
    }

    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  const toggleMenu = () => setIsOpen(!isOpen)
  const closeMenu = () => setIsOpen(false)

  return (
    <>
      {/* Top Header - Mobile and Desktop for Source/Contact */}
      <motion.nav
        initial={{ y: -100 }}
        animate={{ y: 0 }}
        className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
          scrolled 
            ? 'bg-bg-primary/95 backdrop-blur-md border-b border-bg-tertiary' 
            : 'bg-transparent'
        }`}
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            {/* Logo */}
            <Link 
              href="/" 
              className="flex items-center space-x-2 group lg:hidden"
              onClick={closeMenu}
            >
              <div className="w-8 h-8 bg-gradient-to-br from-accent-blue to-accent-purple rounded-lg flex items-center justify-center">
                <BookOpen className="w-5 h-5 text-white" />
              </div>
              <span className="font-bold text-lg gradient-text group-hover:scale-105 transition-transform">
                Optimizing Neural Networks for Equity Premium Prediction
              </span>
            </Link>
            
            {/* Centered Title - Hidden on mobile */}
            <div className="hidden lg:flex flex-1 justify-center">
              <span className="font-bold text-2xl gradient-text text-center">
                Optimizing Neural Networks for Equity Premium Prediction
              </span>
            </div>
            
            {/* Source Code and Contact Links - Always visible */}
            <div className="flex items-center space-x-6">
              <a
                href="https://github.com/Aaronheno/Equity-Premium-Prediction-NN"
                target="_blank"
                rel="noopener noreferrer"
                className="group flex items-center space-x-2 text-text-muted hover:text-accent-blue transition-all duration-200"
                title="View Source Code"
              >
                <div className="p-1.5 bg-bg-secondary/50 group-hover:bg-accent-blue/10 rounded-lg transition-colors border border-bg-tertiary group-hover:border-accent-blue/20">
                  <Github className="w-4 h-4" />
                </div>
                <span className="text-sm font-medium hidden sm:inline">Source Code</span>
              </a>
              
              <a
                href="mailto:aaronheno@gmail.com"
                className="group flex items-center space-x-2 text-text-muted hover:text-accent-green transition-all duration-200"
                title="Contact"
              >
                <div className="p-1.5 bg-bg-secondary/50 group-hover:bg-accent-green/10 rounded-lg transition-colors border border-bg-tertiary group-hover:border-accent-green/20">
                  <Mail className="w-4 h-4" />
                </div>
                <span className="text-sm font-medium hidden sm:inline">Contact</span>
              </a>
              
              {/* Mobile menu button */}
              <button
                onClick={toggleMenu}
                className="p-2 rounded-md text-text-secondary hover:text-accent-blue transition-colors lg:hidden"
              >
                {isOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
              </button>
            </div>
          </div>
        </div>

        {/* Mobile Navigation */}
        <AnimatePresence>
          {isOpen && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="bg-bg-secondary border-t border-bg-tertiary"
            >
              <div className="px-4 py-2 space-y-1 max-h-[70vh] overflow-y-auto">
                {navigationItems.map((item) => {
                  const Icon = item.icon
                  const isActive = pathname === item.href
                  
                  return (
                    <Link
                      key={item.name}
                      href={item.href}
                      onClick={closeMenu}
                      className={`block px-3 py-2 rounded-md text-sm transition-colors ${
                        isActive 
                          ? 'text-accent-blue bg-accent-blue/10' 
                          : 'text-text-secondary hover:text-accent-blue hover:bg-bg-tertiary'
                      }`}
                    >
                      <div className="flex items-start space-x-2">
                        <Icon className="w-4 h-4 mt-0.5 flex-shrink-0" />
                        <div>
                          <div className="font-medium">{item.name}</div>
                          <div className="text-xs text-text-muted">{item.description}</div>
                        </div>
                      </div>
                    </Link>
                  )
                })}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.nav>

      {/* Left Sidebar Navigation - Desktop */}
      <motion.nav
        initial={{ x: -100 }}
        animate={{ x: 0 }}
        className="hidden lg:block fixed top-16 left-0 h-[calc(100vh-4rem)] w-80 bg-bg-secondary/95 backdrop-blur-md border-r border-bg-tertiary z-40 overflow-y-auto"
      >
        <div className="p-6">
          {/* Logo */}
          <Link 
            href="/" 
            className="flex items-center space-x-2 group mb-8"
          >
            <div className="w-8 h-8 bg-gradient-to-br from-accent-blue to-accent-purple rounded-lg flex items-center justify-center">
              <BookOpen className="w-5 h-5 text-white" />
            </div>
            <span className="font-bold text-lg gradient-text group-hover:scale-105 transition-transform">
              Neural Networks
            </span>
          </Link>

          {/* Navigation Items */}
          <div className="space-y-2">
            {navigationItems.map((item, index) => {
              const Icon = item.icon
              const isActive = pathname === item.href
              
              return (
                <motion.div
                  key={item.name}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.3, delay: index * 0.05 }}
                >
                  <Link
                    href={item.href}
                    className={`block px-4 py-3 rounded-lg text-sm transition-all duration-200 ${
                      isActive 
                        ? 'text-accent-blue bg-accent-blue/10 border border-accent-blue/20' 
                        : 'text-text-secondary hover:text-accent-blue hover:bg-bg-tertiary border border-transparent'
                    }`}
                  >
                    <div className="flex items-start space-x-3">
                      <div className={`flex-shrink-0 mt-0.5 ${
                        isActive ? 'text-accent-blue' : 'text-text-muted'
                      }`}>
                        <Icon className="w-4 h-4" />
                      </div>
                      <div className="flex-1">
                        <div className={`font-medium ${isActive ? 'text-accent-blue' : 'text-text-primary'}`}>
                          {item.name}
                        </div>
                        <div className="text-xs text-text-muted mt-1 leading-relaxed">
                          {item.description}
                        </div>
                      </div>
                    </div>
                  </Link>
                </motion.div>
              )
            })}
          </div>

          {/* Author and Contact */}
          <div className="mt-8 pt-6 border-t border-bg-tertiary">
            <div className="text-center">
              <div className="text-sm font-medium text-text-primary mb-3">Aaron Hennessy</div>
              <div className="flex justify-center space-x-4">
                <a 
                  href="https://github.com/Aaronheno/Equity-Premium-Prediction-NN" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="flex items-center space-x-1 text-text-muted hover:text-accent-blue transition-colors"
                >
                  <Github className="w-4 h-4" />
                  <span className="text-xs">GitHub</span>
                </a>
                <a 
                  href="mailto:aaronheno@gmail.com" 
                  className="flex items-center space-x-1 text-text-muted hover:text-accent-green transition-colors"
                >
                  <Mail className="w-4 h-4" />
                  <span className="text-xs">Contact</span>
                </a>
              </div>
            </div>
          </div>
        </div>
      </motion.nav>
    </>
  )
}