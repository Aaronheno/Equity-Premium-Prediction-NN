'use client'

import { motion } from 'framer-motion'
import { useState } from 'react'
import { Play, Target, BarChart3, TrendingUp, Database, Brain, Zap, CheckCircle } from 'lucide-react'
import CodeBlock from '@/components/shared/CodeBlock'
import MathFormula from '@/components/shared/MathFormula'
import NavigationButtons from '@/components/shared/NavigationButtons'

const predictionSteps = [
  {
    step: 1,
    title: 'Data Preparation',
    description: 'Prepare and normalize input financial indicators',
    process: 'Load → Validate → Scale → Tensor Conversion',
    icon: Database,
    color: 'accent-blue',
    details: 'Ensure consistent preprocessing pipeline from training'
  },
  {
    step: 2,
    title: 'Model Inference',
    description: 'Forward pass through trained neural network',
    process: 'Input → Hidden Layers → Output Prediction',
    icon: Brain,
    color: 'accent-orange',
    details: 'Model in evaluation mode with no gradient computation'
  },
  {
    step: 3,
    title: 'Output Processing',
    description: 'Transform raw predictions to interpretable values',
    process: 'Inverse Scale → Clipping → Confidence Intervals',
    icon: Target,
    color: 'accent-green',
    details: 'Convert normalized outputs back to equity premium scale'
  },
  {
    step: 4,
    title: 'Financial Interpretation',
    description: 'Translate predictions into investment insights',
    process: 'Risk Assessment → Portfolio Signals → Economic Context',
    icon: TrendingUp,
    color: 'accent-purple',
    details: 'Actionable investment recommendations from model outputs'
  }
]

const predictionModes = [
  {
    mode: 'Single Prediction',
    description: 'Generate prediction for current market conditions',
    useCase: 'Real-time trading decisions',
    inputFormat: 'Single sample [1, 30]',
    outputFormat: 'Scalar equity premium',
    example: 'Current month prediction: 8.3%'
  },
  {
    mode: 'Batch Predictions', 
    description: 'Process multiple time periods simultaneously',
    useCase: 'Historical analysis and backtesting',
    inputFormat: 'Batch samples [N, 30]',
    outputFormat: 'Vector of predictions [N, 1]',
    example: '12-month forecast: [7.8%, 8.1%, 8.5%, ...]'
  },
  {
    mode: 'Rolling Predictions',
    description: 'Sequential predictions with expanding window',
    useCase: 'Out-of-sample evaluation',
    inputFormat: 'Time series with sliding window',
    outputFormat: 'Time series predictions',
    example: 'Walk-forward validation over 5 years'
  },
  {
    mode: 'Ensemble Predictions',
    description: 'Combine predictions from multiple models',
    useCase: 'Robust forecasting with uncertainty',
    inputFormat: 'Multiple model outputs',
    outputFormat: 'Weighted average + confidence bands',
    example: 'Mean: 8.2%, 95% CI: [6.8%, 9.6%]'
  }
]

const interpretationLevels = [
  {
    level: 'Raw Prediction',
    value: '0.082',
    interpretation: '8.2% annualized equity premium',
    confidence: 'Point estimate from neural network',
    actionability: 'Low - needs context and uncertainty'
  },
  {
    level: 'Historical Context',
    value: '0.082 (75th percentile)',
    interpretation: 'Above-average expected returns',
    confidence: 'Relative to historical distribution',
    actionability: 'Medium - shows relative attractiveness'
  },
  {
    level: 'Risk-Adjusted Signal',
    value: '0.082 ± 0.024 (Sharpe: 1.4)',
    interpretation: 'Strong risk-adjusted opportunity',
    confidence: 'Includes prediction uncertainty',
    actionability: 'High - actionable investment signal'
  },
  {
    level: 'Portfolio Allocation',
    value: '65% equity weight (vs 50% baseline)',
    interpretation: 'Tactical overweight to equities',
    confidence: 'Optimal allocation given forecast',
    actionability: 'Very High - direct portfolio action'
  }
]

export default function PredictionsContent() {
  const [activeStep, setActiveStep] = useState(1)
  const [selectedMode, setSelectedMode] = useState(0)

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
            <span className="gradient-text">Making Predictions</span>
          </h1>
          <p className="max-w-4xl mx-auto text-xl text-text-secondary leading-relaxed">
            Model inference and output processing: transforming trained neural networks into 
            actionable investment insights through proper prediction methodology.
          </p>
        </motion.div>

        {/* Prediction Overview */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Play className="w-8 h-8 text-accent-blue mr-3" />
            From Model to Investment Decision
          </h2>
          
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8 mb-8">
            <p className="text-text-secondary text-lg mb-6 leading-relaxed">
              Making predictions with trained neural networks involves more than a simple forward pass. 
              Proper inference requires careful data preprocessing, model evaluation mode, output 
              scaling, and financial interpretation to generate actionable investment insights.
            </p>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div>
                <h3 className="text-xl font-semibold text-text-primary mb-4">Inference vs Training:</h3>
                <div className="space-y-3 text-sm">
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-blue mt-2"></div>
                    <span className="text-text-secondary"><strong>No gradient computation:</strong> Use torch.no_grad() for efficiency</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-blue mt-2"></div>
                    <span className="text-text-secondary"><strong>Evaluation mode:</strong> Disable dropout, use population statistics</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-blue mt-2"></div>
                    <span className="text-text-secondary"><strong>Consistent preprocessing:</strong> Apply same scaling as training</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-blue mt-2"></div>
                    <span className="text-text-secondary"><strong>Output interpretation:</strong> Transform to financial meaningful scale</span>
                  </div>
                </div>
              </div>
              
              <div>
                <h3 className="text-xl font-semibold text-text-primary mb-4">Financial Considerations:</h3>
                <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 space-y-3 text-sm">
                  <div className="text-accent-blue font-medium">Temporal Consistency</div>
                  <div className="text-text-secondary">• Use only information available at prediction time</div>
                  <div className="text-text-secondary">• Account for data release lags</div>
                  <div className="text-text-secondary">• Consider market regime changes</div>
                  <div className="text-text-secondary">• Validate prediction stability over time</div>
                </div>
              </div>
            </div>
          </div>
        </motion.section>

        {/* Four-Step Process */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Zap className="w-8 h-8 text-accent-orange mr-3" />
            Four-Step Prediction Process
          </h2>
          
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8 mb-8">
            <p className="text-text-secondary text-lg mb-8">
              Each prediction follows a systematic four-step process to ensure consistency, 
              accuracy, and financial interpretability. Click each step to explore the 
              detailed implementation.
            </p>

            {/* Step Selector */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
              {predictionSteps.map((step) => {
                const Icon = step.icon
                const isActive = activeStep === step.step
                return (
                  <button
                    key={step.step}
                    onClick={() => setActiveStep(step.step)}
                    className={`p-4 rounded-lg border transition-all text-left ${
                      isActive 
                        ? `bg-${step.color}/10 border-${step.color}/30` 
                        : 'bg-bg-primary border-bg-tertiary hover:border-accent-blue/30'
                    }`}
                  >
                    <div className="flex items-center space-x-3 mb-3">
                      <div className={`inline-flex items-center justify-center w-8 h-8 rounded-lg bg-${step.color}/10 border border-${step.color}/20`}>
                        <Icon className={`w-4 h-4 text-${step.color}`} />
                      </div>
                      <div>
                        <div className="text-xs text-text-muted">Step {step.step}</div>
                        <div className="font-semibold text-text-primary text-sm">{step.title}</div>
                      </div>
                    </div>
                    <div className="text-xs text-accent-blue">{step.process}</div>
                  </button>
                )
              })}
            </div>

            {/* Active Step Details */}
            {predictionSteps.map((step) => {
              const Icon = step.icon
              if (activeStep !== step.step) return null
              
              return (
                <motion.div
                  key={step.step}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                  className={`bg-${step.color}/5 border border-${step.color}/20 rounded-xl p-8`}
                >
                  <div className="flex items-center space-x-3 mb-6">
                    <div className={`inline-flex items-center justify-center w-12 h-12 rounded-lg bg-${step.color}/10 border border-${step.color}/20`}>
                      <Icon className={`w-6 h-6 text-${step.color}`} />
                    </div>
                    <div>
                      <h3 className="text-2xl font-bold text-text-primary">{step.title}</h3>
                      <p className="text-text-secondary">{step.description}</p>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    <div>
                      <h4 className="font-semibold text-text-primary mb-3">Process Flow:</h4>
                      <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 mb-4">
                        <MathFormula latex={step.process} />
                      </div>
                      
                      <div className={`bg-${step.color}/10 border border-${step.color}/20 rounded-lg p-4`}>
                        <div className={`text-${step.color} text-sm font-medium mb-1`}>Key Consideration:</div>
                        <div className="text-text-secondary text-sm">{step.details}</div>
                      </div>
                    </div>
                    
                    <div>
                      {step.step === 1 && (
                        <div>
                          <h4 className="font-semibold text-text-primary mb-3">Data Validation Checklist:</h4>
                          <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 space-y-2 text-sm">
                            <div>✓ All 30 financial indicators present</div>
                            <div>✓ No missing or NaN values</div>
                            <div>✓ Values within expected ranges</div>
                            <div>✓ Same scaling as training data</div>
                            <div>✓ Temporal alignment verified</div>
                          </div>
                        </div>
                      )}
                      
                      {step.step === 2 && (
                        <div>
                          <h4 className="font-semibold text-text-primary mb-3">Inference Settings:</h4>
                          <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 space-y-2 text-sm font-mono">
                            <div>model.eval()  <span className="text-accent-green"># Evaluation mode</span></div>
                            <div>torch.no_grad():  <span className="text-accent-green"># No gradients</span></div>
                            <div>  prediction = model(input_tensor)</div>
                            <div className="text-accent-orange">Dropout: OFF, BatchNorm: eval</div>
                          </div>
                        </div>
                      )}
                      
                      {step.step === 3 && (
                        <div>
                          <h4 className="font-semibold text-text-primary mb-3">Output Transformation:</h4>
                          <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 space-y-2 text-sm">
                            <div><strong>Inverse scaling:</strong> scaler.inverse_transform()</div>
                            <div><strong>Range clipping:</strong> [0%, 20%] reasonable bounds</div>
                            <div><strong>Uncertainty:</strong> Multiple model ensemble</div>
                            <div><strong>Format:</strong> Percentage with confidence intervals</div>
                          </div>
                        </div>
                      )}
                      
                      {step.step === 4 && (
                        <div>
                          <h4 className="font-semibold text-text-primary mb-3">Investment Translation:</h4>
                          <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 space-y-2 text-sm">
                            <div><strong>Signal strength:</strong> Distance from historical mean</div>
                            <div><strong>Risk adjustment:</strong> Scale by prediction uncertainty</div>
                            <div><strong>Portfolio impact:</strong> Translate to asset allocation</div>
                            <div><strong>Timing:</strong> Consider implementation costs</div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </motion.div>
              )
            })}
          </div>
        </motion.section>

        {/* Prediction Modes */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <BarChart3 className="w-8 h-8 text-accent-green mr-3" />
            Prediction Modes
          </h2>
          
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8 mb-8">
            <p className="text-text-secondary text-lg mb-8">
              Different prediction scenarios require different approaches. From real-time trading 
              decisions to comprehensive backtesting, each mode has specific requirements and outputs.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
              {predictionModes.map((mode, index) => (
                <button
                  key={mode.mode}
                  onClick={() => setSelectedMode(index)}
                  className={`p-6 rounded-lg border transition-all text-left ${
                    selectedMode === index
                      ? 'bg-accent-green/10 border-accent-green/30'
                      : 'bg-bg-primary border-bg-tertiary hover:border-accent-blue/30'
                  }`}
                >
                  <h3 className="font-semibold text-text-primary mb-2">{mode.mode}</h3>
                  <p className="text-text-secondary text-sm mb-3">{mode.description}</p>
                  <div className="text-accent-green text-xs font-medium">{mode.useCase}</div>
                </button>
              ))}
            </div>

            {/* Selected Mode Details */}
            <motion.div
              key={selectedMode}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              className="bg-accent-green/5 border border-accent-green/20 rounded-xl p-8"
            >
              <h3 className="text-2xl font-bold text-text-primary mb-6">{predictionModes[selectedMode].mode}</h3>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div>
                  <h4 className="font-semibold text-text-primary mb-4">Technical Specifications:</h4>
                  <div className="space-y-3 text-sm">
                    <div className="flex justify-between">
                      <span className="text-text-muted">Input Format:</span>
                      <span className="font-mono text-accent-green">{predictionModes[selectedMode].inputFormat}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-text-muted">Output Format:</span>
                      <span className="font-mono text-accent-green">{predictionModes[selectedMode].outputFormat}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-text-muted">Primary Use:</span>
                      <span className="text-text-secondary">{predictionModes[selectedMode].useCase}</span>
                    </div>
                  </div>
                </div>
                
                <div>
                  <h4 className="font-semibold text-text-primary mb-4">Example Output:</h4>
                  <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4">
                    <div className="font-mono text-accent-blue text-sm">
                      {predictionModes[selectedMode].example}
                    </div>
                  </div>
                  
                  <div className="mt-4 bg-accent-green/10 border border-accent-green/20 rounded-lg p-4">
                    <h5 className="font-semibold text-accent-green mb-2">Implementation Notes:</h5>
                    <div className="text-text-secondary text-sm space-y-1">
                      {selectedMode === 0 && (
                        <>
                          <div>• Minimal latency for real-time decisions</div>
                          <div>• Single forward pass through model</div>
                          <div>• Immediate output transformation</div>
                        </>
                      )}
                      {selectedMode === 1 && (
                        <>
                          <div>• Vectorized computation for efficiency</div>
                          <div>• Consistent preprocessing across samples</div>
                          <div>• Batch-wise confidence intervals</div>
                        </>
                      )}
                      {selectedMode === 2 && (
                        <>
                          <div>• Sequential prediction with temporal validation</div>
                          <div>• Expanding or rolling window approach</div>
                          <div>• Out-of-sample performance assessment</div>
                        </>
                      )}
                      {selectedMode === 3 && (
                        <>
                          <div>• Combine multiple model predictions</div>
                          <div>• Weight models by validation performance</div>
                          <div>• Quantify prediction uncertainty</div>
                        </>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        </motion.section>

        {/* Interpretation Levels */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.5 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Target className="w-8 h-8 text-accent-purple mr-3" />
            From Raw Output to Investment Action
          </h2>
          
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8 mb-8">
            <p className="text-text-secondary text-lg mb-8">
              Raw neural network outputs require progressive interpretation to become actionable 
              investment decisions. Each level adds context and reduces the gap between model 
              predictions and portfolio management.
            </p>

            <div className="space-y-4">
              {interpretationLevels.map((level, index) => (
                <motion.div
                  key={level.level}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.6, delay: 0.6 + index * 0.1 }}
                  className="bg-bg-primary border border-bg-tertiary rounded-lg p-6"
                >
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <h3 className="font-semibold text-text-primary text-lg">{level.level}</h3>
                      <div className="font-mono text-accent-blue text-lg">{level.value}</div>
                    </div>
                    <div className={`px-3 py-1 rounded-full text-xs font-medium ${
                      level.actionability === 'Very High' ? 'bg-accent-green/20 text-accent-green' :
                      level.actionability === 'High' ? 'bg-accent-blue/20 text-accent-blue' :
                      level.actionability === 'Medium' ? 'bg-accent-orange/20 text-accent-orange' :
                      'bg-accent-red/20 text-accent-red'
                    }`}>
                      {level.actionability} Actionability
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <h4 className="font-semibold text-text-primary mb-2">Interpretation:</h4>
                      <p className="text-text-secondary text-sm">{level.interpretation}</p>
                    </div>
                    <div>
                      <h4 className="font-semibold text-text-primary mb-2">Confidence Basis:</h4>
                      <p className="text-text-secondary text-sm">{level.confidence}</p>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>

            <div className="mt-8 bg-accent-purple/5 border border-accent-purple/20 rounded-lg p-6">
              <h3 className="text-xl font-semibold text-accent-purple mb-4">Best Practices for Financial Interpretation</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm">
                <div>
                  <h4 className="font-semibold text-text-primary mb-2">Context is Critical:</h4>
                  <div className="space-y-1 text-text-secondary">
                    <div>• Compare to historical distribution</div>
                    <div>• Consider current market regime</div>
                    <div>• Account for prediction uncertainty</div>
                    <div>• Validate against fundamental indicators</div>
                  </div>
                </div>
                <div>
                  <h4 className="font-semibold text-text-primary mb-2">Implementation Guidelines:</h4>
                  <div className="space-y-1 text-text-secondary">
                    <div>• Never act on single prediction</div>
                    <div>• Use ensemble methods for robustness</div>
                    <div>• Include transaction cost considerations</div>
                    <div>• Monitor prediction stability over time</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </motion.section>

        {/* Implementation Example */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <CheckCircle className="w-8 h-8 text-accent-blue mr-3" />
            Prediction Implementation
          </h2>
          
          <p className="text-text-secondary text-lg mb-6">
            Complete implementation covering all prediction modes, from single real-time 
            predictions to comprehensive ensemble forecasting with uncertainty quantification.
          </p>

          <CodeBlock
            language="python"
            title="Comprehensive Prediction System for Financial Neural Networks"
            code={`import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Union
import warnings
from datetime import datetime, timedelta

class FinancialPredictor:
    """
    Comprehensive prediction system for financial neural networks.
    Handles all prediction modes with proper preprocessing and interpretation.
    """
    
    def __init__(self, model: nn.Module, scaler_X: StandardScaler, 
                 scaler_y: StandardScaler, feature_names: List[str]):
        self.model = model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.feature_names = feature_names
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Prediction bounds (reasonable equity premium range)
        self.min_prediction = 0.0  # 0%
        self.max_prediction = 0.20  # 20%
        
        # Historical statistics for context
        self.historical_mean = 0.08  # 8% historical average
        self.historical_std = 0.16   # 16% historical volatility
    
    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """
        Validate and preprocess input data.
        """
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        elif isinstance(X, list):
            X = np.array(X)
        
        # Check dimensions
        if X.ndim == 1:
            X = X.reshape(1, -1)  # Single sample
        
        if X.shape[1] != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} features, got {X.shape[1]}")
        
        # Check for missing values
        if np.any(np.isnan(X)):
            raise ValueError("Input contains NaN values")
        
        # Check for reasonable ranges (basic sanity check)
        if np.any(np.abs(X) > 10):  # After standardization, should be within ~3 std devs
            warnings.warn("Input values seem extreme - check preprocessing")
        
        return X
    
    def _preprocess_input(self, X: np.ndarray) -> torch.Tensor:
        """
        Apply the same preprocessing as training.
        """
        # Standardize using training scalers
        X_scaled = self.scaler_X.transform(X)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled)
        
        return X_tensor
    
    def _postprocess_output(self, predictions: torch.Tensor) -> np.ndarray:
        """
        Transform model output back to financial meaningful scale.
        """
        # Convert to numpy
        predictions_np = predictions.detach().numpy()
        
        # Inverse scale
        predictions_original = self.scaler_y.inverse_transform(predictions_np)
        
        # Clip to reasonable bounds
        predictions_clipped = np.clip(predictions_original, 
                                    self.min_prediction, self.max_prediction)
        
        return predictions_clipped.flatten()
    
    def predict_single(self, X: np.ndarray, return_raw: bool = False) -> Dict[str, float]:
        """
        Make single prediction for real-time trading decisions.
        
        Args:
            X: Single sample of financial indicators [30,]
            return_raw: Whether to return raw (unscaled) model output
            
        Returns:
            Dictionary with prediction and metadata
        """
        # Validate and preprocess
        X_validated = self._validate_input(X)
        X_tensor = self._preprocess_input(X_validated)
        
        # Make prediction
        with torch.no_grad():
            raw_prediction = self.model(X_tensor)
        
        # Process output
        if return_raw:
            prediction = raw_prediction.item()
        else:
            prediction = self._postprocess_output(raw_prediction)[0]
        
        # Add financial interpretation
        result = {
            'prediction': prediction,
            'prediction_pct': prediction * 100 if not return_raw else None,
            'historical_percentile': self._get_percentile(prediction) if not return_raw else None,
            'signal_strength': self._get_signal_strength(prediction) if not return_raw else None,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def predict_batch(self, X: np.ndarray, return_confidence: bool = True) -> Dict[str, np.ndarray]:
        """
        Make batch predictions for multiple samples.
        
        Args:
            X: Multiple samples [N, 30]
            return_confidence: Whether to estimate confidence intervals
            
        Returns:
            Dictionary with predictions and optional confidence intervals
        """
        # Validate and preprocess
        X_validated = self._validate_input(X)
        X_tensor = self._preprocess_input(X_validated)
        
        # Make predictions
        with torch.no_grad():
            raw_predictions = self.model(X_tensor)
        
        # Process outputs
        predictions = self._postprocess_output(raw_predictions)
        
        result = {
            'predictions': predictions,
            'predictions_pct': predictions * 100,
            'mean_prediction': np.mean(predictions),
            'std_prediction': np.std(predictions)
        }
        
        # Add confidence intervals if requested
        if return_confidence:
            # Simple approach: use historical volatility as uncertainty estimate
            prediction_std = self.historical_std / np.sqrt(len(predictions))  # Standard error
            
            result['confidence_lower'] = predictions - 1.96 * prediction_std
            result['confidence_upper'] = predictions + 1.96 * prediction_std
            result['confidence_level'] = 0.95
        
        return result
    
    def predict_rolling(self, X: np.ndarray, window_size: int = 60, 
                       step_size: int = 1) -> pd.DataFrame:
        """
        Make rolling predictions for out-of-sample evaluation.
        
        Args:
            X: Time series data [T, 30]
            window_size: Size of rolling window for predictions
            step_size: Step size between predictions
            
        Returns:
            DataFrame with timestamps, predictions, and metadata
        """
        X_validated = self._validate_input(X)
        n_samples = len(X_validated)
        
        if n_samples < window_size:
            raise ValueError(f"Need at least {window_size} samples for rolling prediction")
        
        results = []
        
        # Rolling prediction loop
        for i in range(window_size, n_samples, step_size):
            # Use current sample for prediction
            current_X = X_validated[i:i+1]  # Single sample
            
            # Make prediction
            prediction_result = self.predict_single(current_X)
            
            # Store with metadata
            results.append({
                'timestamp': i,
                'prediction': prediction_result['prediction'],
                'prediction_pct': prediction_result['prediction_pct'],
                'historical_percentile': prediction_result['historical_percentile'],
                'signal_strength': prediction_result['signal_strength']
            })
        
        return pd.DataFrame(results)
    
    def predict_ensemble(self, models: List[nn.Module], X: np.ndarray, 
                        weights: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Make ensemble prediction using multiple models.
        
        Args:
            models: List of trained models
            X: Input sample [30,]
            weights: Model weights (if None, use equal weights)
            
        Returns:
            Dictionary with ensemble prediction and uncertainty
        """
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        if len(weights) != len(models):
            raise ValueError("Number of weights must match number of models")
        
        # Validate input
        X_validated = self._validate_input(X)
        X_tensor = self._preprocess_input(X_validated)
        
        # Collect predictions from all models
        predictions = []
        for model in models:
            model.eval()
            with torch.no_grad():
                pred = model(X_tensor)
                predictions.append(self._postprocess_output(pred)[0])
        
        predictions = np.array(predictions)
        weights = np.array(weights)
        
        # Weighted ensemble prediction
        ensemble_pred = np.sum(predictions * weights)
        
        # Uncertainty measures
        prediction_std = np.std(predictions)
        prediction_range = np.max(predictions) - np.min(predictions)
        
        result = {
            'ensemble_prediction': ensemble_pred,
            'ensemble_prediction_pct': ensemble_pred * 100,
            'individual_predictions': predictions.tolist(),
            'prediction_std': prediction_std,
            'prediction_range': prediction_range,
            'confidence_lower': ensemble_pred - 1.96 * prediction_std,
            'confidence_upper': ensemble_pred + 1.96 * prediction_std,
            'signal_strength': self._get_signal_strength(ensemble_pred),
            'model_agreement': 1.0 - (prediction_std / np.mean(predictions))  # Agreement score
        }
        
        return result
    
    def _get_percentile(self, prediction: float) -> float:
        """
        Get historical percentile of prediction.
        """
        # Simplified: assume normal distribution around historical mean
        z_score = (prediction - self.historical_mean) / self.historical_std
        # Convert to percentile (approximation)
        percentile = 50 + 34.13 * z_score if abs(z_score) <= 1 else (
            50 + 47.72 * np.sign(z_score) if abs(z_score) <= 2 else
            50 + 49.87 * np.sign(z_score)
        )
        return np.clip(percentile, 0, 100)
    
    def _get_signal_strength(self, prediction: float) -> str:
        """
        Categorize prediction strength relative to historical mean.
        """
        deviation = abs(prediction - self.historical_mean)
        
        if deviation < 0.01:  # Within 1%
            return "Neutral"
        elif deviation < 0.02:  # 1-2%
            return "Weak"
        elif deviation < 0.04:  # 2-4%
            return "Moderate"
        else:  # >4%
            return "Strong"
    
    def generate_investment_signal(self, prediction_result: Dict[str, float], 
                                 current_allocation: float = 0.6) -> Dict[str, Union[float, str]]:
        """
        Translate prediction into actionable investment signal.
        
        Args:
            prediction_result: Output from predict_single or predict_ensemble
            current_allocation: Current equity allocation (0-1)
            
        Returns:
            Investment recommendation dictionary
        """
        prediction = prediction_result['prediction']
        signal_strength = prediction_result.get('signal_strength', 'Unknown')
        
        # Calculate optimal allocation using simple mean-variance approach
        # Simplified: allocation = (expected_return - risk_free) / (risk_aversion * variance)
        risk_free_rate = 0.02  # 2% risk-free rate
        risk_aversion = 3.0    # Moderate risk aversion
        equity_variance = self.historical_std ** 2
        
        excess_return = prediction - risk_free_rate
        optimal_allocation = excess_return / (risk_aversion * equity_variance)
        optimal_allocation = np.clip(optimal_allocation, 0.2, 1.0)  # 20-100% bounds
        
        # Calculate allocation change
        allocation_change = optimal_allocation - current_allocation
        
        # Generate recommendation
        if abs(allocation_change) < 0.05:  # Less than 5%
            action = "Hold"
            urgency = "Low"
        elif allocation_change > 0.1:  # Increase by more than 10%
            action = "Increase Equity"
            urgency = "High" if signal_strength in ["Strong", "Moderate"] else "Medium"
        elif allocation_change < -0.1:  # Decrease by more than 10%
            action = "Decrease Equity"
            urgency = "High" if signal_strength in ["Strong", "Moderate"] else "Medium"
        else:
            action = "Adjust" if allocation_change > 0 else "Reduce"
            urgency = "Medium"
        
        return {
            'action': action,
            'optimal_allocation': optimal_allocation,
            'current_allocation': current_allocation,
            'allocation_change': allocation_change,
            'urgency': urgency,
            'rationale': f"Predicted equity premium: {prediction:.1%}, Signal: {signal_strength}",
            'risk_adjusted_score': excess_return / self.historical_std  # Sharpe-like ratio
        }

# Example usage and testing
def demonstrate_prediction_system():
    """
    Demonstrate all prediction capabilities with example data.
    """
    from src.models.nns import Net3
    
    # Create example model (normally loaded from saved state)
    model = Net3(n_feature=30, n_hidden1=64, n_hidden2=32, n_hidden3=16, n_output=1)
    
    # Create example scalers (normally saved from training)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # Fit with dummy data for demonstration
    dummy_X = np.random.randn(1000, 30)
    dummy_y = np.random.randn(1000, 1) * 0.02 + 0.08
    scaler_X.fit(dummy_X)
    scaler_y.fit(dummy_y)
    
    feature_names = [f"indicator_{i+1}" for i in range(30)]
    
    # Create predictor
    predictor = FinancialPredictor(model, scaler_X, scaler_y, feature_names)
    
    # Example 1: Single prediction
    print("=== Single Prediction ===")
    current_data = np.random.randn(30)  # Current market indicators
    single_result = predictor.predict_single(current_data)
    print(f"Prediction: {single_result['prediction']:.1%}")
    print(f"Signal Strength: {single_result['signal_strength']}")
    print(f"Historical Percentile: {single_result['historical_percentile']:.1f}%")
    
    # Example 2: Batch predictions
    print("\\n=== Batch Predictions ===")
    batch_data = np.random.randn(12, 30)  # 12 months of data
    batch_result = predictor.predict_batch(batch_data)
    print(f"Mean Prediction: {batch_result['mean_prediction']:.1%}")
    print(f"Prediction Range: {batch_result['predictions_pct'].min():.1f}% - {batch_result['predictions_pct'].max():.1f}%")
    
    # Example 3: Investment signal
    print("\\n=== Investment Signal ===")
    signal = predictor.generate_investment_signal(single_result, current_allocation=0.6)
    print(f"Recommendation: {signal['action']}")
    print(f"Optimal Allocation: {signal['optimal_allocation']:.1%}")
    print(f"Urgency: {signal['urgency']}")
    print(f"Rationale: {signal['rationale']}")
    
    # Example 4: Ensemble prediction (with multiple models)
    print("\\n=== Ensemble Prediction ===")
    # Create additional models for ensemble
    model2 = Net3(n_feature=30, n_hidden1=48, n_hidden2=24, n_hidden3=12, n_output=1)
    model3 = Net3(n_feature=30, n_hidden1=80, n_hidden2=40, n_hidden3=20, n_output=1)
    
    ensemble_result = predictor.predict_ensemble([model, model2, model3], current_data)
    print(f"Ensemble Prediction: {ensemble_result['ensemble_prediction']:.1%}")
    print(f"Model Agreement: {ensemble_result['model_agreement']:.1%}")
    print(f"Confidence Interval: [{ensemble_result['confidence_lower']:.1%}, {ensemble_result['confidence_upper']:.1%}]")

if __name__ == "__main__":
    demonstrate_prediction_system()`}
          />
        </motion.section>

        {/* Navigation */}
        <NavigationButtons 
          prevHref="/hyperparameter-optimization"
          prevLabel="Hyperparameter Optimization"
          nextHref="/evaluation"
          nextLabel="Evaluation"
        />
      </div>
    </div>
  )
}