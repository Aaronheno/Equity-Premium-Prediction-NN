'use client'

import { motion } from 'framer-motion'
import { useState } from 'react'
import { Settings, TrendingUp, Calendar, Shield, Database, BarChart3, Target, AlertTriangle, Code2, Play } from 'lucide-react'
import CodeBlock from '@/components/shared/CodeBlock'
import MathFormula from '@/components/shared/MathFormula'
import NavigationButtons from '@/components/shared/NavigationButtons'

const preprocessingSteps = [
  {
    title: 'Feature Standardization',
    icon: Settings,
    color: 'accent-blue',
    description: 'StandardScaler normalization to center features at zero with unit variance',
    purpose: 'Ensures all financial indicators contribute equally to neural network training'
  },
  {
    title: 'Temporal Validation',
    icon: Calendar,
    color: 'accent-green',
    description: 'Time-aware train/validation splits that respect temporal ordering',
    purpose: 'Prevents look-ahead bias and maintains realistic trading conditions'
  },
  {
    title: 'PyTorch Tensor Conversion',
    icon: Database,
    color: 'accent-purple',
    description: 'Convert pandas DataFrames to PyTorch tensors for neural network processing',
    purpose: 'Optimizes data format for GPU acceleration and automatic differentiation'
  },
  {
    title: 'Batch Preparation',
    icon: BarChart3,
    color: 'accent-orange',
    description: 'Organize data into batches for efficient mini-batch gradient descent',
    purpose: 'Enables parallel processing and stable gradient computation'
  }
]

const temporalConstraints = [
  {
    constraint: 'No Look-Ahead Bias',
    description: 'Training data at time t cannot include information from time t+1 or later',
    example: 'To predict January 2020 returns, only use data through December 2019',
    risk: 'Using future data creates artificially high performance that cannot be replicated in real trading'
  },
  {
    constraint: 'Expanding Window Training',
    description: 'Training set grows over time, never discarding historical data',
    example: 'For 2020 predictions: train on 1926-2019, for 2021: train on 1926-2020',
    risk: 'Fixed windows may discard valuable long-term relationships'
  },
  {
    constraint: 'Monthly Frequency Alignment',
    description: 'All predictors and targets must be aligned to month-end observations',
    example: 'December 2019 predictors → January 2020 equity premium target',
    risk: 'Misaligned timing can introduce spurious predictive relationships'
  },
  {
    constraint: 'Consistent Feature Availability',
    description: 'All 32 features must be available at each prediction time',
    example: 'Cannot use features that become available mid-month or with delays',
    risk: 'Inconsistent feature availability creates unrealistic backtesting scenarios'
  }
]

export default function PreprocessingContent() {
  const [activeConstraint, setActiveConstraint] = useState(0)

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
            <span className="gradient-text">Data Preprocessing</span>
          </h1>
          <p className="max-w-4xl mx-auto text-xl text-text-secondary leading-relaxed">
            Critical preprocessing steps that transform raw financial data into neural network-ready format
            while maintaining temporal integrity and preventing look-ahead bias.
          </p>
        </motion.div>

        {/* Why Preprocessing Matters */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Target className="w-8 h-8 text-accent-blue mr-3" />
            Why Preprocessing is Critical
          </h2>
          
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8 mb-8">
            <p className="text-text-secondary text-lg mb-6 leading-relaxed">
              Neural networks are sensitive to input data characteristics. Financial data presents unique challenges that require 
              careful preprocessing to achieve reliable predictions.
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <h3 className="text-xl font-semibold text-text-primary">Raw Data Issues:</h3>
                <div className="space-y-3">
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-red mt-2"></div>
                    <span className="text-text-secondary"><strong>Scale differences:</strong> Interest rates (0-20%) vs. ratios (0.01-5.0)</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-red mt-2"></div>
                    <span className="text-text-secondary"><strong>Temporal ordering:</strong> Time series require careful train/test splits</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-red mt-2"></div>
                    <span className="text-text-secondary"><strong>Look-ahead bias:</strong> Easy to accidentally use future information</span>
                  </div>
                </div>
              </div>
              <div className="space-y-4">
                <h3 className="text-xl font-semibold text-text-primary">After Preprocessing:</h3>
                <div className="space-y-3">
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-green mt-2"></div>
                    <span className="text-text-secondary"><strong>Normalized scales:</strong> All features centered at 0 with unit variance</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-green mt-2"></div>
                    <span className="text-text-secondary"><strong>Temporal integrity:</strong> Strict chronological data ordering</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-green mt-2"></div>
                    <span className="text-text-secondary"><strong>Realistic constraints:</strong> Only past information used for predictions</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </motion.section>

        {/* Preprocessing Steps Overview */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Settings className="w-8 h-8 text-accent-purple mr-3" />
            Preprocessing Pipeline Overview
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {preprocessingSteps.map((step, index) => {
              const Icon = step.icon
              return (
                <motion.div
                  key={step.title}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: 0.4 + index * 0.1 }}
                  className="bg-bg-secondary border border-bg-tertiary rounded-xl p-6"
                >
                  <div className="flex items-center space-x-3 mb-4">
                    <div className={`inline-flex items-center justify-center w-12 h-12 rounded-lg bg-${step.color}/10 border border-${step.color}/20`}>
                      <Icon className={`w-6 h-6 text-${step.color}`} />
                    </div>
                    <div>
                      <h3 className="text-xl font-bold text-text-primary">{step.title}</h3>
                      <div className="text-xs text-text-muted">Step {index + 1}</div>
                    </div>
                  </div>
                  <p className="text-text-secondary mb-3">{step.description}</p>
                  <div className={`bg-${step.color}/5 border border-${step.color}/20 rounded-lg p-3`}>
                    <div className={`text-${step.color} text-sm font-medium mb-1`}>Purpose:</div>
                    <div className="text-text-secondary text-sm">{step.purpose}</div>
                  </div>
                </motion.div>
              )
            })}
          </div>
        </motion.section>

        {/* Feature Standardization */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <TrendingUp className="w-8 h-8 text-accent-blue mr-3" />
            Step 1: Feature Standardization with StandardScaler
          </h2>
          
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8 mb-8">
            <p className="text-text-secondary text-lg mb-6">
              StandardScaler transforms each feature to have zero mean and unit variance. This ensures all financial indicators 
              contribute equally to neural network training, regardless of their natural scales.
            </p>

            <h3 className="text-xl font-semibold text-text-primary mb-4">Mathematical Foundation</h3>
            <div className="text-center mb-6">
              <MathFormula 
                latex={String.raw`X_{\text{std}} = \frac{X - \mu}{\sigma}`}
                block={true}
                explanation="Where μ is the sample mean and σ is the sample standard deviation"
              />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-6">
              <div>
                <h4 className="font-semibold text-text-primary mb-3">Before Standardization:</h4>
                <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-text-secondary">Dividend Yield:</span>
                    <span className="font-mono text-text-primary">0.01 to 0.08 (1% to 8%)</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-text-secondary">Term Spread:</span>
                    <span className="font-mono text-text-primary">-0.03 to 0.04 (-3% to 4%)</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-text-secondary">Book-to-Market:</span>
                    <span className="font-mono text-text-primary">0.2 to 1.8</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-text-secondary">Inflation:</span>
                    <span className="font-mono text-text-primary">-0.15 to 0.20 (-15% to 20%)</span>
                  </div>
                </div>
              </div>
              <div>
                <h4 className="font-semibold text-text-primary mb-3">After Standardization:</h4>
                <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-text-secondary">All features:</span>
                    <span className="font-mono text-text-primary">Mean = 0.0</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-text-secondary">All features:</span>
                    <span className="font-mono text-text-primary">Std = 1.0</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-text-secondary">Typical range:</span>
                    <span className="font-mono text-text-primary">-3.0 to +3.0</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-text-secondary">Units:</span>
                    <span className="font-mono text-text-primary">Standard deviations</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-accent-blue/5 border border-accent-blue/20 rounded-lg p-6">
              <h4 className="text-xl font-semibold text-accent-blue mb-3">Why This Matters for Neural Networks</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                <div className="space-y-2">
                  <div className="flex items-start space-x-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-accent-blue mt-1.5"></div>
                    <span className="text-text-secondary"><strong>Equal influence:</strong> No single feature dominates due to scale</span>
                  </div>
                  <div className="flex items-start space-x-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-accent-blue mt-1.5"></div>
                    <span className="text-text-secondary"><strong>Stable gradients:</strong> Prevents exploding/vanishing gradients</span>
                  </div>
                </div>
                <div className="space-y-2">
                  <div className="flex items-start space-x-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-accent-blue mt-1.5"></div>
                    <span className="text-text-secondary"><strong>Faster convergence:</strong> Optimization converges more quickly</span>
                  </div>
                  <div className="flex items-start space-x-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-accent-blue mt-1.5"></div>
                    <span className="text-text-secondary"><strong>Better initialization:</strong> Works well with default weight initialization</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <CodeBlock
            language="python"
            title="StandardScaler Implementation"
            code={`# StandardScaler implementation for financial data preprocessing

from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

def standardize_features(X_train, X_test=None):
    """
    Standardize features using training data statistics.
    
    Args:
        X_train: Training features (pandas DataFrame or numpy array)
        X_test: Test features (optional, same format as X_train)
    
    Returns:
        dict containing:
        - X_train_scaled: Standardized training features
        - X_test_scaled: Standardized test features (if provided)
        - scaler: Fitted StandardScaler object for future use
    """
    
    # Initialize and fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    result = {
        'X_train_scaled': X_train_scaled,
        'scaler': scaler,
        'feature_means': scaler.mean_,
        'feature_stds': scaler.scale_
    }
    
    # Transform test data using training statistics
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)  # Use fit from training data
        result['X_test_scaled'] = X_test_scaled
    
    return result

# Example usage with financial indicators
def example_standardization():
    """Demonstrate standardization with sample financial data."""
    
    # Sample financial data (normally loaded from Excel)
    financial_data = pd.DataFrame({
        'DP': [0.015, 0.025, 0.035, 0.020, 0.030],      # Dividend-Price ratio
        'DY': [0.018, 0.028, 0.038, 0.023, 0.033],      # Dividend Yield  
        'EP': [0.045, 0.065, 0.055, 0.050, 0.070],      # Earnings-Price ratio
        'TBL': [0.02, 0.03, 0.01, 0.025, 0.015],        # Treasury Bill rate
        'LTY': [0.05, 0.06, 0.04, 0.055, 0.045],        # Long-term Yield
        'INFL': [0.02, 0.03, 0.01, 0.025, 0.015]        # Inflation rate
    })
    
    print("Before standardization:")
    print(f"DP mean: {financial_data['DP'].mean():.4f}, std: {financial_data['DP'].std():.4f}")
    print(f"TBL mean: {financial_data['TBL'].mean():.4f}, std: {financial_data['TBL'].std():.4f}")
    
    # Apply standardization
    result = standardize_features(financial_data)
    
    # Convert back to DataFrame for display
    scaled_df = pd.DataFrame(
        result['X_train_scaled'], 
        columns=financial_data.columns
    )
    
    print("\\nAfter standardization:")
    print(f"DP mean: {scaled_df['DP'].mean():.4f}, std: {scaled_df['DP'].std():.4f}")
    print(f"TBL mean: {scaled_df['TBL'].mean():.4f}, std: {scaled_df['TBL'].std():.4f}")
    
    return scaled_df

# Critical: Always use training data statistics for test data
# Never fit scaler on test data - this would introduce look-ahead bias`}
          />
        </motion.section>

        {/* Temporal Constraints */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.5 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Calendar className="w-8 h-8 text-accent-green mr-3" />
            Step 2: Temporal Validation & Constraints
          </h2>
          
          <p className="text-text-secondary text-lg mb-8">
            Time series prediction requires strict temporal ordering to prevent look-ahead bias. 
            The system must only use information available at prediction time.
          </p>

          {/* Constraint Navigation */}
          <div className="flex flex-wrap gap-2 mb-8">
            {temporalConstraints.map((constraint, index) => (
              <button
                key={index}
                onClick={() => setActiveConstraint(index)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  activeConstraint === index
                    ? 'bg-accent-green/20 text-accent-green border border-accent-green/30'
                    : 'bg-bg-secondary text-text-secondary border border-bg-tertiary hover:border-accent-green/30'
                }`}
              >
                {constraint.constraint}
              </button>
            ))}
          </div>

          {/* Active Constraint Details */}
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8 mb-8">
            <h3 className="text-2xl font-bold text-text-primary mb-4">
              {temporalConstraints[activeConstraint].constraint}
            </h3>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div>
                <h4 className="font-semibold text-accent-green mb-3">Description:</h4>
                <p className="text-text-secondary mb-6">{temporalConstraints[activeConstraint].description}</p>
                
                <h4 className="font-semibold text-accent-blue mb-3">Example:</h4>
                <div className="bg-accent-blue/5 border border-accent-blue/20 rounded-lg p-4">
                  <p className="text-text-secondary text-sm">{temporalConstraints[activeConstraint].example}</p>
                </div>
              </div>
              
              <div>
                <h4 className="font-semibold text-accent-red mb-3">Risk if Violated:</h4>
                <div className="bg-accent-red/5 border border-accent-red/20 rounded-lg p-4">
                  <div className="flex items-start space-x-3">
                    <AlertTriangle className="w-5 h-5 text-accent-red mt-0.5 flex-shrink-0" />
                    <p className="text-text-secondary text-sm">{temporalConstraints[activeConstraint].risk}</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Temporal Setup Implementation */}
          <CodeBlock
            language="python"
            title="Temporal Data Alignment Implementation"
            code={`# Temporal alignment ensuring no look-ahead bias

def create_temporal_prediction_data(df_merged, oos_start_year_month_int, predictor_cols):
    """
    Create temporally aligned prediction arrays from merged financial data.
    
    Critical insight: To predict equity premium at time t+1, we can only use
    predictor information available at time t.
    
    Args:
        df_merged: DataFrame with predictors and log_equity_premium
        oos_start_year_month_int: Start of out-of-sample period (e.g., 200001)
        predictor_cols: List of predictor column names
        
    Returns:
        dict with temporally aligned arrays for prediction
    """
    
    # CRITICAL: Temporal alignment
    # Target: log_equity_premium at time t+1 (what we want to predict)
    log_ep_tplus1 = df_merged['log_equity_premium'].values[1:]  # [t+1, t+2, ..., T]
    
    # Predictors: Features available at time t (when making prediction)  
    X_t = df_merged[predictor_cols].iloc[:-1, :]  # [t, t+1, ..., T-1]
    
    # Dates: Time periods when predictors are observed (time t)
    dates_t = df_merged['month'].dt.strftime('%Y%m').astype(int).values[:-1]
    
    # Combine into prediction array: [target, features]
    prediction_array = np.concatenate([
        log_ep_tplus1.reshape(-1, 1),  # Target column
        X_t.values                     # Feature columns
    ], axis=1)
    
    # Find out-of-sample start index
    oos_start_idx = np.where(dates_t >= oos_start_year_month_int)[0][0]
    
    return {
        'prediction_array': prediction_array,
        'dates_t': dates_t,
        'target_values': log_ep_tplus1,
        'oos_start_idx': oos_start_idx,
        'alignment_check': {
            'predictor_periods': len(X_t),
            'target_periods': len(log_ep_tplus1),
            'first_prediction_date': dates_t[0],
            'last_prediction_date': dates_t[-1]
        }
    }

def validate_temporal_alignment(data_dict):
    """
    Verify that temporal alignment is correct - no look-ahead bias.
    """
    
    print("Temporal Alignment Validation:")
    print("=" * 40)
    
    alignment = data_dict['alignment_check']
    print(f"✓ Predictor periods: {alignment['predictor_periods']}")
    print(f"✓ Target periods: {alignment['target_periods']}")
    print(f"✓ Periods match: {alignment['predictor_periods'] == alignment['target_periods']}")
    
    print(f"\\n📅 Prediction Timeline:")
    print(f"  First prediction for: {alignment['first_prediction_date'] + 1}")
    print(f"  Using predictors from: {alignment['first_prediction_date']}")
    print(f"  Last prediction for: {alignment['last_prediction_date'] + 1}")
    print(f"  Using predictors from: {alignment['last_prediction_date']}")
    
    # Example validation for specific period
    dates = data_dict['dates_t']
    targets = data_dict['target_values']
    
    print(f"\\n🔍 Example Alignment (first 3 predictions):")
    for i in range(min(3, len(dates))):
        pred_date = dates[i]
        target_date = pred_date + 1 if pred_date % 100 < 12 else (pred_date // 100 + 1) * 100 + 1
        print(f"  Predictors from {pred_date} → Predict {target_date} equity premium")
    
    return True

# Example of proper train/test split maintaining temporal order
def temporal_train_test_split(data_dict, test_start_date):
    """
    Split data maintaining temporal order - all training data comes before test data.
    """
    
    dates = data_dict['dates_t']
    prediction_array = data_dict['prediction_array']
    
    # Find split index
    split_idx = np.where(dates >= test_start_date)[0][0]
    
    # Split maintaining temporal order
    X_train = prediction_array[:split_idx, 1:]     # Features before test period
    y_train = prediction_array[:split_idx, 0]      # Targets before test period
    X_test = prediction_array[split_idx:, 1:]      # Features during test period  
    y_test = prediction_array[split_idx:, 0]       # Targets during test period
    
    print(f"Temporal Split Summary:")
    print(f"  Training period: {dates[0]} to {dates[split_idx-1]}")
    print(f"  Test period: {dates[split_idx]} to {dates[-1]}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    return X_train, y_train, X_test, y_test, split_idx`}
          />
        </motion.section>

        {/* PyTorch Tensor Conversion */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Database className="w-8 h-8 text-accent-purple mr-3" />
            Step 3: PyTorch Tensor Conversion
          </h2>
          
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8 mb-8">
            <p className="text-text-secondary text-lg mb-6">
              Neural networks in PyTorch require data in tensor format for GPU acceleration and automatic differentiation. 
              This conversion step transforms preprocessed NumPy arrays into PyTorch tensors.
            </p>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-6">
              <div>
                <h3 className="text-xl font-semibold text-text-primary mb-4">Tensor Properties:</h3>
                <div className="space-y-3">
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-purple mt-2"></div>
                    <span className="text-text-secondary"><strong>Automatic differentiation:</strong> Enables backpropagation for training</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-purple mt-2"></div>
                    <span className="text-text-secondary"><strong>GPU acceleration:</strong> Can be moved to GPU for faster computation</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-purple mt-2"></div>
                    <span className="text-text-secondary"><strong>Batch processing:</strong> Efficient parallel operations on batches</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-purple mt-2"></div>
                    <span className="text-text-secondary"><strong>Memory efficiency:</strong> Optimized memory layout for deep learning</span>
                  </div>
                </div>
              </div>
              <div>
                <h3 className="text-xl font-semibold text-text-primary mb-4">Data Type Considerations:</h3>
                <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4 space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-text-secondary">Input features:</span>
                    <span className="font-mono text-text-primary">torch.float32</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-text-secondary">Target values:</span>
                    <span className="font-mono text-text-primary">torch.float32</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-text-secondary">Device:</span>
                    <span className="font-mono text-text-primary">CPU or CUDA</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-text-secondary">Gradients:</span>
                    <span className="font-mono text-text-primary">requires_grad=True for training</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <CodeBlock
            language="python"
            title="PyTorch Tensor Conversion Implementation"
            code={`# PyTorch tensor conversion for neural network training

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

def convert_to_tensors(X_train, y_train, X_test=None, y_test=None, device='cpu'):
    """
    Convert preprocessed data to PyTorch tensors.
    
    Args:
        X_train: Training features (numpy array)
        y_train: Training targets (numpy array) 
        X_test: Test features (optional)
        y_test: Test targets (optional)
        device: 'cpu' or 'cuda' for GPU acceleration
        
    Returns:
        Dictionary containing PyTorch tensors ready for neural network training
    """
    
    # Convert training data to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)
    
    # Reshape targets to [batch_size, 1] for regression
    if y_train_tensor.dim() == 1:
        y_train_tensor = y_train_tensor.unsqueeze(1)
    
    result = {
        'X_train': X_train_tensor,
        'y_train': y_train_tensor,
        'device': device,
        'train_shape': X_train_tensor.shape,
        'target_shape': y_train_tensor.shape
    }
    
    # Convert test data if provided
    if X_test is not None and y_test is not None:
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device=device)
        
        if y_test_tensor.dim() == 1:
            y_test_tensor = y_test_tensor.unsqueeze(1)
        
        result.update({
            'X_test': X_test_tensor,
            'y_test': y_test_tensor,
            'test_shape': X_test_tensor.shape
        })
    
    return result

def complete_preprocessing_pipeline(df_merged, predictor_cols, oos_start_date, device='cpu'):
    """
    Complete preprocessing pipeline from raw data to PyTorch tensors.
    
    This function demonstrates the full preprocessing workflow used in the
    equity premium prediction system.
    """
    
    print("🔄 Starting Complete Preprocessing Pipeline")
    print("=" * 50)
    
    # Step 1: Create temporal prediction arrays
    print("📅 Step 1: Creating temporal prediction arrays...")
    
    # Target: log equity premium at time t+1
    log_ep_tplus1 = df_merged['log_equity_premium'].values[1:]
    
    # Predictors: Financial indicators at time t  
    X_t = df_merged[predictor_cols].iloc[:-1, :].values
    
    # Dates: Time periods for prediction alignment
    dates_t = df_merged['month'].dt.strftime('%Y%m').astype(int).values[:-1]
    
    print(f"  ✓ Created {len(log_ep_tplus1)} prediction pairs")
    print(f"  ✓ Features shape: {X_t.shape}")
    print(f"  ✓ Targets shape: {log_ep_tplus1.shape}")
    
    # Step 2: Split into train/test maintaining temporal order
    print(f"\\n🔀 Step 2: Temporal train/test split at {oos_start_date}...")
    
    split_idx = np.where(dates_t >= oos_start_date)[0][0]
    
    X_train_raw = X_t[:split_idx]
    y_train = log_ep_tplus1[:split_idx]
    X_test_raw = X_t[split_idx:]
    y_test = log_ep_tplus1[split_idx:]
    
    print(f"  ✓ Training samples: {len(X_train_raw)} ({dates_t[0]} to {dates_t[split_idx-1]})")
    print(f"  ✓ Test samples: {len(X_test_raw)} ({dates_t[split_idx]} to {dates_t[-1]})")
    
    # Step 3: Standardize features using only training data
    print(f"\\n📊 Step 3: Standardizing features...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)  # Use training statistics
    
    print(f"  ✓ Training data: mean={X_train_scaled.mean():.6f}, std={X_train_scaled.std():.6f}")
    print(f"  ✓ Features standardized using training statistics only")
    
    # Step 4: Convert to PyTorch tensors
    print(f"\\n🔥 Step 4: Converting to PyTorch tensors on {device}...")
    
    tensors = convert_to_tensors(
        X_train_scaled, y_train, 
        X_test_scaled, y_test, 
        device=device
    )
    
    print(f"  ✓ Training tensors: {tensors['train_shape']} features, {tensors['target_shape']} targets")
    if 'test_shape' in tensors:
        print(f"  ✓ Test tensors: {tensors['test_shape']} features")
    
    # Step 5: Validation checks
    print(f"\\n✅ Step 5: Final validation...")
    
    # Check for NaN or infinite values
    train_has_nan = torch.isnan(tensors['X_train']).any() or torch.isnan(tensors['y_train']).any()
    print(f"  ✓ No NaN values in training data: {not train_has_nan}")
    
    # Check tensor properties
    print(f"  ✓ Device: {tensors['X_train'].device}")
    print(f"  ✓ Data type: {tensors['X_train'].dtype}")
    print(f"  ✓ Requires gradient: {tensors['X_train'].requires_grad}")
    
    # Add metadata for tracking
    tensors.update({
        'scaler': scaler,
        'split_date': oos_start_date,
        'split_idx': split_idx,
        'feature_names': predictor_cols,
        'preprocessing_complete': True
    })
    
    print(f"\\n🎯 Preprocessing pipeline complete!")
    print(f"   Ready for neural network training and evaluation.")
    
    return tensors

# Example usage
if __name__ == "__main__":
    # This would be called with actual financial data
    # tensors = complete_preprocessing_pipeline(
    #     df_merged=df_financial_data,
    #     predictor_cols=PREDICTOR_COLS_30,
    #     oos_start_date=200001,  # January 2000
    #     device='cuda' if torch.cuda.is_available() else 'cpu'
    # )`}
          />
        </motion.section>

        {/* Data Quality Checks */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.7 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Shield className="w-8 h-8 text-accent-orange mr-3" />
            Step 4: Data Quality Validation
          </h2>
          
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8 mb-8">
            <p className="text-text-secondary text-lg mb-6">
              Before training neural networks, comprehensive validation ensures data integrity and identifies potential issues 
              that could affect model performance or introduce bias.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <h3 className="text-xl font-semibold text-text-primary">Validation Checks:</h3>
                <div className="space-y-3 text-sm">
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-orange mt-2"></div>
                    <span className="text-text-secondary"><strong>Missing values:</strong> Ensure no NaN or missing data</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-orange mt-2"></div>
                    <span className="text-text-secondary"><strong>Infinite values:</strong> Check for inf/-inf from calculations</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-orange mt-2"></div>
                    <span className="text-text-secondary"><strong>Data ranges:</strong> Verify values are within expected bounds</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-orange mt-2"></div>
                    <span className="text-text-secondary"><strong>Temporal consistency:</strong> Confirm chronological ordering</span>
                  </div>
                </div>
              </div>
              <div className="space-y-4">
                <h3 className="text-xl font-semibold text-text-primary">Statistical Properties:</h3>
                <div className="space-y-3 text-sm">
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-green mt-2"></div>
                    <span className="text-text-secondary"><strong>Standardization check:</strong> Mean ≈ 0, Std ≈ 1</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-green mt-2"></div>
                    <span className="text-text-secondary"><strong>Distribution analysis:</strong> Identify outliers and skewness</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-green mt-2"></div>
                    <span className="text-text-secondary"><strong>Correlation structure:</strong> Check feature relationships</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-green mt-2"></div>
                    <span className="text-text-secondary"><strong>Target distribution:</strong> Verify reasonable target values</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <CodeBlock
            language="python"
            title="Comprehensive Data Quality Validation"
            code={`# Comprehensive data quality validation for financial neural networks

import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List

def validate_preprocessing_quality(tensors: Dict[str, torch.Tensor], 
                                 feature_names: List[str]) -> Dict[str, Any]:
    """
    Comprehensive validation of preprocessed data quality.
    
    Args:
        tensors: Dictionary containing X_train, y_train, X_test, y_test tensors
        feature_names: List of feature names for detailed reporting
        
    Returns:
        Validation report with pass/fail status and detailed statistics
    """
    
    print("🔍 Data Quality Validation Report")
    print("=" * 50)
    
    validation_results = {
        'passed': True,
        'warnings': [],
        'errors': [],
        'statistics': {}
    }
    
    # 1. Check for missing or invalid values
    print("\\n1️⃣ Missing/Invalid Value Checks:")
    
    for name, tensor in tensors.items():
        if name.startswith('X_') or name.startswith('y_'):
            has_nan = torch.isnan(tensor).any().item()
            has_inf = torch.isinf(tensor).any().item()
            
            print(f"  {name:12s}: NaN={has_nan:5s}, Inf={has_inf:5s}, Shape={tensor.shape}")
            
            if has_nan:
                validation_results['errors'].append(f"{name} contains NaN values")
                validation_results['passed'] = False
            
            if has_inf:
                validation_results['errors'].append(f"{name} contains infinite values")
                validation_results['passed'] = False
    
    # 2. Standardization validation
    print("\\n2️⃣ Standardization Validation:")
    
    if 'X_train' in tensors:
        X_train = tensors['X_train']
        feature_means = X_train.mean(dim=0)
        feature_stds = X_train.std(dim=0)
        
        # Check if features are properly standardized (allowing small numerical errors)
        mean_check = torch.abs(feature_means) < 1e-6
        std_check = torch.abs(feature_stds - 1.0) < 1e-6
        
        mean_passed = mean_check.all().item()
        std_passed = std_check.all().item()
        
        print(f"  Mean ≈ 0: {mean_passed} (max deviation: {torch.abs(feature_means).max():.2e})")
        print(f"  Std ≈ 1:  {std_passed} (max deviation: {torch.abs(feature_stds - 1.0).max():.2e})")
        
        if not mean_passed:
            validation_results['warnings'].append("Feature means deviate from zero")
        if not std_passed:
            validation_results['warnings'].append("Feature standard deviations deviate from one")
        
        # Store statistics
        validation_results['statistics']['feature_means'] = feature_means.tolist()
        validation_results['statistics']['feature_stds'] = feature_stds.tolist()
    
    # 3. Target variable validation
    print("\\n3️⃣ Target Variable Validation:")
    
    if 'y_train' in tensors:
        y_train = tensors['y_train']
        
        target_mean = y_train.mean().item()
        target_std = y_train.std().item()
        target_min = y_train.min().item()
        target_max = y_train.max().item()
        
        print(f"  Mean: {target_mean:8.4f}")
        print(f"  Std:  {target_std:8.4f}")
        print(f"  Min:  {target_min:8.4f}")
        print(f"  Max:  {target_max:8.4f}")
        print(f"  Range: [{target_min:.4f}, {target_max:.4f}]")
        
        # Check for reasonable equity premium values (log returns typically -50% to +50%)
        if target_min < -0.5 or target_max > 0.5:
            validation_results['warnings'].append(
                f"Target values outside typical range: [{target_min:.4f}, {target_max:.4f}]"
            )
        
        validation_results['statistics']['target_stats'] = {
            'mean': target_mean,
            'std': target_std,
            'min': target_min,
            'max': target_max
        }
    
    # 4. Data shape consistency
    print("\\n4️⃣ Shape Consistency Validation:")
    
    if 'X_train' in tensors and 'y_train' in tensors:
        X_train_samples = tensors['X_train'].shape[0]
        y_train_samples = tensors['y_train'].shape[0]
        
        shape_consistent = X_train_samples == y_train_samples
        print(f"  Training shapes match: {shape_consistent} ({X_train_samples} vs {y_train_samples})")
        
        if not shape_consistent:
            validation_results['errors'].append("Training data shape mismatch")
            validation_results['passed'] = False
    
    if 'X_test' in tensors and 'y_test' in tensors:
        X_test_samples = tensors['X_test'].shape[0]
        y_test_samples = tensors['y_test'].shape[0]
        
        test_shape_consistent = X_test_samples == y_test_samples
        print(f"  Test shapes match: {test_shape_consistent} ({X_test_samples} vs {y_test_samples})")
        
        if not test_shape_consistent:
            validation_results['errors'].append("Test data shape mismatch")
            validation_results['passed'] = False
    
    # 5. Feature-specific validation
    print("\\n5️⃣ Feature-Specific Validation:")
    
    if 'X_train' in tensors and feature_names:
        X_train = tensors['X_train']
        n_features_expected = len(feature_names)
        n_features_actual = X_train.shape[1]
        
        feature_count_match = n_features_expected == n_features_actual
        print(f"  Feature count: {feature_count_match} (expected {n_features_expected}, got {n_features_actual})")
        
        if not feature_count_match:
            validation_results['errors'].append(
                f"Feature count mismatch: expected {n_features_expected}, got {n_features_actual}"
            )
            validation_results['passed'] = False
        
        # Check for features with zero variance (post-standardization)
        zero_variance_features = (X_train.std(dim=0) < 1e-8).nonzero().flatten()
        if len(zero_variance_features) > 0:
            zero_var_names = [feature_names[i] for i in zero_variance_features]
            print(f"  Zero variance features: {zero_var_names}")
            validation_results['warnings'].append(f"Features with zero variance: {zero_var_names}")
    
    # 6. Memory and device validation
    print("\\n6️⃣ Memory and Device Validation:")
    
    total_memory_mb = 0
    devices = set()
    
    for name, tensor in tensors.items():
        if isinstance(tensor, torch.Tensor):
            memory_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
            total_memory_mb += memory_mb
            devices.add(str(tensor.device))
            
            print(f"  {name:12s}: {memory_mb:6.1f} MB on {tensor.device}")
    
    print(f"  Total memory: {total_memory_mb:.1f} MB")
    print(f"  Devices used: {list(devices)}")
    
    if len(devices) > 1:
        validation_results['warnings'].append("Tensors on different devices")
    
    # 7. Final validation summary
    print("\\n" + "=" * 50)
    
    if validation_results['passed']:
        if len(validation_results['warnings']) == 0:
            print("✅ ALL VALIDATION CHECKS PASSED")
        else:
            print("⚠️  VALIDATION PASSED WITH WARNINGS")
            for warning in validation_results['warnings']:
                print(f"    Warning: {warning}")
    else:
        print("❌ VALIDATION FAILED")
        for error in validation_results['errors']:
            print(f"    Error: {error}")
    
    return validation_results

# Usage example
def example_validation():
    """Example of how to use the validation function."""
    
    # Simulated tensors (in practice, these come from preprocessing)
    tensors = {
        'X_train': torch.randn(1000, 30),  # Standardized features
        'y_train': torch.randn(1000, 1) * 0.1,  # Log equity premiums
        'X_test': torch.randn(200, 30),
        'y_test': torch.randn(200, 1) * 0.1
    }
    
    feature_names = [f"feature_{i}" for i in range(30)]
    
    validation_report = validate_preprocessing_quality(tensors, feature_names)
    
    return validation_report`}
          />
        </motion.section>

        {/* Navigation */}
        <NavigationButtons 
          prevHref="/data-setup"
          prevLabel="Data & Problem Setup"
          nextHref="/architecture"
          nextLabel="Neural Network Architecture"
        />
      </div>
    </div>
  )
}