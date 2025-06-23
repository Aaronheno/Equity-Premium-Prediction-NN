'use client'

import { motion } from 'framer-motion'
import { useState } from 'react'
import { Database, TrendingUp, Calculator, FileText, BarChart3, Clock, Target, AlertTriangle } from 'lucide-react'
import CodeBlock from '@/components/shared/CodeBlock'
import MathFormula from '@/components/shared/MathFormula'
import NavigationButtons from '@/components/shared/NavigationButtons'

const financialIndicators = {
  'Valuation Ratios': {
    color: 'accent-blue',
    icon: TrendingUp,
    description: 'Traditional predictors measuring market valuation levels',
    indicators: [
      { code: 'DP', name: 'Dividend-Price Ratio', description: 'Annual dividends divided by current price' },
      { code: 'DY', name: 'Dividend Yield', description: 'Dividends per share divided by price per share' },
      { code: 'EP', name: 'Earnings-Price Ratio', description: 'Annual earnings divided by current price' },
      { code: 'BM', name: 'Book-to-Market Ratio', description: 'Book value of equity divided by market value' }
    ],
    intuition: 'When stocks are expensive relative to fundamentals (low DP, DY, EP, BM), future returns tend to be lower.',
    reference: 'Campbell & Shiller (1988), Fama & French (1988)'
  },
  'Interest Rates & Yield Curve': {
    color: 'accent-green',
    icon: BarChart3,
    description: 'Interest rate environment and yield curve variables',
    indicators: [
      { code: 'TBL', name: 'Treasury Bill Rate', description: 'Short-term risk-free rate' },
      { code: 'LTR', name: 'Long-Term Return', description: 'Long-term government bond return' },
      { code: 'LTY', name: 'Long-Term Yield', description: 'Long-term government bond yield' },
      { code: 'TMS', name: 'Term Spread', description: 'Difference between long and short-term rates' }
    ],
    intuition: 'Interest rates reflect economic conditions and alternative investment opportunities. Steep yield curves often predict higher growth.',
    reference: 'Fama & French (1989), Campbell (1987)'
  },
  'Credit & Risk Measures': {
    color: 'accent-purple',
    icon: AlertTriangle,
    description: 'Credit spreads and risk indicators',
    indicators: [
      { code: 'DFY', name: 'Default Yield Spread', description: 'Difference between BAA and AAA corporate bond yields' },
      { code: 'DFR', name: 'Default Return Spread', description: 'Difference between long-term corporate and government bond returns' },
      { code: 'NTIS', name: 'Net Equity Expansion', description: 'Net issues by NYSE listed stocks' }
    ],
    intuition: 'Higher credit spreads indicate economic stress and higher risk premiums. When companies issue more equity (high NTIS), it often signals overvaluation.',
    reference: 'Fama & French (1989), Baker & Wurgler (2000)'
  },
  'Inflation & Volatility': {
    color: 'accent-orange',
    icon: TrendingUp,
    description: 'Inflation and market volatility measures',
    indicators: [
      { code: 'INFL', name: 'Inflation Rate', description: 'Consumer price inflation' },
      { code: 'SVAR', name: 'Stock Variance', description: 'Measure of market volatility' },
      { code: 'DE', name: 'Debt-to-Equity Ratio', description: 'Corporate leverage measure' }
    ],
    intuition: 'High inflation erodes real returns. High volatility indicates uncertainty and typically demands higher risk premiums.',
    reference: 'Fama & Schwert (1977), French et al. (1987)'
  },
  'Technical Indicators': {
    color: 'accent-red',
    icon: BarChart3,
    description: 'Moving averages, momentum, volatility, and volume indicators',
    indicators: [
      { code: 'MA', name: 'Moving Averages', description: '1, 2, 3-month prices relative to 9 and 12-month moving averages (6 total)' },
      { code: 'MOM', name: 'Momentum', description: '1, 2, 3, 6, 9, 12-month momentum indicators (6 total)' },
      { code: 'VOL', name: 'Volatility Ratios', description: '1, 2, 3-month volatility relative to 9 and 12-month averages (6 total)' },
      { code: 'OBV', name: 'On Balance Volume', description: 'Cumulative volume-based momentum indicator (post-1950)' }
    ],
    intuition: 'Technical indicators capture momentum and trend-following behavior. Moving averages and volatility patterns signal regime shifts in market conditions.',
    reference: 'Jegadeesh & Titman (1993), Lo et al. (2000)'
  }
}

const challenges = [
  {
    title: 'Low Signal-to-Noise Ratio',
    description: 'Market returns are ~95% noise, ~5% signal',
    icon: Target,
    color: 'accent-red'
  },
  {
    title: 'Time-Varying Relationships',
    description: 'What predicts returns changes over market cycles',
    icon: Clock,
    color: 'accent-orange'
  },
  {
    title: 'Economic vs Statistical Significance',
    description: 'Small or even negative R² can still be economically valuable',
    icon: TrendingUp,
    color: 'accent-green'
  },
  {
    title: 'Look-Ahead Bias Prevention',
    description: 'Must prevent any future information leakage',
    icon: AlertTriangle,
    color: 'accent-blue'
  }
]

export default function DataSetupContent() {
  const [activeIndicatorGroup, setActiveIndicatorGroup] = useState('Valuation Ratios')

  const activeGroup = financialIndicators[activeIndicatorGroup as keyof typeof financialIndicators]

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
            <span className="gradient-text">Data & Problem Setup</span>
          </h1>
          <p className="max-w-4xl mx-auto text-xl text-text-secondary leading-relaxed">
            Understanding the equity premium prediction challenge: from 32 financial indicators 
            to the mathematical foundations that make neural network prediction possible.
          </p>
        </motion.div>

        {/* Problem Definition */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Calculator className="w-8 h-8 text-accent-blue mr-3" />
            What is Equity Premium Prediction?
          </h2>
          
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8 mb-8">
            <p className="text-text-secondary text-lg mb-6 leading-relaxed">
              The <strong className="text-text-primary">equity premium</strong> is the excess return that investing in the stock market provides over a risk-free rate.
            </p>
            
            <div className="text-center mb-6">
              <MathFormula 
                latex="EP = R_m - R_f"
                block={true}
              />
            </div>
            
            <p className="text-text-secondary mb-6">
              Equity premium prediction asks: <em>"Can we use current financial and economic information to predict future stock market excess returns?"</em>{' '}
              This prediction is crucial for:
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-3">
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 rounded-full bg-accent-blue mt-2"></div>
                  <div>
                    <strong className="text-text-primary">Portfolio allocation:</strong>
                    <span className="text-text-secondary"> How much should investors allocate to stocks vs. bonds?</span>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 rounded-full bg-accent-blue mt-2"></div>
                  <div>
                    <strong className="text-text-primary">Market timing:</strong>
                    <span className="text-text-secondary"> When should investors increase or decrease market exposure?</span>
                  </div>
                </div>
              </div>
              <div className="space-y-3">
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 rounded-full bg-accent-purple mt-2"></div>
                  <div>
                    <strong className="text-text-primary">Economic policy:</strong>
                    <span className="text-text-secondary"> Understanding what drives market risk premiums helps policymakers</span>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 rounded-full bg-accent-purple mt-2"></div>
                  <div>
                    <strong className="text-text-primary">Academic research:</strong>
                    <span className="text-text-secondary"> Testing theories about market efficiency and risk pricing</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </motion.section>

        {/* The Challenge */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <AlertTriangle className="w-8 h-8 text-accent-orange mr-3" />
            The Challenge
          </h2>
          
          <p className="text-text-secondary text-lg mb-8 leading-relaxed">
            Predicting equity premiums is notoriously difficult due to fundamental characteristics of financial markets:
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {challenges.map((challenge, index) => {
              const Icon = challenge.icon
              return (
                <motion.div
                  key={challenge.title}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: 0.4 + index * 0.1 }}
                  className="bg-bg-secondary border border-bg-tertiary rounded-xl p-6"
                >
                  <div className="flex items-center space-x-3 mb-4">
                    <div className={`inline-flex items-center justify-center w-10 h-10 rounded-lg bg-${challenge.color}/10 border border-${challenge.color}/20`}>
                      <Icon className={`w-5 h-5 text-${challenge.color}`} />
                    </div>
                    <h3 className="font-semibold text-text-primary">{challenge.title}</h3>
                  </div>
                  <p className="text-text-secondary text-sm">{challenge.description}</p>
                </motion.div>
              )
            })}
          </div>
        </motion.section>

        {/* 31 Financial Indicators */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Database className="w-8 h-8 text-accent-green mr-3" />
            The Data: 32 Financial Indicators
          </h2>
          
          <p className="text-text-secondary text-lg mb-8 leading-relaxed">
            The neural networks use 32 carefully selected financial indicators as input features. 
            These variables capture different aspects of market conditions, valuation, and economic environment.
            The macroeconomic and fundamental variables are available from{' '}
            <a href="https://sites.google.com/view/agoyal145" target="_blank" rel="noopener noreferrer" className="text-accent-blue hover:underline">
              Amit Goyal's webpage
            </a>.
          </p>

          {/* Indicator Group Selector */}
          <div className="flex flex-wrap gap-2 mb-8">
            {Object.keys(financialIndicators).map((group) => {
              const groupData = financialIndicators[group as keyof typeof financialIndicators]
              return (
                <button
                  key={group}
                  onClick={() => setActiveIndicatorGroup(group)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                    activeIndicatorGroup === group
                      ? `bg-${groupData.color}/20 text-${groupData.color} border border-${groupData.color}/30`
                      : 'bg-bg-secondary text-text-secondary border border-bg-tertiary hover:border-accent-blue/30'
                  }`}
                >
                  {group}
                </button>
              )
            })}
          </div>

          {/* Active Indicator Group Details */}
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8">
            <div className="flex items-center space-x-3 mb-6">
              <div className={`inline-flex items-center justify-center w-12 h-12 rounded-lg bg-${activeGroup.color}/10 border border-${activeGroup.color}/20`}>
                <activeGroup.icon className={`w-6 h-6 text-${activeGroup.color}`} />
              </div>
              <div>
                <h3 className="text-2xl font-bold text-text-primary">{activeIndicatorGroup}</h3>
                <p className="text-text-secondary">{activeGroup.description}</p>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
              {activeGroup.indicators.map((indicator, index) => (
                <div key={index} className="bg-bg-primary border border-bg-tertiary rounded-lg p-4">
                  <div className="flex items-start space-x-3">
                    <div className={`inline-flex items-center justify-center w-8 h-8 rounded bg-${activeGroup.color}/20 text-${activeGroup.color} font-mono text-xs font-bold`}>
                      {indicator.code}
                    </div>
                    <div>
                      <h4 className="font-semibold text-text-primary text-sm">{indicator.name}</h4>
                      <p className="text-text-secondary text-xs">{indicator.description}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <div className="bg-accent-blue/5 border border-accent-blue/20 rounded-lg p-4">
              <h4 className="font-semibold text-accent-blue mb-2">Economic Intuition:</h4>
              <p className="text-text-secondary text-sm">{activeGroup.intuition}</p>
              {activeGroup.reference && (
                <p className="text-text-muted text-xs mt-2">
                  <span className="font-medium">References:</span> {activeGroup.reference}
                </p>
              )}
            </div>
          </div>
        </motion.section>

        {/* Target Variable */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.5 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Target className="w-8 h-8 text-accent-purple mr-3" />
            Target Variable: Log Equity Premium
          </h2>
          
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8">
            <p className="text-text-secondary mb-6">
              The target variable is the <strong className="text-text-primary">log equity premium</strong>:
            </p>
            
            <div className="text-center mb-8">
              <MathFormula 
                latex="\log(EP) = \log(1 + R_m) - \log(1 + R_f)"
                block={true}
              />
            </div>

            <h3 className="text-xl font-semibold text-text-primary mb-4">Why Use Log Transformation?</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
              <div className="space-y-4">
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 rounded-full bg-accent-purple mt-2"></div>
                  <div>
                    <strong className="text-text-primary">Mathematical convenience:</strong>
                    <span className="text-text-secondary"> Log returns are additive across time periods</span>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 rounded-full bg-accent-purple mt-2"></div>
                  <div>
                    <strong className="text-text-primary">Symmetry:</strong>
                    <span className="text-text-secondary"> A 50% gain and 33% loss have equal magnitude in log space</span>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 rounded-full bg-accent-purple mt-2"></div>
                  <div>
                    <strong className="text-text-primary">Normality:</strong>
                    <span className="text-text-secondary"> Log returns are closer to normally distributed</span>
                  </div>
                </div>
              </div>
              <div className="space-y-4">
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 rounded-full bg-accent-purple mt-2"></div>
                  <div>
                    <strong className="text-text-primary">Compounding:</strong>
                    <span className="text-text-secondary"> Log returns naturally account for compound growth</span>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 rounded-full bg-accent-purple mt-2"></div>
                  <div>
                    <strong className="text-text-primary">Stability:</strong>
                    <span className="text-text-secondary"> Reduces the impact of extreme outliers</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-accent-purple/5 border border-accent-purple/20 rounded-lg p-4">
              <p className="text-text-secondary text-sm mb-2">
                For small returns, log returns approximate simple returns:
              </p>
              <div className="text-center">
                <MathFormula latex="\log(1 + r) \approx r" />
                <p className="text-text-muted text-xs mt-2">(when r is small)</p>
              </div>
            </div>
          </div>
        </motion.section>

        {/* Temporal Structure */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Clock className="w-8 h-8 text-accent-blue mr-3" />
            Time Series Nature & Temporal Constraints
          </h2>
          
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8">
            <p className="text-text-secondary text-lg mb-6">
              This is a <strong className="text-text-primary">time series prediction problem</strong> with strict temporal constraints:
            </p>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div>
                <h3 className="text-xl font-semibold text-text-primary mb-4">The Prediction Setup</h3>
                <div className="space-y-3 mb-6">
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-blue mt-2"></div>
                    <span className="text-text-secondary"><strong>Training data:</strong> All information available up to time t</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-blue mt-2"></div>
                    <span className="text-text-secondary"><strong>Prediction target:</strong> Equity premium at time t+1</span>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 rounded-full bg-accent-red mt-2"></div>
                    <span className="text-text-secondary"><strong>Key constraint:</strong> No look-ahead bias (no future information)</span>
                  </div>
                </div>

                <div className="text-center mb-4">
                  <MathFormula latex="y_{t+1} = f(X_t) + \varepsilon_{t+1}" block={true} />
                </div>

                <div className="text-sm text-text-muted space-y-1">
                  <p>Where:</p>
                  <p>• <MathFormula latex="y_{t+1}" /> = log equity premium at time t+1 (what we predict)</p>
                  <p>• <MathFormula latex="X_t" /> = vector of 32 predictor variables known at time t</p>
                  <p>• <MathFormula latex="f(\cdot)" /> = our neural network function</p>
                  <p>• <MathFormula latex="\varepsilon_{t+1}" /> = unpredictable error term</p>
                </div>
              </div>

              <div>
                <h3 className="text-xl font-semibold text-text-primary mb-4">Temporal Structure</h3>
                <div className="space-y-4">
                  <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4">
                    <h4 className="font-semibold text-accent-green mb-2">Monthly frequency</h4>
                    <p className="text-text-secondary text-sm">Predictions are made monthly</p>
                  </div>
                  <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4">
                    <h4 className="font-semibold text-accent-blue mb-2">Expanding window</h4>
                    <p className="text-text-secondary text-sm">Training data grows over time (no fixed window)</p>
                  </div>
                  <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4">
                    <h4 className="font-semibold text-accent-purple mb-2">Annual retraining</h4>
                    <p className="text-text-secondary text-sm">Hyperparameters re-optimized yearly</p>
                  </div>
                  <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4">
                    <h4 className="font-semibold text-accent-orange mb-2">Out-of-sample testing</h4>
                    <p className="text-text-secondary text-sm">Strict separation between training and testing periods</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </motion.section>

        {/* Data Loading Code */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.7 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <FileText className="w-8 h-8 text-accent-green mr-3" />
            Loading Data: Implementation
          </h2>
          
          <p className="text-text-secondary text-lg mb-6">
            Here's how the system loads and prepares the raw data from <code className="text-accent-green bg-code-bg px-2 py-1 rounded">data/ml_equity_premium_data.xlsx</code>:
          </p>

          <CodeBlock
            language="python"
            title="Data Loading Implementation"
            code={`# From src/utils/io.py

def load_and_prepare_oos_data(oos_start_year_month_int, predictor_cols=None):
    """
    Loads and prepares data for Out-of-Sample evaluation.
    
    Args:
        oos_start_year_month_int: OOS start date (e.g., 200001 for Jan 2000)
        predictor_cols: List of predictor column names (default: 32 standard predictors)
    
    Returns:
        dict with arrays aligned for prediction:
        - predictor_array_for_oos: [y_{t+1}, X_t] for all time periods
        - dates_all_t_np: Time stamps for each observation
        - actual_log_ep_all_np: True equity premiums
        - oos_start_idx_in_arrays: Where OOS period begins
    """
    
    # Load raw Excel data
    df_result_predictor, df_market_rf = _load_raw_data_from_excel()
    
    # Merge predictor data with market returns
    df_merged = pd.merge(df_result_predictor, df_market_rf, on='month', how='inner')
    
    # Create the prediction array: [y_{t+1}, X_t]
    # Target: log_equity_premium at time t+1
    log_ep_tplus1 = df_merged['log_equity_premium'].values[1:]  # Shape: (N-1,)
    
    # Predictors: X_t from previous period
    X_t_df = df_merged[predictor_cols].iloc[:-1, :]  # Shape: (N-1, 32)
    
    # Combine into prediction array
    predictor_array_for_oos = np.concatenate(
        [log_ep_tplus1.reshape(-1, 1), X_t_df.values], axis=1
    )  # Shape: (N-1, 33) - first column is target, next 32 are features
    
    # Extract dates (time t when predictors are observed)
    dates_t = df_merged['month'].dt.strftime('%Y%m').astype(int).values[:-1]
    
    return {
        'dates_all_t_np': dates_t,
        'predictor_array_for_oos': predictor_array_for_oos,
        'actual_log_ep_all_np': log_ep_tplus1,
        'oos_start_idx_in_arrays': np.where(dates_t >= oos_start_year_month_int)[0][0]
    }`}
          />

          <div className="mt-6 bg-bg-secondary border border-bg-tertiary rounded-xl p-6">
            <h3 className="text-xl font-semibold text-text-primary mb-4">Data File Structure</h3>
            <p className="text-text-secondary mb-4">
              The raw data comes from <code className="text-accent-green bg-code-bg px-2 py-1 rounded">data/ml_equity_premium_data.xlsx</code> with two sheets:
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4">
                <h4 className="font-semibold text-accent-blue mb-3">'result_predictor' sheet:</h4>
                <div className="space-y-2 text-sm text-text-secondary">
                  <p>• Contains the 32 predictor variables</p>
                  <p>• Contains <code className="text-accent-green">log_equity_premium</code> (our target)</p>
                  <p>• Monthly data with 'month' column in YYYYMM format</p>
                </div>
              </div>
              
              <div className="bg-bg-primary border border-bg-tertiary rounded-lg p-4">
                <h4 className="font-semibold text-accent-purple mb-3">'PredictorData1926-2023' sheet:</h4>
                <div className="space-y-2 text-sm text-text-secondary">
                  <p>• Contains <code className="text-accent-green">CRSP_SPvw</code> (market returns)</p>
                  <p>• Contains <code className="text-accent-green">Rfree</code> (risk-free rates)</p>
                  <p>• Used to construct equity premiums and verify calculations</p>
                </div>
              </div>
            </div>
          </div>
        </motion.section>

        {/* Mathematical Notation */}
        <motion.section
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.8 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-text-primary mb-8 flex items-center">
            <Calculator className="w-8 h-8 text-accent-orange mr-3" />
            Mathematical Notation Setup
          </h2>
          
          <div className="bg-bg-secondary border border-bg-tertiary rounded-xl p-8">
            <p className="text-text-secondary mb-6">
              Throughout this documentation, consistent notation is employed:
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
              <div className="space-y-4">
                <div className="flex items-start space-x-3">
                  <MathFormula latex="t" />
                  <span className="text-text-secondary">denotes the current time period</span>
                </div>
                <div className="flex items-start space-x-3">
                  <MathFormula latex="X_t" />
                  <span className="text-text-secondary">= 32-dimensional predictor vector at time t</span>
                </div>
                <div className="flex items-start space-x-3">
                  <MathFormula latex="y_{t+1}" />
                  <span className="text-text-secondary">= log equity premium at time t+1</span>
                </div>
              </div>
              <div className="space-y-4">
                <div className="flex items-start space-x-3">
                  <MathFormula latex="\hat{y}_{t+1}" />
                  <span className="text-text-secondary">= neural network prediction for <MathFormula latex="y_{t+1}" /></span>
                </div>
                <div className="flex items-start space-x-3">
                  <MathFormula latex="f(X_t; \theta)" />
                  <span className="text-text-secondary">= neural network function with parameters <MathFormula latex="\theta" /></span>
                </div>
                <div className="flex items-start space-x-3">
                  <MathFormula latex="\{(X_t, y_{t+1})\}_{t=1}^{T-1}" />
                  <span className="text-text-secondary">= training data available at time T (to predict period T+1)</span>
                </div>
              </div>
            </div>

            <div className="bg-accent-orange/5 border border-accent-orange/20 rounded-lg p-6">
              <h3 className="text-xl font-semibold text-accent-orange mb-4">The Core Prediction Equation</h3>
              <div className="text-center mb-4">
                <MathFormula latex="\hat{y}_{t+1} = f(X_t; \theta)" block={true} />
              </div>
              <p className="text-text-secondary text-center">
                This equation represents the fundamental challenge: using current financial conditions (<MathFormula latex="X_t" />) 
                to predict next period's excess market return (<MathFormula latex="y_{t+1}" />).
              </p>
            </div>
          </div>
        </motion.section>

        {/* Navigation */}
        <NavigationButtons 
          prevHref="/"
          prevLabel="Introduction"
          nextHref="/preprocessing"
          nextLabel="Data Preprocessing"
        />
      </div>
    </div>
  )
}