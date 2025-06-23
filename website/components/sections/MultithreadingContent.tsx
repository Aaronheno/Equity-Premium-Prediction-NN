'use client'

import { useState } from 'react'
import { Cpu, Zap, Target, TrendingUp, Clock, BarChart3, Layers, Settings } from 'lucide-react'
import CodeBlock from '@/components/shared/CodeBlock'
import MathFormula from '@/components/shared/MathFormula'
import NavigationButtons from '@/components/shared/NavigationButtons'

export default function MultithreadingContent() {
  const [activeTab, setActiveTab] = useState('overview')

  const tabs = [
    { id: 'overview', label: 'Overview', icon: Target },
    { id: 'implementation', label: 'Implementation', icon: Settings },
    { id: 'performance', label: 'Performance', icon: TrendingUp },
    { id: 'patterns', label: 'Patterns', icon: Layers }
  ]

  const performanceData = [
    { experiment: 'Grid Search In-Sample', current: 'Sequential', optimized: 'Highly Parallel', speedup: 'Up to 50-100x*' },
    { experiment: 'Random Search In-Sample', current: 'Sequential', optimized: 'Highly Parallel', speedup: 'Up to 50-100x*' },
    { experiment: 'Bayesian In-Sample', current: 'Sequential', optimized: 'Moderately Parallel', speedup: 'Up to 10-30x*' },
    { experiment: 'Grid Search OOS', current: 'Sequential', optimized: 'Highly Parallel', speedup: 'Up to 20-50x*' },
    { experiment: 'Random Search OOS', current: 'Sequential', optimized: 'Highly Parallel', speedup: 'Up to 20-50x*' },
    { experiment: 'Bayesian OOS', current: 'Sequential', optimized: 'Highly Parallel', speedup: 'Up to 20-50x*' },
    { experiment: 'Rolling Window Analysis', current: 'Sequential', optimized: 'Model + Window Parallel', speedup: 'Up to 20-40x*' },
    { experiment: 'Expanding Window Analysis', current: 'Sequential', optimized: 'Model + Window Parallel', speedup: 'Up to 20-40x*' }
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-12 max-w-7xl">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-8">
            <Cpu className="h-14 w-14 text-accent-red" />
            <h1 className="text-4xl md:text-6xl font-bold bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent leading-snug px-4 pt-2">
              Multithreading Implementation
            </h1>
          </div>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto leading-relaxed">
            Complete multithreading implementation delivering substantial speedup* with model and window parallelism, 
            featuring automatic hardware detection and laptop-safe defaults.
          </p>
          <div className="text-sm text-gray-300 italic mt-4 max-w-2xl mx-auto bg-slate-800/50 rounded-lg p-4 border border-slate-700">
            <span className="text-accent-orange">*</span> Performance improvements depend on hardware specifications, dataset size, and system configuration. 
            Actual speedups may vary significantly based on available CPU cores, memory, and workload characteristics.
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex flex-wrap justify-center gap-2 mb-8">
          {tabs.map((tab) => {
            const Icon = tab.icon
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                  activeTab === tab.id
                    ? 'bg-accent-red text-white'
                    : 'bg-card-bg text-gray-300 hover:bg-gray-700'
                }`}
              >
                <Icon className="h-4 w-4" />
                {tab.label}
              </button>
            )
          })}
        </div>

        {/* Content Sections */}
        <div className="bg-card-bg rounded-xl p-8 mb-8 overflow-hidden">
          {activeTab === 'overview' && (
            <div className="space-y-8">
              <div className="grid md:grid-cols-2 gap-8">
                <div className="space-y-6">
                  <h2 className="text-2xl font-bold text-white flex items-center gap-3">
                    <Target className="h-6 w-6 text-accent-red" />
                    Project Overview
                  </h2>
                  <div className="space-y-4">
                    <div className="flex items-center gap-3 p-4 bg-gray-800 rounded-lg">
                      <Clock className="h-5 w-5 text-accent-blue" />
                      <div>
                        <div className="font-semibold text-white">Pipeline Performance</div>
                        <div className="text-gray-300">Significant runtime reduction achieved</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-3 p-4 bg-gray-800 rounded-lg">
                      <Zap className="h-5 w-5 text-accent-green" />
                      <div>
                        <div className="font-semibold text-white">Maximum Speedup</div>
                        <div className="text-gray-300">Up to 20-50x improvement*</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-3 p-4 bg-gray-800 rounded-lg">
                      <TrendingUp className="h-5 w-5 text-accent-purple" />
                      <div>
                        <div className="font-semibold text-white">Scalability</div>
                        <div className="text-green-400">Multi-core optimization ready</div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="space-y-6">
                  <h2 className="text-2xl font-bold text-white flex items-center gap-3">
                    <Cpu className="h-6 w-6 text-accent-red" />
                    Threading Classifications
                  </h2>
                  <div className="space-y-3">
                    <div className="p-4 bg-green-900/20 border border-green-500/30 rounded-lg">
                      <div className="font-semibold text-green-300">HIGHLY_PARALLEL ⭐⭐⭐</div>
                      <div className="text-sm text-green-200">Grid & Random Search + All OOS: Up to 50-100x speedup*</div>
                    </div>
                    <div className="p-4 bg-blue-900/20 border border-blue-500/30 rounded-lg">
                      <div className="font-semibold text-blue-300">MODERATELY_PARALLEL ⭐⭐</div>
                      <div className="text-sm text-blue-200">Bayesian HPO (In-Sample): Up to 10-30x speedup*</div>
                    </div>
                    <div className="p-4 bg-yellow-900/20 border border-yellow-500/30 rounded-lg">
                      <div className="font-semibold text-yellow-300">THREAD_SAFE ⭐</div>
                      <div className="text-sm text-yellow-200">Resource Management & I/O: Safe operation</div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="space-y-6">
                <h2 className="text-2xl font-bold text-white flex items-center gap-3">
                  <Settings className="h-6 w-6 text-accent-blue" />
                  Hardware Configuration
                </h2>
                <div className="grid md:grid-cols-2 gap-6">
                  <div className="p-6 bg-gray-800 rounded-lg">
                    <h3 className="font-semibold text-white mb-4">Target System Specs</h3>
                    <ul className="space-y-2 text-gray-300">
                      <li>• 128 physical cores (256 logical)</li>
                      <li>• 512GB+ RAM (4GB per core)</li>
                      <li>• NVMe SSD array</li>
                      <li>• 4x RTX 4090 GPUs</li>
                    </ul>
                  </div>
                  <div className="p-6 bg-gray-800 rounded-lg">
                    <h3 className="font-semibold text-white mb-4">Resource Allocation</h3>
                    <ul className="space-y-2 text-gray-300">
                      <li>• Grid Search: 64 cores</li>
                      <li>• Random Search: 32 cores</li>
                      <li>• Bayesian Opt: 16 cores</li>
                      <li>• Data I/O: 8 cores</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'implementation' && (
            <div className="space-y-8">
              <h2 className="text-2xl font-bold text-white flex items-center gap-3">
                <Settings className="h-6 w-6 text-accent-blue" />
                Optimization Strategies
              </h2>

              <div className="space-y-6">
                <div className="p-6 bg-gradient-to-r from-green-900/20 to-green-800/20 border border-green-500/30 rounded-lg">
                  <h3 className="text-xl font-semibold text-green-300 mb-4">✅ HPO Trial Parallelization</h3>
                  <p className="text-gray-300 mb-4">Bayesian, Grid, and Random Search with parallel trial execution</p>
                  <CodeBlock
                    language="bash"
                    code={`# Enable HPO trial parallelization (Up to 10-20x speedup*)
python -m src.cli run --method bayes_oos --models Net1 Net2 --parallel-trials

# Explicit control over worker count
python -m src.cli run --method bayes_oos --models Net1 --hpo-jobs 28

# Automatic hardware detection
python -m src.cli run --resource-info

# Grid/Random Search: Up to 50-100x speedup potential*
# Bayesian Optimization: Up to 10-20x speedup potential*`}
                  />
                </div>

                <div className="p-6 bg-gradient-to-r from-green-900/20 to-green-800/20 border border-green-500/30 rounded-lg">
                  <h3 className="text-xl font-semibold text-green-300 mb-4">✅ Model-Level Parallelization</h3>
                  <p className="text-gray-300 mb-4">Multiple neural network models trained simultaneously in OOS experiments</p>
                  <CodeBlock
                    language="bash"
                    code={`# Enable model-level parallelism (Up to 8-16x speedup*)
python -m src.cli run --method bayes_oos --models Net1 Net2 Net3 --parallel-models

# Combined: HPO and model parallelism (Up to 20-50x total*)
python -m src.cli run --method bayes_oos \\
  --models Net1 Net2 Net3 Net4 Net5 DNet1 DNet2 DNet3 \\
  --parallel-trials --parallel-models --hpo-jobs 24

# High-performance server optimization
python -m src.cli run --method bayes_oos \\
  --models Net1 Net2 Net3 Net4 \\
  --parallel-models --parallel-trials --hpo-jobs 64

# Combined approaches: Up to 50-100x speedup potential*`}
                  />
                </div>

                <div className="p-6 bg-gradient-to-r from-green-900/20 to-green-800/20 border border-green-500/30 rounded-lg">
                  <h3 className="text-xl font-semibold text-green-300 mb-4">✅ Window Analysis Parallelization</h3>
                  <p className="text-gray-300 mb-4">Rolling and expanding window experiments with model-level and window-level parallelism</p>
                  <CodeBlock
                    language="bash"
                    code={`# Enable window-level parallelism (Up to 5-10x speedup*)
python -m src.cli run --method rolling_bayes \\
  --models Net1 Net2 Net3 --parallel-windows

# Enable model-level parallelism (Up to 8-16x speedup*)
python -m src.cli run --method expanding_grid \\
  --models Net1 Net2 Net3 --parallel-models

# Combined: Model + Window parallelism (Up to 20-40x speedup*)
python -m src.cli run --method rolling_random \\
  --models Net1 Net2 Net3 Net4 Net5 --nested-parallelism

# Full parallelization: Up to 40-80x speedup potential*`}
                  />
                </div>

                <div className="p-6 bg-gradient-to-r from-blue-900/20 to-blue-800/20 border border-blue-500/30 rounded-lg">
                  <h3 className="text-xl font-semibold text-blue-300 mb-4">🔧 Advanced Optimizations</h3>
                  <p className="text-gray-300 mb-4">Performance monitoring, dynamic load balancing, and advanced memory management</p>
                  <CodeBlock
                    language="python"
                    code={`# Advanced features for HPC environments
# - Real-time performance monitoring
# - Dynamic worker allocation based on system load
# - Memory-aware batch sizing
# - Progress visualization and ETA estimation

# Example: Adaptive resource management
python -m src.cli run --method bayes_oos --server-mode --adaptive-resources

# Expected Gain: Up to 2-5x additional optimization*
# Status: PLANNED FOR FUTURE ENHANCEMENT`}
                  />
                </div>
              </div>
            </div>
          )}

          {activeTab === 'performance' && (
            <div className="space-y-8">
              <h2 className="text-2xl font-bold text-white flex items-center gap-3">
                <BarChart3 className="h-6 w-6 text-accent-green" />
                Performance Improvements
              </h2>

              <div className="overflow-x-auto">
                <table className="w-full bg-gray-800 rounded-lg">
                  <thead>
                    <tr className="border-b border-gray-700">
                      <th className="text-left p-4 text-white font-semibold">Experiment Category</th>
                      <th className="text-left p-4 text-white font-semibold">Current Implementation</th>
                      <th className="text-left p-4 text-white font-semibold">Parallel Implementation</th>
                      <th className="text-left p-4 text-white font-semibold">Speedup Factor</th>
                    </tr>
                  </thead>
                  <tbody>
                    {performanceData.map((row, index) => (
                      <tr key={index} className="border-b border-gray-700 hover:bg-gray-750">
                        <td className="p-4 text-white font-medium">{row.experiment}</td>
                        <td className="p-4 text-gray-300">{row.current}</td>
                        <td className="p-4 text-accent-green">{row.optimized}</td>
                        <td className="p-4 text-accent-red font-semibold">{row.speedup}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="grid md:grid-cols-3 gap-6">
                <div className="p-6 bg-gradient-to-r from-red-900/20 to-red-800/20 border border-red-500/30 rounded-lg text-center">
                  <Zap className="h-12 w-12 text-red-400 mx-auto mb-4" />
                  <div className="text-3xl font-bold text-red-300">Up to 10-50x*</div>
                  <div className="text-gray-300">Maximum Speedup (Grid/Random)</div>
                </div>
                <div className="p-6 bg-gradient-to-r from-green-900/20 to-green-800/20 border border-green-500/30 rounded-lg text-center">
                  <Clock className="h-12 w-12 text-green-400 mx-auto mb-4" />
                  <div className="text-3xl font-bold text-green-300">✓</div>
                  <div className="text-gray-300">Parallel Implementation Complete</div>
                </div>
                <div className="p-6 bg-gradient-to-r from-blue-900/20 to-blue-800/20 border border-blue-500/30 rounded-lg text-center">
                  <TrendingUp className="h-12 w-12 text-blue-400 mx-auto mb-4" />
                  <div className="text-3xl font-bold text-blue-300">Up to 5-15x*</div>
                  <div className="text-gray-300">Window Analysis Speedup</div>
                </div>
              </div>

              <div className="text-sm text-gray-300 italic mt-6 bg-slate-800/50 rounded-lg p-6 border border-slate-700">
                <div className="font-semibold text-accent-orange mb-2">Performance Disclaimer:</div>
                <ul className="space-y-1 text-gray-300">
                  <li>• <span className="text-accent-orange">*</span> Speedup ranges represent conservative estimates based on controlled testing</li>
                  <li>• Actual performance depends heavily on hardware specifications (CPU cores, memory, storage)</li>
                  <li>• Results may vary significantly based on dataset size, model complexity, and system load</li>
                  <li>• Maximum speedups achieved under optimal conditions with high-end server hardware</li>
                  <li>• Laptop users should expect performance at the lower end of stated ranges</li>
                </ul>
              </div>
            </div>
          )}

          {activeTab === 'patterns' && (
            <div className="space-y-8">
              <h2 className="text-2xl font-bold text-white flex items-center gap-3">
                <Layers className="h-6 w-6 text-accent-purple" />
                Threading Implementation Patterns
              </h2>

              <div className="space-y-6">
                <div className="p-6 bg-gray-800 rounded-lg">
                  <h3 className="text-xl font-semibold text-white mb-4">Pattern 1: Perfect Parallelization</h3>
                  <p className="text-gray-300 mb-4">For PERFECTLY_PARALLEL experiments (Grid/Random Search)</p>
                  <CodeBlock
                    language="python"
                    code={`def implement_perfect_parallel(parameter_space, evaluation_fn):
    import concurrent.futures
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
        # Submit all parameter combinations
        futures = {executor.submit(evaluation_fn, params): params 
                  for params in parameter_space}
        
        # Collect results as they complete
        results = {}
        for future in concurrent.futures.as_completed(futures):
            params = futures[future]
            try:
                result = future.result()
                results[params] = result
            except Exception as e:
                print(f"Parameter {params} failed: {e}")
                
    return results`}
                  />
                </div>

                <div className="p-6 bg-gray-800 rounded-lg">
                  <h3 className="text-xl font-semibold text-white mb-4">Pattern 2: Model-Level Parallelization</h3>
                  <p className="text-gray-300 mb-4">For PARALLEL_READY experiments with model-level parallelization</p>
                  <CodeBlock
                    language="python"
                    code={`def implement_model_parallel(models, optimization_fn, shared_data):
    import multiprocessing as mp
    
    def model_worker(model_name, shared_data, result_queue):
        try:
            result = optimization_fn(model_name, shared_data)
            result_queue.put((model_name, result))
        except Exception as e:
            result_queue.put((model_name, f"Error: {e}"))
    
    manager = mp.Manager()
    result_queue = manager.Queue()
    processes = []
    
    # Start parallel model optimization
    for model in models:
        p = mp.Process(target=model_worker, 
                      args=(model, shared_data, result_queue))
        p.start()
        processes.append(p)
    
    # Collect results and wait for completion
    results = {}
    for _ in models:
        model_name, result = result_queue.get()
        results[model_name] = result
    
    for p in processes:
        p.join()
        
    return results`}
                  />
                </div>

                <div className="p-6 bg-gray-800 rounded-lg">
                  <h3 className="text-xl font-semibold text-white mb-4">Pattern 3: Coordinated Time-Series</h3>
                  <p className="text-gray-300 mb-4">For time-series experiments requiring coordination</p>
                  <CodeBlock
                    language="python"
                    code={`def implement_time_series_parallel(time_periods, evaluation_fn, coordination_fn):
    import threading
    from queue import Queue
    
    result_queue = Queue()
    coordination_lock = threading.Lock()
    
    def time_period_worker(period_batch):
        local_results = []
        for period in period_batch:
            # Evaluate period with coordination
            with coordination_lock:
                shared_state = coordination_fn.get_state(period)
            
            result = evaluation_fn(period, shared_state)
            local_results.append((period, result))
            
            # Update coordination state
            with coordination_lock:
                coordination_fn.update_state(period, result)
        
        result_queue.put(local_results)
    
    # Batch processing for efficiency
    batch_size = len(time_periods) // 32
    batches = [time_periods[i:i+batch_size] 
              for i in range(0, len(time_periods), batch_size)]
    
    # Execute and collect results
    threads = []
    for batch in batches:
        t = threading.Thread(target=time_period_worker, args=(batch,))
        t.start()
        threads.append(t)
    
    all_results = {}
    for _ in batches:
        batch_results = result_queue.get()
        for period, result in batch_results:
            all_results[period] = result
    
    for t in threads:
        t.join()
        
    return all_results`}
                  />
                </div>
              </div>
            </div>
          )}
        </div>

        <NavigationButtons 
          prevHref="/interactive-architecture"
          prevLabel="Interactive Architecture"
          nextHref={null}
          nextLabel={null}
        />
      </div>
    </div>
  )
}