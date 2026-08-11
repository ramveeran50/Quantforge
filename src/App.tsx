import React, { useState, useMemo } from 'react';
import { Header } from './components/Header';
import { MetricsOverview } from './components/MetricsOverview';
import { SignalChart } from './components/SignalChart';
import { ErrorChart } from './components/ErrorChart';
import { RateDistortionCurve } from './components/RateDistortionCurve';
import { HistogramChart } from './components/HistogramChart';
import { InspectorTable } from './components/InspectorTable';
import { TheoryGuide } from './components/TheoryGuide';
import { PythonRunnerModal } from './components/PythonRunnerModal';
import { ModelCompressionDashboard } from './components/ModelCompressionDashboard';
import {
  quantizeAndEvaluate,
  generateSyntheticDataset,
} from './lib/quantizerEngine';
import {
  Sliders,
  RefreshCw,
  Cpu,
  BarChart2,
  Table,
  LineChart as LineChartIcon,
} from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState<'demo' | 'model' | 'theory' | 'code'>('model');

  // Interactive Lab Controls State
  const [datasetType, setDatasetType] = useState<'neural' | 'sine' | 'sparse' | 'uniform'>('neural');
  const [sampleCount, setSampleCount] = useState<number>(500);
  const [bitWidth, setBitWidth] = useState<number>(4);
  const [sampleRange, setSampleRange] = useState<number>(80);
  const [seed, setSeed] = useState<number>(42);

  // Generate dataset based on controls
  const originalData = useMemo(() => {
    return generateSyntheticDataset(datasetType, sampleCount, seed);
  }, [datasetType, sampleCount, seed]);

  // Compute quantization metrics
  const metrics = useMemo(() => {
    return quantizeAndEvaluate(originalData, bitWidth);
  }, [originalData, bitWidth]);

  return (
    <div className="min-h-screen bg-[#f2efeb] text-[#1a1a1a] font-sans antialiased pb-12">
      {/* Top Header */}
      <Header activeTab={activeTab} setActiveTab={setActiveTab} />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-6">
        {activeTab === 'demo' && (
          <div className="space-y-6">
            
            {/* Control Panel Bar */}
            <div className="terminal-card bg-[#f2efeb] p-4 sm:p-5 border-2 border-[#1a1a1a]">
              <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
                
                {/* Left: Section Title */}
                <div className="flex items-center space-x-2.5">
                  <div className="p-2 bg-[#1a1a1a] text-[#f2efeb]">
                    <Sliders className="w-4 h-4" />
                  </div>
                  <div>
                    <h2 className="text-sm font-syne uppercase tracking-wider text-[#1a1a1a]">Quantization Controls</h2>
                    <p className="text-xs text-[#1a1a1a]/60 font-mono">Configure numerical precision, distribution, and sample size</p>
                  </div>
                </div>

                {/* Right: Interactive Controls */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono">
                  
                  {/* Bit Width Selector */}
                  <div>
                    <label className="block text-[10px] uppercase tracking-wider font-bold text-[#1a1a1a]/70 mb-1">
                      Precision: <span className="text-[#4f46e5] font-bold">{bitWidth}-BIT</span>
                    </label>
                    <div className="flex items-center space-x-1 bg-[#ffffff] p-1 border border-[#1a1a1a] text-xs">
                      {[16, 8, 4, 2].map((b) => (
                        <button
                          key={b}
                          onClick={() => setBitWidth(b)}
                          className={`flex-1 py-1 font-bold text-xs uppercase tracking-wider transition-all ${
                            bitWidth === b
                              ? 'bg-[#1a1a1a] text-[#f2efeb]'
                              : 'text-[#1a1a1a]/70 hover:text-[#1a1a1a]'
                          }`}
                        >
                          {b}B
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Dataset Type Selector */}
                  <div>
                    <label className="block text-[10px] uppercase tracking-wider font-bold text-[#1a1a1a]/70 mb-1">
                      Distribution
                    </label>
                    <select
                      value={datasetType}
                      onChange={(e) => setDatasetType(e.target.value as any)}
                      className="w-full px-2.5 py-1.5 text-xs font-mono border border-[#1a1a1a] bg-[#ffffff] text-[#1a1a1a] focus:outline-none focus:border-[#4f46e5]"
                    >
                      <option value="neural">LLM Weights (Gaussian)</option>
                      <option value="sine">Sine Wave Signal</option>
                      <option value="sparse">Sparse Outliers</option>
                      <option value="uniform">Uniform Random</option>
                    </select>
                  </div>

                  {/* Sample Count */}
                  <div>
                    <label className="block text-[10px] uppercase tracking-wider font-bold text-[#1a1a1a]/70 mb-1">
                      Sample Size
                    </label>
                    <select
                      value={sampleCount}
                      onChange={(e) => setSampleCount(Number(e.target.value))}
                      className="w-full px-2.5 py-1.5 text-xs font-mono border border-[#1a1a1a] bg-[#ffffff] text-[#1a1a1a] focus:outline-none focus:border-[#4f46e5]"
                    >
                      <option value={200}>200 Samples</option>
                      <option value={500}>500 Samples</option>
                      <option value={1000}>1,000 Samples</option>
                      <option value={10000}>10,000 Samples</option>
                    </select>
                  </div>

                  {/* Regenerate Seed */}
                  <div className="flex items-end">
                    <button
                      onClick={() => setSeed((prev) => prev + 1)}
                      className="w-full py-1.5 px-3 text-xs font-bold uppercase tracking-wider bg-[#1a1a1a] hover:bg-[#4f46e5] text-[#f2efeb] border border-[#1a1a1a] transition-colors flex items-center justify-center"
                    >
                      <RefreshCw className="w-3.5 h-3.5 mr-1.5" />
                      Re-seed
                    </button>
                  </div>

                </div>

              </div>
            </div>

            {/* Metrics Stat Cards */}
            <MetricsOverview metrics={metrics} />

            {/* Charts Grid Row 1 */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <SignalChart
                originalData={originalData}
                metrics={metrics}
                sampleRange={sampleRange}
              />
              <ErrorChart
                originalData={originalData}
                metrics={metrics}
                sampleRange={sampleRange}
              />
            </div>

            {/* Charts Grid Row 2 */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <RateDistortionCurve
                originalData={originalData}
                currentBits={bitWidth}
              />
              <HistogramChart
                originalData={originalData}
                metrics={metrics}
              />
            </div>

            {/* Element-Level Transformation Inspector */}
            <InspectorTable
              originalData={originalData}
              metrics={metrics}
            />

          </div>
        )}

        {activeTab === 'model' && <ModelCompressionDashboard />}

        {activeTab === 'theory' && <TheoryGuide />}

        {activeTab === 'code' && <PythonRunnerModal />}
      </main>
    </div>
  );
}
