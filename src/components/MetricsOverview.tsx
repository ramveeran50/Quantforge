import React from 'react';
import { QuantizationMetrics } from '../lib/quantizerEngine';
import { HardDrive, ArrowDownRight, Activity, Percent, Scale, Crosshair } from 'lucide-react';

interface MetricsOverviewProps {
  metrics: QuantizationMetrics;
}

export const MetricsOverview: React.FC<MetricsOverviewProps> = ({ metrics }) => {
  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 mb-6 font-mono">
      
      {/* Card 1: Actual Storage Ratio */}
      <div className="bg-[#ffffff] p-4 border-2 border-[#1a1a1a] transition-colors">
        <div className="flex items-center justify-between text-[#1a1a1a]/70 mb-1">
          <span className="text-[10px] uppercase tracking-wider font-bold text-[#4f46e5]">Actual Ratio</span>
          <HardDrive className="w-4 h-4 text-[#4f46e5]" />
        </div>
        <div className="text-xl font-bold text-[#1a1a1a]">{metrics.actualCompressionRatio.toFixed(2)}x</div>
        <div className="text-[10px] text-[#1a1a1a]/60 mt-1 flex justify-between">
          <span>Theo: <strong className="text-[#1a1a1a]">{metrics.theoreticalCompressionRatio.toFixed(1)}x</strong></span>
        </div>
      </div>

      {/* Card 2: Memory Saved */}
      <div className="bg-[#ffffff] p-4 border-2 border-[#1a1a1a] transition-colors">
        <div className="flex items-center justify-between text-[#1a1a1a]/70 mb-1">
          <span className="text-[10px] uppercase tracking-wider font-bold">Storage Saved</span>
          <Percent className="w-4 h-4 text-emerald-600" />
        </div>
        <div className="text-xl font-bold text-emerald-700">{metrics.actualStorageSavedPercent.toFixed(1)}%</div>
        <div className="text-[10px] text-[#1a1a1a]/60 mt-1 truncate">
          {metrics.totalCompressedSizeBytes.toLocaleString()}B / {metrics.originalSizeBytes.toLocaleString()}B
        </div>
      </div>

      {/* Card 3: Mean Squared Error (MSE) */}
      <div className="bg-[#ffffff] p-4 border-2 border-[#1a1a1a] transition-colors">
        <div className="flex items-center justify-between text-[#1a1a1a]/70 mb-1">
          <span className="text-[10px] uppercase tracking-wider font-bold">MSE Error</span>
          <Activity className="w-4 h-4 text-amber-600" />
        </div>
        <div className="text-xl font-bold text-[#1a1a1a]">
          {metrics.mse < 0.0001 ? metrics.mse.toExponential(3) : metrics.mse.toFixed(5)}
        </div>
        <div className="text-[10px] text-[#1a1a1a]/60 mt-1">
          Mean squared error
        </div>
      </div>

      {/* Card 4: Mean Absolute Error (MAE) */}
      <div className="bg-[#ffffff] p-4 border-2 border-[#1a1a1a] transition-colors">
        <div className="flex items-center justify-between text-[#1a1a1a]/70 mb-1">
          <span className="text-[10px] uppercase tracking-wider font-bold">MAE Error</span>
          <ArrowDownRight className="w-4 h-4 text-blue-600" />
        </div>
        <div className="text-xl font-bold text-[#1a1a1a]">
          {metrics.mae < 0.0001 ? metrics.mae.toExponential(3) : metrics.mae.toFixed(5)}
        </div>
        <div className="text-[10px] text-[#1a1a1a]/60 mt-1">
          Mean absolute error
        </div>
      </div>

      {/* Card 5: Scale S */}
      <div className="bg-[#ffffff] p-4 border-2 border-[#1a1a1a] transition-colors">
        <div className="flex items-center justify-between text-[#1a1a1a]/70 mb-1">
          <span className="text-[10px] uppercase tracking-wider font-bold">Scale (S)</span>
          <Scale className="w-4 h-4 text-purple-600" />
        </div>
        <div className="text-xl font-bold text-[#1a1a1a]">
          {metrics.scale.toFixed(5)}
        </div>
        <div className="text-[10px] text-[#1a1a1a]/60 mt-1">
          Float step per level
        </div>
      </div>

      {/* Card 6: Zero-Point Z */}
      <div className="bg-[#ffffff] p-4 border-2 border-[#1a1a1a] transition-colors">
        <div className="flex items-center justify-between text-[#1a1a1a]/70 mb-1">
          <span className="text-[10px] uppercase tracking-wider font-bold">Zero-Point</span>
          <Crosshair className="w-4 h-4 text-rose-600" />
        </div>
        <div className="text-xl font-bold text-[#1a1a1a]">
          {metrics.zeroPoint}
        </div>
        <div className="text-[10px] text-[#1a1a1a]/60 mt-1">
          Integer level for 0.0
        </div>
      </div>

    </div>
  );
};
