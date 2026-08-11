import React, { useMemo } from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceLine,
} from 'recharts';
import { QuantizationMetrics } from '../lib/quantizerEngine';

interface ErrorChartProps {
  originalData: number[];
  metrics: QuantizationMetrics;
  sampleRange: number;
}

export const ErrorChart: React.FC<ErrorChartProps> = ({
  originalData,
  metrics,
  sampleRange,
}) => {
  const chartData = useMemo(() => {
    const subsetLength = Math.min(originalData.length, sampleRange);
    const data = [];
    for (let i = 0; i < subsetLength; i++) {
      const err = originalData[i] - metrics.reconstructedData[i];
      data.push({
        index: i,
        ResidualError: parseFloat(err.toFixed(5)),
        AbsError: parseFloat(Math.abs(err).toFixed(5)),
      });
    }
    return data;
  }, [originalData, metrics, sampleRange]);

  return (
    <div className="bg-[#ffffff] p-5 border-2 border-[#1a1a1a]">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm font-syne uppercase tracking-wider text-[#1a1a1a]">
            Residual Error Noise (x - x̂)
          </h3>
          <p className="text-xs text-[#1a1a1a]/60 font-mono">
            Difference between original float and reconstructed value per element
          </p>
        </div>
        <div className="text-[10px] font-mono font-bold bg-[#1a1a1a] text-[#f2efeb] px-2.5 py-1 uppercase tracking-wider">
          Max: {metrics.maxError.toFixed(5)}
        </div>
      </div>

      <div className="h-64 w-full font-mono">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(26, 26, 26, 0.1)" />
            <XAxis dataKey="index" tick={{ fontSize: 10, fill: '#1a1a1a' }} />
            <YAxis tick={{ fontSize: 10, fill: '#1a1a1a' }} />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1a1a1a',
                borderColor: '#1a1a1a',
                borderRadius: '0px',
                color: '#f2efeb',
                fontSize: '11px',
                fontFamily: 'Space Mono',
              }}
            />
            <ReferenceLine y={0} stroke="rgba(26, 26, 26, 0.3)" strokeDasharray="3 3" />
            <Bar dataKey="ResidualError" fill="#dc2626" opacity={0.85} name="Error (x - x_hat)" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
