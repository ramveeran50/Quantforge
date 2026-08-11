import React, { useMemo } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from 'recharts';
import { QuantizationMetrics } from '../lib/quantizerEngine';

interface SignalChartProps {
  originalData: number[];
  metrics: QuantizationMetrics;
  sampleRange: number;
}

export const SignalChart: React.FC<SignalChartProps> = ({
  originalData,
  metrics,
  sampleRange,
}) => {
  const chartData = useMemo(() => {
    const subsetLength = Math.min(originalData.length, sampleRange);
    const data = [];
    for (let i = 0; i < subsetLength; i++) {
      data.push({
        index: i,
        Original: parseFloat(originalData[i].toFixed(4)),
        Reconstructed: parseFloat(metrics.reconstructedData[i].toFixed(4)),
        QuantizedBin: metrics.quantizedData[i],
      });
    }
    return data;
  }, [originalData, metrics, sampleRange]);

  return (
    <div className="bg-[#ffffff] p-5 border-2 border-[#1a1a1a]">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm font-syne uppercase tracking-wider text-[#1a1a1a]">
            Original FP32 vs Reconstructed ({metrics.bits}-bit)
          </h3>
          <p className="text-xs text-[#1a1a1a]/60 font-mono">
            Comparing true continuous FP32 values against reconstructed values after quantization
          </p>
        </div>
        <div className="text-[10px] font-mono font-bold bg-[#f2efeb] text-[#1a1a1a] border border-[#1a1a1a] px-2.5 py-1">
          {sampleRange} samples
        </div>
      </div>

      <div className="h-72 w-full font-mono">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(26, 26, 26, 0.1)" />
            <XAxis dataKey="index" tick={{ fontSize: 10, fill: '#1a1a1a' }} />
            <YAxis tick={{ fontSize: 10, fill: '#1a1a1a' }} domain={['auto', 'auto']} />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1a1a1a',
                borderColor: '#1a1a1a',
                borderRadius: '0px',
                color: '#f2efeb',
                fontSize: '11px',
                fontFamily: 'Space Mono',
              }}
              formatter={(value: any, name: any) => [value, name]}
            />
            <Legend wrapperStyle={{ paddingTop: '10px', fontSize: '11px', color: '#1a1a1a' }} />
            <Line
              type="monotone"
              dataKey="Original"
              stroke="#2563eb"
              strokeWidth={2}
              dot={false}
              name="Original FP32"
            />
            <Line
              type="stepAfter"
              dataKey="Reconstructed"
              stroke="#4f46e5"
              strokeWidth={2}
              strokeDasharray="4 2"
              dot={false}
              name={`Reconstructed (${metrics.bits}-bit)`}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
