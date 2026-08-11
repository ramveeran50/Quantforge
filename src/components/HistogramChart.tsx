import React, { useMemo } from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from 'recharts';
import { QuantizationMetrics } from '../lib/quantizerEngine';

interface HistogramChartProps {
  originalData: number[];
  metrics: QuantizationMetrics;
}

export const HistogramChart: React.FC<HistogramChartProps> = ({
  originalData,
  metrics,
}) => {
  const histogramData = useMemo(() => {
    // Count occurrences of discrete quantized bins
    const counts = new Map<number, number>();
    for (const q of metrics.quantizedData) {
      counts.set(q, (counts.get(q) || 0) + 1);
    }

    const data = [];
    const keys = Array.from(counts.keys()).sort((a, b) => a - b);

    for (const key of keys) {
      const recVal = (key - metrics.zeroPoint) * metrics.scale;
      data.push({
        binIndex: key,
        binLabel: `Level ${key}`,
        reconstructedVal: parseFloat(recVal.toFixed(3)),
        count: counts.get(key) || 0,
      });
    }

    return data;
  }, [metrics]);

  return (
    <div className="bg-[#ffffff] p-5 border-2 border-[#1a1a1a]">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm font-syne uppercase tracking-wider text-[#1a1a1a]">
            Quantized Bin Distribution ({metrics.bits}-bit)
          </h3>
          <p className="text-xs text-[#1a1a1a]/60 font-mono">
            Element frequency mapped to each integer bin [0, {metrics.qMax}]
          </p>
        </div>
        <div className="text-[10px] font-mono font-bold bg-[#f2efeb] text-[#1a1a1a] border border-[#1a1a1a] px-2.5 py-1">
          {histogramData.length} bins
        </div>
      </div>

      <div className="h-64 w-full font-mono">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={histogramData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(26, 26, 26, 0.1)" />
            <XAxis dataKey="binLabel" tick={{ fontSize: 10, fill: '#1a1a1a' }} />
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
              formatter={(value: any, name: any, item: any) => [
                `${value} items (~${item.payload.reconstructedVal})`,
                'Count',
              ]}
            />
            <Bar dataKey="count" fill="#4f46e5" opacity={0.9} name="Count" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
