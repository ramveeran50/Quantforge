import React, { useMemo } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Dot,
} from 'recharts';
import { quantizeAndEvaluate } from '../lib/quantizerEngine';

interface RateDistortionCurveProps {
  originalData: number[];
  currentBits: number;
}

export const RateDistortionCurve: React.FC<RateDistortionCurveProps> = ({
  originalData,
  currentBits,
}) => {
  const curveData = useMemo(() => {
    const bitLevels = [2, 4, 8, 16, 32];
    return bitLevels.map((bits) => {
      if (bits === 32) {
        return {
          bits: '32-bit (FP32)',
          bitNum: 32,
          MSE: 0,
          MAE: 0,
          ActualRatio: 1.0,
          TheoreticalRatio: 1.0,
          Saved: '0%',
        };
      }
      const res = quantizeAndEvaluate(originalData, bits);
      return {
        bits: `${bits}-bit`,
        bitNum: bits,
        MSE: parseFloat(res.mse.toFixed(6)),
        MAE: parseFloat(res.mae.toFixed(6)),
        ActualRatio: parseFloat(res.actualCompressionRatio.toFixed(2)),
        TheoreticalRatio: parseFloat(res.theoreticalCompressionRatio.toFixed(1)),
        Saved: `${res.actualStorageSavedPercent.toFixed(1)}%`,
      };
    });
  }, [originalData]);

  return (
    <div className="bg-[#ffffff] p-5 border-2 border-[#1a1a1a]">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm font-syne uppercase tracking-wider text-[#1a1a1a]">
            Rate-Distortion Curve: Bit-Width vs MSE
          </h3>
          <p className="text-xs text-[#1a1a1a]/60 font-mono">
            Trade-off relationship between bit compression and information loss
          </p>
        </div>
      </div>

      <div className="h-64 w-full font-mono">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={curveData} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(26, 26, 26, 0.1)" />
            <XAxis dataKey="bits" tick={{ fontSize: 10, fill: '#1a1a1a' }} />
            <YAxis
              tick={{ fontSize: 10, fill: '#1a1a1a' }}
              label={{ value: 'MSE Error', angle: -90, position: 'insideLeft', style: { fontSize: 10, fill: '#1a1a1a' } }}
            />
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
                `${value} (Actual: ${item.payload.ActualRatio}x, Theo: ${item.payload.TheoreticalRatio}x)`,
                'MSE',
              ]}
            />
            <Line
              type="monotone"
              dataKey="MSE"
              stroke="#4f46e5"
              strokeWidth={2.5}
              dot={(props: any) => {
                const isCurrent = props.payload.bitNum === currentBits;
                return (
                  <circle
                    cx={props.cx}
                    cy={props.cy}
                    r={isCurrent ? 6 : 4}
                    fill={isCurrent ? '#1a1a1a' : '#4f46e5'}
                    stroke={isCurrent ? '#4f46e5' : 'none'}
                    strokeWidth={2}
                    key={props.index}
                  />
                );
              }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-5 gap-2 mt-4 pt-3 border-t border-[#1a1a1a]/20 text-center font-mono">
        {curveData.map((item) => (
          <div
            key={item.bits}
            className={`p-2 transition-colors border-2 ${
              item.bitNum === currentBits
                ? 'bg-[#1a1a1a] border-[#1a1a1a] text-[#f2efeb]'
                : 'bg-[#f2efeb] border-[#1a1a1a]/30 text-[#1a1a1a]'
            }`}
          >
            <div className="text-[11px] font-bold uppercase">{item.bits}</div>
            <div className="text-[10px] opacity-80 mt-0.5">MSE: {item.MSE}</div>
            <div className="text-[9px] font-bold tracking-tight">
              Actual: {item.ActualRatio}x
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
