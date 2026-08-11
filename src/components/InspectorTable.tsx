import React, { useState } from 'react';
import { QuantizationMetrics } from '../lib/quantizerEngine';
import { ArrowRight, Search } from 'lucide-react';

interface InspectorTableProps {
  originalData: number[];
  metrics: QuantizationMetrics;
}

export const InspectorTable: React.FC<InspectorTableProps> = ({
  originalData,
  metrics,
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [displayCount, setDisplayCount] = useState(15);

  const rows = originalData.map((orig, i) => {
    const q = metrics.quantizedData[i];
    const rec = metrics.reconstructedData[i];
    const absErr = Math.abs(orig - rec);
    return {
      index: i,
      original: orig,
      quantized: q,
      reconstructed: rec,
      absError: absErr,
    };
  });

  const filteredRows = rows
    .filter(
      (r) =>
        r.index.toString().includes(searchTerm) ||
        r.original.toString().includes(searchTerm) ||
        r.quantized.toString().includes(searchTerm)
    )
    .slice(0, displayCount);

  return (
    <div className="bg-[#ffffff] border-2 border-[#1a1a1a] overflow-hidden">
      <div className="p-4 border-b-2 border-[#1a1a1a] flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-[#f2efeb]">
        <div>
          <h3 className="text-sm font-syne uppercase tracking-wider text-[#1a1a1a]">
            Element-Level Transformation Inspector
          </h3>
          <p className="text-xs text-[#1a1a1a]/60 font-mono">
            Step-by-step trace of FP32 float &rarr; quantized integer &rarr; reconstructed float
          </p>
        </div>

        <div className="flex items-center space-x-2 font-mono">
          <div className="relative">
            <Search className="w-3.5 h-3.5 absolute left-2.5 top-2.5 text-[#1a1a1a]/40" />
            <input
              type="text"
              placeholder="Filter index or value..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-8 pr-3 py-1.5 text-xs border border-[#1a1a1a] bg-[#ffffff] text-[#1a1a1a] focus:outline-none focus:border-[#4f46e5]"
            />
          </div>

          <select
            value={displayCount}
            onChange={(e) => setDisplayCount(Number(e.target.value))}
            className="px-2.5 py-1.5 text-xs border border-[#1a1a1a] bg-[#ffffff] text-[#1a1a1a] focus:outline-none focus:border-[#4f46e5]"
          >
            <option value={10}>Show 10</option>
            <option value={20}>Show 20</option>
            <option value={50}>Show 50</option>
            <option value={100}>Show 100</option>
          </select>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs text-[#1a1a1a]">
          <thead className="bg-[#1a1a1a] text-[#f2efeb] font-mono font-bold uppercase tracking-wider text-[10px]">
            <tr>
              <th className="px-4 py-2.5">Index</th>
              <th className="px-4 py-2.5">Original (FP32)</th>
              <th className="px-4 py-2.5 text-center">Trace</th>
              <th className="px-4 py-2.5">Quantized (INT{metrics.bits})</th>
              <th className="px-4 py-2.5">Reconstructed (x̂)</th>
              <th className="px-4 py-2.5">Abs Error (|x - x̂|)</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[#1a1a1a]/15 font-mono">
            {filteredRows.map((row) => (
              <tr key={row.index} className="hover:bg-[#f2efeb]/60 transition-colors">
                <td className="px-4 py-2 font-medium text-[#1a1a1a]/50">#{row.index}</td>
                <td className="px-4 py-2 text-[#1a1a1a] font-bold">{row.original.toFixed(6)}</td>
                <td className="px-4 py-2 text-center text-[#4f46e5]">
                  <ArrowRight className="w-3.5 h-3.5 inline-block" />
                </td>
                <td className="px-4 py-2">
                  <span className="inline-flex items-center px-2 py-0.5 text-xs font-bold bg-[#1a1a1a] text-[#f2efeb]">
                    {row.quantized}
                  </span>
                  <span className="text-[10px] text-[#1a1a1a]/50 ml-1.5">
                    ({row.quantized}/{metrics.qMax})
                  </span>
                </td>
                <td className="px-4 py-2 text-[#1a1a1a]">{row.reconstructed.toFixed(6)}</td>
                <td className="px-4 py-2">
                  <span
                    className={`font-semibold ${
                      row.absError > 0.1
                        ? 'text-rose-600'
                        : row.absError > 0.01
                        ? 'text-amber-600'
                        : 'text-emerald-700'
                    }`}
                  >
                    {row.absError.toFixed(6)}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
