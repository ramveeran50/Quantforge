import React from 'react';
import { BookOpen, HelpCircle, CheckCircle, AlertTriangle, Layers, Compass } from 'lucide-react';

export const TheoryGuide: React.FC = () => {
  return (
    <div className="space-y-6 max-w-5xl mx-auto font-sans text-[#1a1a1a]">
      
      {/* Hero card */}
      <div className="bg-[#ffffff] border-2 border-[#1a1a1a] p-6 sm:p-8">
        <div className="flex items-center space-x-2 text-[#4f46e5] text-xs font-mono font-bold uppercase tracking-wider mb-2">
          <BookOpen className="w-4 h-4" />
          <span>Research Reference</span>
        </div>
        <h2 className="text-xl sm:text-2xl font-syne uppercase tracking-wider mb-3 text-[#1a1a1a]">
          Understanding AI Quantization & Compression
        </h2>
        <p className="text-[#1a1a1a]/80 text-xs leading-relaxed max-w-3xl font-mono">
          Quantization maps high-precision FP32 floats into low-bit discrete integers (INT16, INT8, INT4, INT2), enabling parameter-dense AI models to run on resource-constrained accelerators by dramatically reducing memory footprint and bandwidth requirements.
        </p>
      </div>

      {/* Grid 1: Core Concepts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        
        <div className="bg-[#ffffff] border-2 border-[#1a1a1a] p-6">
          <h3 className="text-sm font-syne uppercase tracking-wider text-[#1a1a1a] mb-2 flex items-center">
            <HelpCircle className="w-4 h-4 mr-2 text-[#4f46e5]" />
            1. What is Quantization?
          </h3>
          <p className="text-xs text-[#1a1a1a]/80 font-mono leading-relaxed">
            Quantization maps a continuous numerical domain to a finite set of discrete integer levels. In neural network compression, 32-bit floating-point weights are discretized into 8-bit, 4-bit, or 2-bit integer representations.
          </p>
        </div>

        <div className="bg-[#ffffff] border-2 border-[#1a1a1a] p-6">
          <h3 className="text-sm font-syne uppercase tracking-wider text-[#1a1a1a] mb-2 flex items-center">
            <CheckCircle className="w-4 h-4 mr-2 text-emerald-600" />
            2. Compression Feasibility
          </h3>
          <p className="text-xs text-[#1a1a1a]/80 font-mono leading-relaxed">
            Neural network weights exhibit Gaussian or Laplace distributions centered around zero. Minor quantization perturbations produce minimal impact on inference accuracy while unlocking massive RAM savings.
          </p>
        </div>

      </div>

      {/* Formula Card */}
      <div className="bg-[#ffffff] border-2 border-[#1a1a1a] p-6 font-mono">
        <h3 className="text-sm font-syne uppercase tracking-wider text-[#1a1a1a] mb-3 flex items-center">
          <Layers className="w-4 h-4 mr-2 text-[#4f46e5]" />
          3. Uniform Scalar Quantization Equations
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 my-4 text-xs">
          <div className="p-4 bg-[#f2efeb] border border-[#1a1a1a]">
            <div className="font-bold text-[#1a1a1a] mb-1 uppercase tracking-wider text-[11px]">Scale (S) Equation:</div>
            <div className="text-[#4f46e5] font-bold text-xs">S = (x_max - x_min) / (q_max - q_min)</div>
            <p className="text-[10px] text-[#1a1a1a]/70 mt-2">
              The continuous float step size corresponding to one integer level increment.
            </p>
          </div>

          <div className="p-4 bg-[#f2efeb] border border-[#1a1a1a]">
            <div className="font-bold text-[#1a1a1a] mb-1 uppercase tracking-wider text-[11px]">Zero-Point (Z) Equation:</div>
            <div className="text-rose-600 font-bold text-xs">Z = clamp(round(-x_min / S), q_min, q_max)</div>
            <p className="text-[10px] text-[#1a1a1a]/70 mt-2">
              The integer index mapping directly to 0.0 float, preserving exact zero padding.
            </p>
          </div>
        </div>

        <div className="bg-[#1a1a1a] text-[#f2efeb] p-4 border border-[#1a1a1a] text-xs space-y-1">
          <div><span className="opacity-50">// Quantize (Float -&gt; Packed Integer Bin):</span></div>
          <div className="text-emerald-400 font-bold">q = clamp(round(x / S) + Z, q_min, q_max)</div>
          <div className="pt-2"><span className="opacity-50">// Reconstruct (Integer Bin -&gt; Approx Float):</span></div>
          <div className="text-[#818cf8] font-bold">x_hat = (q - Z) * S</div>
        </div>
      </div>

      {/* Bit Packing & Actual Storage Card */}
      <div className="bg-[#ffffff] border-2 border-[#1a1a1a] p-6 font-mono">
        <h3 className="text-sm font-syne uppercase tracking-wider text-[#1a1a1a] mb-3 flex items-center">
          <Layers className="w-4 h-4 mr-2 text-[#4f46e5]" />
          4. Bit Packing Math: Real vs Theoretical Compression
        </h3>

        <p className="text-xs text-[#1a1a1a]/80 leading-relaxed mb-4">
          Standard C/NumPy arrays store integers in 8-bit uint8 bytes. To achieve actual physical compression for 4-bit and 2-bit quantization, multiple values must be packed bitwise into shared byte registers:
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          <div className="p-4 bg-[#f2efeb] border border-[#1a1a1a]">
            <div className="font-bold text-[#4f46e5] mb-1 uppercase tracking-wider">4-Bit Bit-Packing (2 values / byte):</div>
            <div className="text-[#1a1a1a] bg-[#ffffff] p-2 border border-[#1a1a1a] font-bold text-[11px] mb-2">
              byte = (val_0 &lt;&lt; 4) | val_1
            </div>
            <p className="text-[10px] text-[#1a1a1a]/70">
              10,000 FP32 values (40,000B) consume 5,000 packed bytes (8.0x reduction).
            </p>
          </div>

          <div className="p-4 bg-[#f2efeb] border border-[#1a1a1a]">
            <div className="font-bold text-purple-600 mb-1 uppercase tracking-wider">2-Bit Bit-Packing (4 values / byte):</div>
            <div className="text-[#1a1a1a] bg-[#ffffff] p-2 border border-[#1a1a1a] font-bold text-[11px] mb-2">
              byte = (v0 &lt;&lt; 6) | (v1 &lt;&lt; 4) | (v2 &lt;&lt; 2) | v3
            </div>
            <p className="text-[10px] text-[#1a1a1a]/70">
              10,000 FP32 values (40,000B) consume 2,500 packed bytes (16.0x reduction).
            </p>
          </div>
        </div>

        <div className="mt-4 p-3 bg-[#f2efeb] border border-[#1a1a1a] text-xs text-[#1a1a1a]">
          <strong>Header Overhead:</strong> Dequantization requires metadata storage (Scale S [4B] + Zero-Point Z [4B] + Count N [4B] + Bits B [1B] = 13B). Actual ratio = <code className="text-[#4f46e5] font-bold">Original_Bytes / (Packed_Bytes + 13)</code>.
        </div>
      </div>

      {/* Difference from Google's TurboQuant */}
      <div className="bg-[#ffffff] border-2 border-[#1a1a1a] p-6 font-mono">
        <h3 className="text-sm font-syne uppercase tracking-wider text-[#1a1a1a] mb-2 flex items-center">
          <AlertTriangle className="w-4 h-4 mr-2 text-amber-600" />
          Prototype Distinction from Google's TurboQuant
        </h3>
        <p className="text-xs text-[#1a1a1a]/80 leading-relaxed mb-3">
          This prototype demonstrates <strong>uniform scalar quantization</strong> for education and benchmarking. Production TurboQuant uses advanced non-linear vector codebooks and activation outlier preservation.
        </p>
        <ul className="list-disc list-inside text-xs text-[#1a1a1a]/80 space-y-1">
          <li><strong>Scalar vs Vector:</strong> This prototype quantizes scalars independently. TurboQuant quantizes multi-dimensional weight vectors with codebooks.</li>
          <li><strong>Outlier Isolation:</strong> TurboQuant isolates attention activation spikes into high-precision channels.</li>
        </ul>
      </div>

    </div>
  );
};
