import React from 'react';
import { Cpu, Zap, BookOpen, Terminal } from 'lucide-react';

interface HeaderProps {
  activeTab: 'demo' | 'model' | 'theory' | 'code';
  setActiveTab: (tab: 'demo' | 'model' | 'theory' | 'code') => void;
}

export const Header: React.FC<HeaderProps> = ({ activeTab, setActiveTab }) => {
  return (
    <header className="border-b-2 border-[#1a1a1a] bg-[#f2efeb] text-[#1a1a1a]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        
        <div className="flex flex-col md:flex-row md:items-center justify-between py-4 gap-4">
          
          {/* Logo & Branding */}
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-[#1a1a1a] text-[#f2efeb]">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h1 className="text-xl font-syne uppercase tracking-tight text-[#1a1a1a]">
                  QUANTFORGE
                </h1>
                <span className="px-2 py-0.5 text-[10px] font-mono font-bold bg-[#4f46e5] text-white uppercase tracking-wider">
                  PROTOTYPE 2
                </span>
              </div>
              <p className="text-xs text-[#1a1a1a]/70 font-mono">
                Neural Weight Quantization & Compression Engine
              </p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="flex flex-wrap gap-2">
            <button
              onClick={() => setActiveTab('demo')}
              className={`inline-flex items-center px-3.5 py-2 text-xs font-mono font-bold uppercase tracking-wider transition-all border-2 ${
                activeTab === 'demo'
                  ? 'bg-[#1a1a1a] text-[#f2efeb] border-[#1a1a1a]'
                  : 'bg-transparent text-[#1a1a1a] border-[#1a1a1a]/30 hover:border-[#1a1a1a]'
              }`}
            >
              <Zap className="w-3.5 h-3.5 mr-1.5" />
              Signal Lab (P1)
            </button>

            <button
              onClick={() => setActiveTab('model')}
              className={`inline-flex items-center px-3.5 py-2 text-xs font-mono font-bold uppercase tracking-wider transition-all border-2 ${
                activeTab === 'model'
                  ? 'bg-[#1a1a1a] text-[#f2efeb] border-[#1a1a1a]'
                  : 'bg-transparent text-[#1a1a1a] border-[#1a1a1a]/30 hover:border-[#1a1a1a]'
              }`}
            >
              <Cpu className="w-3.5 h-3.5 mr-1.5 text-[#4f46e5]" />
              Model Weights (P2)
            </button>

            <button
              onClick={() => setActiveTab('theory')}
              className={`inline-flex items-center px-3.5 py-2 text-xs font-mono font-bold uppercase tracking-wider transition-all border-2 ${
                activeTab === 'theory'
                  ? 'bg-[#1a1a1a] text-[#f2efeb] border-[#1a1a1a]'
                  : 'bg-transparent text-[#1a1a1a] border-[#1a1a1a]/30 hover:border-[#1a1a1a]'
              }`}
            >
              <BookOpen className="w-3.5 h-3.5 mr-1.5" />
              Quantization
            </button>

            <button
              onClick={() => setActiveTab('code')}
              className={`inline-flex items-center px-3.5 py-2 text-xs font-mono font-bold uppercase tracking-wider transition-all border-2 ${
                activeTab === 'code'
                  ? 'bg-[#1a1a1a] text-[#f2efeb] border-[#1a1a1a]'
                  : 'bg-transparent text-[#1a1a1a] border-[#1a1a1a]/30 hover:border-[#1a1a1a]'
              }`}
            >
              <Terminal className="w-3.5 h-3.5 mr-1.5" />
              Python Source
            </button>
          </nav>

        </div>
      </div>
    </header>
  );
};

