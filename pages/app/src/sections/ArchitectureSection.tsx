import { useState } from 'react';
import SectionHeader from '@/components/SectionHeader';
import { useScrollReveal } from '@/hooks/useScrollReveal';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

function TokenFormerDiagram() {
  return (
    <div className="glass-card p-8 md:p-10">
      <div className="flex flex-col items-center gap-4 font-mono text-sm">
        {/* Input */}
        <div className="w-full max-w-lg bg-dark-elevated border border-white/10 rounded-lg p-4 text-center">
          <span className="text-text-muted text-xs uppercase tracking-wider">输入</span>
          <p className="text-white mt-1">Token 序列 X ∈ ℝ^(B×N×D)</p>
        </div>
        
        <span className="text-accent-blue text-lg">↓</span>
        
        {/* RMSNorm */}
        <div className="w-48 bg-dark-elevated border border-white/10 rounded-lg p-3 text-center">
          <span className="text-accent-teal">RMSNorm</span>
        </div>
        
        <span className="text-accent-blue text-lg">↓</span>
        
        {/* Attention + Gating */}
        <div className="w-full max-w-2xl grid grid-cols-2 gap-4">
          <div className="bg-dark-elevated border border-white/10 rounded-lg p-4">
            <span className="text-text-muted text-xs uppercase tracking-wider block mb-2">投影</span>
            <span className="text-white">W_q, W_k, W_v, W_o</span>
          </div>
          <div className="bg-dark-elevated border border-white/10 rounded-lg p-4">
            <span className="text-text-muted text-xs uppercase tracking-wider block mb-2">门控</span>
            <span className="text-accent-purple">NLIR Gating</span>
            <span className="text-white block text-xs mt-1">W_g (gate_bias=2.0)</span>
          </div>
        </div>
        
        <span className="text-accent-blue text-lg">↓</span>
        
        {/* Attention Block */}
        <div className="w-full max-w-lg bg-gradient-to-r from-accent-blue/10 to-accent-purple/10 border border-accent-blue/30 rounded-lg p-5 text-center">
          <span className="text-white font-medium">缩放点积注意力</span>
          <span className="text-text-muted block text-xs mt-1">+ RoPE + BFTS 掩码</span>
        </div>
        
        <span className="text-accent-blue text-lg">↓</span>
        
        {/* Gated Output + Residual */}
        <div className="w-full max-w-lg grid grid-cols-2 gap-4">
          <div className="bg-dark-elevated border border-accent-teal/30 rounded-lg p-3 text-center">
            <span className="text-accent-teal">gate × A</span>
          </div>
          <div className="bg-dark-elevated border border-white/10 rounded-lg p-3 text-center">
            <span className="text-white">+ 残差连接</span>
          </div>
        </div>
        
        <span className="text-accent-blue text-lg">↓</span>
        
        {/* FFN */}
        <div className="w-56 bg-dark-elevated border border-accent-purple/30 rounded-lg p-4 text-center">
          <span className="text-accent-purple font-medium">SwiGLU FFN</span>
        </div>
        
        <span className="text-accent-blue text-lg">↓</span>
        
        {/* Output */}
        <div className="w-full max-w-lg bg-dark-elevated border border-white/10 rounded-lg p-4 text-center">
          <span className="text-text-muted text-xs uppercase tracking-wider">输出</span>
          <p className="text-white mt-1">X&apos; = I + FFN(I)</p>
        </div>
      </div>
    </div>
  );
}

function TokenStreamDiagram() {
  return (
    <div className="glass-card p-8 md:p-10">
      <div className="font-mono text-sm overflow-x-auto">
        <p className="text-text-muted text-xs uppercase tracking-wider mb-6">统一 Token 流表示</p>
        
        <div className="flex items-center gap-1 mb-8 min-w-max">
          <span className="text-white text-lg">[</span>
          
          {/* Static Feature Tokens */}
          <div className="flex flex-col items-center">
            <div className="flex gap-1">
              <span className="px-3 py-2 bg-accent-blue/20 border border-accent-blue/40 rounded text-accent-blue">F1</span>
              <span className="px-3 py-2 bg-accent-blue/20 border border-accent-blue/40 rounded text-accent-blue">F2</span>
              <span className="px-3 py-2 bg-accent-blue/10 border border-accent-blue/30 rounded text-text-muted">...</span>
              <span className="px-3 py-2 bg-accent-blue/20 border border-accent-blue/40 rounded text-accent-blue">F_n</span>
            </div>
            <span className="text-xs text-accent-blue mt-2">静态特征 Token (user_int, item_int)</span>
          </div>
          
          <span className="px-2 py-2 bg-accent-red/20 border border-accent-red/40 rounded text-accent-red mx-1">SEP</span>
          
          {/* Sequence Tokens */}
          <div className="flex flex-col items-center">
            <div className="flex gap-1">
              <span className="px-3 py-2 bg-accent-purple/20 border border-accent-purple/40 rounded text-accent-purple">T1</span>
              <span className="px-3 py-2 bg-accent-purple/20 border border-accent-purple/40 rounded text-accent-purple">T2</span>
              <span className="px-3 py-2 bg-accent-purple/10 border border-accent-purple/30 rounded text-text-muted">...</span>
              <span className="px-3 py-2 bg-accent-purple/20 border border-accent-purple/40 rounded text-accent-purple">T_m</span>
            </div>
            <span className="text-xs text-accent-purple mt-2">序列 Token (用户行为历史)</span>
          </div>
          
          <span className="px-2 py-2 bg-accent-red/20 border border-accent-red/40 rounded text-accent-red mx-1">SEP</span>
          
          {/* Target Token */}
          <div className="flex flex-col items-center">
            <span className="px-3 py-2 bg-accent-teal/20 border border-accent-teal/40 rounded text-accent-teal">V</span>
            <span className="text-xs text-accent-teal mt-2">目标 Token (CTR)</span>
          </div>
          
          <span className="text-white text-lg">]</span>
        </div>
        
        {/* Legend */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded bg-accent-blue/20 border border-accent-blue/40" />
            <span className="text-text-muted">静态特征 (31 user + 15 item fids)</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded bg-accent-purple/20 border border-accent-purple/40" />
            <span className="text-text-muted">序列特征 (per_field Tokenizer)</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded bg-accent-red/20 border border-accent-red/40" />
            <span className="text-text-muted">分隔符 (Domain 边界)</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded bg-accent-teal/20 border border-accent-teal/40" />
            <span className="text-text-muted">目标变量 (CTR 预估)</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function ArchitectureSection() {
  const [activeTab, setActiveTab] = useState('tokenformer');
  const sectionRef = useScrollReveal<HTMLElement>('.reveal-item', {
    y: 30,
    stagger: 0.15,
    start: 'top 80%',
  });

  return (
    <section
      id="architecture"
      ref={sectionRef}
      className="py-[20vh] px-[5vw]"
    >
      <div className="max-w-[1400px] mx-auto">
        <div className="reveal-item">
          <SectionHeader overline="MODEL ARCHITECTURE" title="模型架构" />
        </div>

        <div className="reveal-item">
          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="mb-8 bg-dark-card border border-white/10 p-1">
              <TabsTrigger
                value="tokenformer"
                className="px-6 py-2.5 text-sm font-mono uppercase tracking-wider data-[state=active]:bg-dark-elevated data-[state=active]:text-white text-text-muted transition-all"
              >
                TokenFormer 模块
              </TabsTrigger>
              <TabsTrigger
                value="tokenstream"
                className="px-6 py-2.5 text-sm font-mono uppercase tracking-wider data-[state=active]:bg-dark-elevated data-[state=active]:text-white text-text-muted transition-all"
              >
                统一 Token 流
              </TabsTrigger>
            </TabsList>
            
            <TabsContent value="tokenformer" className="transition-opacity duration-300">
              <TokenFormerDiagram />
            </TabsContent>
            
            <TabsContent value="tokenstream" className="transition-opacity duration-300">
              <TokenStreamDiagram />
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </section>
  );
}
