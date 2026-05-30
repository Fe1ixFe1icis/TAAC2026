import { useEffect, useRef } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import SectionHeader from '@/components/SectionHeader';
import { Zap, Layers, Trophy, Rocket } from 'lucide-react';

gsap.registerPlugin(ScrollTrigger);

const phases = [
  {
    id: 'A',
    title: '训练效率（快速收益）',
    icon: Zap,
    color: '#4ECDC4',
    items: [
      '✅ AMP BF16 混合精度',
      '✅ Batch size 512（从 256 提升）',
      '✅ SWA 窗口调参',
    ],
    description: '首先优化训练效率，通过混合精度和更大的 batch size 加速训练，为后续实验节省时间。',
  },
  {
    id: 'B',
    title: '结构改进',
    icon: Layers,
    color: '#667eea',
    items: [
      '✅ OneTrans 混合参数机制',
      '✅ 逐域门控 + 改进初始化',
      '⚠️ d_model 扩展放弃（显存限制）',
    ],
    description: '引入 OneTrans 混合参数和门控机制，尝试提升模型表达能力。',
  },
  {
    id: 'C',
    title: '训练时长',
    icon: Trophy,
    color: '#764ba2',
    items: [
      '✅ 扩展到 10K 步',
      '✅ 门控偏置初始化 = 2.0',
      '🏆 最佳 AUC: 0.7741',
    ],
    description: '关键发现：延长训练是最有效的优化手段。门控初始化 bias=2.0 带来显著提升。',
  },
  {
    id: 'D',
    title: '未来方向',
    icon: Rocket,
    color: '#FF6B6B',
    items: [
      '🔄 全局 Token（CLS 风格聚合）',
      '🔄 跨层残差（深层模型）',
      '🔄 稀疏逐域 MoE',
      '🔄 FP8 量化',
    ],
    description: '基于 TokenMixer-Large 论文的启发，规划下一步优化方向。',
  },
];

export default function OptimizationJourneySection() {
  const sectionRef = useRef<HTMLElement>(null);
  const timelineRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const section = sectionRef.current;
    const timeline = timelineRef.current;
    if (!section || !timeline) return;

    const ctx = gsap.context(() => {
      // Timeline line draw animation
      gsap.from(timeline, {
        scaleY: 0,
        transformOrigin: 'top center',
        duration: 1.5,
        ease: 'power3.out',
        scrollTrigger: {
          trigger: section,
          start: 'top 60%',
          toggleActions: 'play none none none',
        },
      });

      // Phase cards animation
      const cards = section.querySelectorAll('.timeline-card');
      cards.forEach((card, i) => {
        const fromX = i % 2 === 0 ? -50 : 50;
        gsap.from(card, {
          x: fromX,
          opacity: 0,
          duration: 0.8,
          ease: 'power3.out',
          scrollTrigger: {
            trigger: card,
            start: 'top 85%',
            toggleActions: 'play none none none',
          },
        });
      });

      // Phase nodes animation
      const nodes = section.querySelectorAll('.timeline-node');
      nodes.forEach((node) => {
        gsap.from(node, {
          scale: 0,
          opacity: 0,
          duration: 0.6,
          ease: 'back.out(1.7)',
          scrollTrigger: {
            trigger: node,
            start: 'top 85%',
            toggleActions: 'play none none none',
          },
        });
      });
    }, section);

    return () => ctx.revert();
  }, []);

  return (
    <section
      id="journey"
      ref={sectionRef}
      className="py-[20vh] px-[5vw]"
    >
      <div className="max-w-[1400px] mx-auto">
        <SectionHeader overline="OPTIMIZATION JOURNEY" title="优化历程" />

        <div className="relative">
          {/* Timeline Line */}
          <div
            ref={timelineRef}
            className="absolute left-1/2 top-0 bottom-0 w-[2px] -translate-x-1/2 hidden md:block"
            style={{
              background: 'linear-gradient(to bottom, #4ECDC4, #667eea, #764ba2, #FF6B6B)',
            }}
          />

          {/* Mobile Timeline Line */}
          <div
            className="absolute left-6 top-0 bottom-0 w-[2px] md:hidden"
            style={{
              background: 'linear-gradient(to bottom, #4ECDC4, #667eea, #764ba2, #FF6B6B)',
            }}
          />

          {/* Phases */}
          <div className="space-y-16 md:space-y-24">
            {phases.map((phase, index) => {
              const Icon = phase.icon;
              const isLeft = index % 2 === 0;

              return (
                <div
                  key={phase.id}
                  className={`relative flex items-start gap-8 ${
                    isLeft ? 'md:flex-row' : 'md:flex-row-reverse'
                  } flex-row`}
                >
                  {/* Content Card */}
                  <div
                    className={`timeline-card flex-1 md:max-w-[480px] ${
                      isLeft ? 'md:text-right md:ml-0' : 'md:text-left md:mr-0'
                    } ml-16 md:ml-0`}
                  >
                    <div
                      className="glass-card p-6 md:p-8"
                      style={{ borderLeft: `3px solid ${phase.color}` }}
                    >
                      <p className="font-mono text-xs uppercase tracking-[0.1em] text-text-muted mb-3">
                        Phase {phase.id}
                      </p>
                      <h3 className="text-xl font-medium text-white mb-3">
                        {phase.title}
                      </h3>
                      <ul className="space-y-2 mb-4">
                        {phase.items.map((item, i) => (
                          <li
                            key={i}
                            className={`text-sm ${
                              item.startsWith('🏆')
                                ? 'text-accent-blue font-semibold'
                                : item.startsWith('⚠️')
                                ? 'text-accent-red'
                                : 'text-text-secondary'
                            }`}
                          >
                            {item}
                          </li>
                        ))}
                      </ul>
                      <p className="text-xs text-text-muted leading-relaxed">
                        {phase.description}
                      </p>
                    </div>
                  </div>

                  {/* Timeline Node */}
                  <div className="timeline-node absolute left-6 md:left-1/2 -translate-x-1/2 w-10 h-10 rounded-full bg-dark-bg border-2 flex items-center justify-center z-10 flex-shrink-0">
                    <Icon size={16} style={{ color: phase.color }} />
                  </div>

                  {/* Spacer for opposite side */}
                  <div className="hidden md:block flex-1 max-w-[480px]" />
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
}
