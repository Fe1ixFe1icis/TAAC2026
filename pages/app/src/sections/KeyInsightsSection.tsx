import SectionHeader from '@/components/SectionHeader';
import InsightCard from '@/components/InsightCard';
import { useScrollReveal } from '@/hooks/useScrollReveal';

const insights = [
  {
    number: '+0.0060',
    title: '🎯 训练时长最关键',
    description: '10K 步 vs 5K 步 → AUC 提升 +0.0060。这是所有优化中收益最大的单一因素。说明 TokenFormer 架构需要充分训练才能发挥潜力。',
    accentColor: '#4ECDC4',
  },
  {
    number: '2.0',
    title: '🔓 门控初始化至关重要',
    description: 'gate_bias=2.0 (sigmoid≈0.88) 显著优于 bias=0.0。初始门控接近"通过"状态避免了早期训练的信息损失。',
    accentColor: '#667eea',
  },
  {
    number: '<0.0001',
    title: '📐 SWA 窗口影响微弱',
    description: '[32,16] vs [64,32] vs [128,64] → AUC 差异 < 0.0001。滑动窗口注意力对短序列 CTR 任务不敏感。',
    accentColor: '#764ba2',
  },
  {
    number: '2×',
    title: '⚖️ OneTrans 性价比不高',
    description: '混合参数带来 +0.0007 AUC，但训练时间翻倍（13min → 30min）。在资源受限场景下不推荐。',
    accentColor: '#FF6B6B',
  },
];

export default function KeyInsightsSection() {
  const sectionRef = useScrollReveal<HTMLElement>('.insight-card-reveal', {
    y: 40,
    stagger: 0.15,
    start: 'top 80%',
  });

  return (
    <section
      id="insights"
      ref={sectionRef}
      className="py-[20vh] px-[5vw]"
    >
      <div className="max-w-[1400px] mx-auto">
        <SectionHeader overline="KEY INSIGHTS" title="关键洞察" />

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {insights.map((insight) => (
            <div key={insight.title} className="insight-card-reveal">
              <InsightCard
                number={insight.number}
                title={insight.title}
                description={insight.description}
                accentColor={insight.accentColor}
              />
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
