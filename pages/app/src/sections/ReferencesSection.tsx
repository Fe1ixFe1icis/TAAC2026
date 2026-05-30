import SectionHeader from '@/components/SectionHeader';
import { useScrollReveal } from '@/hooks/useScrollReveal';

const references = [
  {
    index: '01',
    title: 'TokenMixer-Large',
    authors: 'Yuchen Jiang, Jie Zhu, Xintian Han, et al. (ByteDance AML)',
    detail: 'Scaling Up Large Ranking Models in Industrial Recommenders — arXiv:2602.06563v2 [cs.IR] — 核心洞察: Mixing & Reverting, Sparse Per-token MoE',
  },
  {
    index: '02',
    title: 'RankMixer',
    authors: 'ByteDance',
    detail: '面向排序的 TokenMixer 骨干网络 — 硬件协同设计实现高 MFU',
  },
  {
    index: '03',
    title: 'PCVRHyFormer',
    authors: 'TAAC 基线',
    detail: '多数据集 CTR 预估框架 — NS Tokenizer, RoPE, Flash Attention',
  },
  {
    index: '04',
    title: 'HSTU',
    authors: 'Meta, 2024',
    detail: '层次化序列转导单元 — 用于推荐系统的 Transformer 变体',
  },
  {
    index: '05',
    title: 'DHEN',
    authors: 'Google',
    detail: '深度层次集成网络 — 多尺度特征交互架构',
  },
  {
    index: '06',
    title: 'Wukong',
    authors: 'ByteDance',
    detail: '推荐系统 Scaling Law — 大规模推荐模型的 scaling 行为研究',
  },
];

export default function ReferencesSection() {
  const sectionRef = useScrollReveal<HTMLElement>('.reference-row', {
    y: 20,
    stagger: 0.08,
    start: 'top 80%',
  });

  return (
    <section
      id="references"
      ref={sectionRef}
      className="py-[20vh] px-[5vw]"
    >
      <div className="max-w-[1400px] mx-auto">
        <SectionHeader overline="REFERENCES" title="参考文献" />

        <div className="space-y-0">
          {references.map((ref) => (
            <div key={ref.index} className="reference-row">
              <div className="flex items-start gap-6">
                <span className="font-mono text-xs text-text-muted flex-shrink-0 mt-1">
                  {ref.index}
                </span>
                <div>
                  <h3 className="text-lg font-medium text-white mb-1">
                    {ref.title}
                  </h3>
                  <p className="text-sm text-text-secondary mb-1">
                    {ref.authors}
                  </p>
                  <p className="text-sm text-text-muted italic">
                    {ref.detail}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
