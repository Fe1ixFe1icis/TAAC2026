import SectionHeader from '@/components/SectionHeader';
import FutureCard from '@/components/FutureCard';
import HorizontalCarousel from '@/components/effects/HorizontalCarousel';
import { useScrollReveal } from '@/hooks/useScrollReveal';

const directions = [
  {
    icon: '🌐',
    title: '全局 Token',
    description: '借鉴 BERT 的 [CLS] 设计，在输入序列开头添加全局聚合 Token，增强整体表征能力。预期收益: +0.001-0.003 AUC',
  },
  {
    icon: '🔗',
    title: '跨层残差',
    description: 'TokenMixer-Large 的 Inter-layer Residuals，解决深层模型梯度消失。每 N 层添加跨层残差连接，预期收益: +0.002-0.005 AUC',
  },
  {
    icon: '⚡',
    title: '稀疏 MoE',
    description: '逐域稀疏 Mixture of Experts，提升模型容量不增加推理成本。预期收益: +0.003-0.008 AUC',
  },
  {
    icon: '🎯',
    title: 'FP8 量化',
    description: '从 BF16 升级到 FP8，进一步加速训练和推理。结合 Token Parallel 大幅降低训练成本。',
  },
  {
    icon: '📈',
    title: '数据 Scaling',
    description: '探索更大规模数据集上的 Scaling Law 表现。当前 94M 样本已验证数据量对性能有显著影响。',
  },
];

export default function FutureDirectionsSection() {
  const sectionRef = useScrollReveal<HTMLElement>('.reveal-item', {
    y: 40,
    stagger: 0.1,
    start: 'top 80%',
  });

  return (
    <section
      id="future"
      ref={sectionRef}
      className="py-[20vh] px-[5vw]"
    >
      <div className="max-w-[1400px] mx-auto">
        <div className="reveal-item mb-4">
          <SectionHeader overline="FUTURE WORK" title="未来方向" />
        </div>
        <div className="reveal-item mb-12">
          <p className="text-text-secondary text-base max-w-2xl">
            基于 TokenMixer-Large 论文和实验观察，我们规划了以下四个探索方向。
          </p>
        </div>

        <div className="reveal-item">
          <HorizontalCarousel>
            {directions.map((dir) => (
              <FutureCard
                key={dir.title}
                icon={dir.icon}
                title={dir.title}
                description={dir.description}
              />
            ))}
          </HorizontalCarousel>
        </div>
      </div>
    </section>
  );
}
