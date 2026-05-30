import SectionHeader from '@/components/SectionHeader';
import MetricCard from '@/components/MetricCard';
import { useScrollReveal } from '@/hooks/useScrollReveal';

export default function OverviewSection() {
  const sectionRef = useScrollReveal<HTMLElement>('.reveal-item', {
    y: 40,
    stagger: 0.1,
    start: 'top 80%',
  });

  return (
    <section
      id="overview"
      ref={sectionRef}
      className="py-[20vh] px-[5vw]"
    >
      <div className="max-w-[1400px] mx-auto">
        <div className="reveal-item">
          <SectionHeader overline="PROJECT OVERVIEW" title="项目概述" />
        </div>

        <div className="flex flex-col lg:flex-row gap-12 lg:gap-20">
          {/* Left - Text */}
          <div className="lg:w-[45%] space-y-6">
            <div className="reveal-item">
              <p className="text-text-secondary text-base leading-relaxed">
                本项目是 <strong className="text-white">TAAC 2026</strong>（腾讯广告算法大赛）的参赛方案，探索基于{' '}
                <strong className="text-white">TokenFormer</strong>{' '}
                架构的点击率预估方法。核心思想是将所有特征（静态特征、连续特征、序列特征）统一表示为
                Token 序列，通过注意力机制实现端到端的特征交互。
              </p>
            </div>
            <div className="reveal-item">
              <p className="text-text-secondary text-base leading-relaxed">
                我们系统性地验证了模型各组件对 CTR 预测的影响（21 组实验 × 5000 steps），发现
                SwiGLU encoder 和紧凑 NS 表示对短序列 CTR 任务最有效。模型最终配置在{' '}
                <strong className="text-white">Avazu</strong>、{' '}
                <strong className="text-white">Criteo</strong> 和{' '}
                <strong className="text-white">Amazon</strong>{' '}
                三个不同数据集上进行了验证，证明其良好的泛化能力。
              </p>
            </div>
            <div className="reveal-item">
              <ul className="space-y-3 text-text-secondary text-sm">
                <li className="flex items-center gap-3">
                  <span className="w-1.5 h-1.5 rounded-full bg-accent-blue flex-shrink-0" />
                  <span>
                    数据集: TAAC 2026 广告 CTR 数据集（94M 联合样本）
                  </span>
                </li>
                <li className="flex items-center gap-3">
                  <span className="w-1.5 h-1.5 rounded-full bg-accent-purple flex-shrink-0" />
                  <span>基线: Last Login + Item2Vec + MLP (AUC 0.7452)</span>
                </li>
                <li className="flex items-center gap-3">
                  <span className="w-1.5 h-1.5 rounded-full bg-accent-teal flex-shrink-0" />
                  <span>目标: 探索 Scaling Law 在 CTR 预估中的有效性</span>
                </li>
              </ul>
            </div>
          </div>

          {/* Right - Metrics Grid */}
          <div className="lg:w-[55%] grid grid-cols-2 gap-4">
            <div className="reveal-item">
              <MetricCard value="0.7741" label="Best AUC" isAccent />
            </div>
            <div className="reveal-item">
              <MetricCard value="↑ 3.88%" label="相对基线提升" />
            </div>
            <div className="reveal-item">
              <MetricCard value="94M+" label="联合训练样本" />
            </div>
            <div className="reveal-item">
              <MetricCard value="130.4M" label="模型参数量" />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
