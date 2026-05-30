import { useEffect, useRef } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import SectionHeader from '@/components/SectionHeader';
import { useScrollReveal } from '@/hooks/useScrollReveal';

gsap.registerPlugin(ScrollTrigger);

const experiments = [
  { name: 'E6 (最佳)', auc: 0.7741, logloss: 0.4725, steps: '10K', config: 'gate_bias=2.0', delta: '+0.78%', isBest: true },
  { name: 'E4 (OneTrans)', auc: 0.7689, logloss: 0.4761, steps: '5K', config: 'mixed_params', delta: '+0.10%', isBest: false },
  { name: 'E2 (SWA 大窗口)', auc: 0.7682, logloss: 0.4767, steps: '5K', config: 'SWA=[128,64]', delta: '+0.01%', isBest: false },
  { name: 'E1 (基线)', auc: 0.7681, logloss: 0.4767, steps: '5K', config: '—', delta: '—', isBest: false },
  { name: 'E3 (SWA 小窗口)', auc: 0.7681, logloss: 0.4767, steps: '5K', config: 'SWA=[32,16]', delta: '0.00%', isBest: false },
  { name: 'E5 (门控 v1)', auc: 0.7680, logloss: 0.4768, steps: '5K', config: 'gate_bias=0.0', delta: '-0.01%', isBest: false },
];

const minAuc = 0.767;
const maxAuc = 0.775;

function AucBarChart() {
  const chartRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;

    const bars = chart.querySelectorAll('.chart-bar');
    
    const ctx = gsap.context(() => {
      gsap.from(bars, {
        scaleX: 0,
        duration: 1,
        stagger: 0.1,
        ease: 'power3.out',
        transformOrigin: 'left center',
        scrollTrigger: {
          trigger: chart,
          start: 'top 80%',
          toggleActions: 'play none none none',
        },
      });
    }, chart);

    return () => ctx.revert();
  }, []);

  return (
    <div ref={chartRef} className="glass-card p-8">
      <p className="section-overline mb-6">AUC TIMELINE</p>
      <div className="space-y-4">
        {experiments.map((exp) => {
          const width = ((exp.auc - minAuc) / (maxAuc - minAuc)) * 100;
          return (
            <div key={exp.name} className="flex items-center gap-4">
              <span className="text-xs font-mono text-text-muted w-28 text-right flex-shrink-0">
                {exp.name}
              </span>
              <div className="flex-1 h-8 bg-dark-elevated rounded-full overflow-hidden relative">
                <div
                  className="chart-bar h-full rounded-full relative"
                  style={{
                    width: `${width}%`,
                    background: exp.isBest
                      ? 'linear-gradient(90deg, #667eea, #764ba2)'
                      : 'linear-gradient(90deg, rgba(102,126,234,0.4), rgba(118,75,162,0.4))',
                    boxShadow: exp.isBest ? '0 0 20px rgba(102,126,234,0.4)' : 'none',
                  }}
                />
              </div>
              <span
                className={`text-sm font-mono font-semibold w-16 flex-shrink-0 ${
                  exp.isBest ? 'text-gradient' : 'text-text-muted'
                }`}
              >
                {exp.auc.toFixed(4)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function CoreResultsSection() {
  const sectionRef = useScrollReveal<HTMLElement>('.reveal-item', {
    y: 40,
    stagger: 0.1,
    start: 'top 80%',
  });

  return (
    <section
      id="results"
      ref={sectionRef}
      className="py-[20vh] px-[5vw]"
    >
      <div className="max-w-[1400px] mx-auto">
        <div className="reveal-item">
          <SectionHeader overline="CORE RESULTS" title="核心实验结果" />
        </div>

        {/* Comparison Table */}
        <div className="reveal-item mb-12 overflow-x-auto">
          <div className="glass-card p-0 overflow-hidden">
            <table className="data-table">
              <thead>
                <tr>
                  <th>实验</th>
                  <th>AUC</th>
                  <th>LogLoss</th>
                  <th>训练步数</th>
                  <th>关键配置</th>
                  <th>相对基线</th>
                </tr>
              </thead>
              <tbody>
                {experiments.map((exp) => (
                  <tr key={exp.name} className={exp.isBest ? 'best-row' : ''}>
                    <td className={exp.isBest ? 'font-semibold text-white' : ''}>
                      {exp.isBest && <span className="mr-2">⭐</span>}
                      {exp.name}
                    </td>
                    <td className={exp.isBest ? 'text-accent-blue font-bold' : ''}>
                      {exp.auc.toFixed(4)}
                    </td>
                    <td>{exp.logloss.toFixed(4)}</td>
                    <td>{exp.steps}</td>
                    <td className="text-text-muted">{exp.config}</td>
                    <td
                      className={
                        exp.delta.startsWith('+')
                          ? 'text-accent-teal'
                          : exp.delta.startsWith('-')
                          ? 'text-accent-red'
                          : 'text-text-muted'
                      }
                    >
                      {exp.delta}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* AUC Bar Chart */}
        <div className="reveal-item mb-12">
          <AucBarChart />
        </div>

        {/* ASCII Visualization */}
        <div className="reveal-item">
          <div className="glass-card p-8">
            <p className="section-overline mb-4">AUC IMPROVEMENT TIMELINE</p>
            <pre className="font-mono text-sm text-text-muted overflow-x-auto leading-relaxed">
{`AUC 提升时间线
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
基线 (E1)         ████████████████████████████████████  0.7681
+ AMP BF16        ████████████████████████████████████  0.7681  (速度 ↑)
+ SWA 调参        ████████████████████████████████████░ 0.7682  (+0.01%)
+ OneTrans        ████████████████████████████████████░ 0.7689  (+0.10%)
+ 10K 步          █████████████████████████████████████ 0.7741  (+0.78%) ⭐
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`}
            </pre>
          </div>
        </div>
      </div>
    </section>
  );
}
