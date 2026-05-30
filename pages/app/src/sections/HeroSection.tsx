import { useEffect, useRef } from 'react';
import gsap from 'gsap';
import LiquidLensShader from '@/components/effects/LiquidLensShader';
import { useScrollTo } from '@/hooks/useLenis';
import { ChevronRight } from 'lucide-react';

export default function HeroSection() {
  const sectionRef = useRef<HTMLElement>(null);
  const overlineRef = useRef<HTMLParagraphElement>(null);
  const titleRef = useRef<HTMLHeadingElement>(null);
  const subtitleRef = useRef<HTMLParagraphElement>(null);
  const badgeRef = useRef<HTMLDivElement>(null);
  const buttonsRef = useRef<HTMLDivElement>(null);
  const scrollTo = useScrollTo();

  useEffect(() => {
    const tl = gsap.timeline();

    tl.from(overlineRef.current, {
      opacity: 0,
      y: 30,
      duration: 0.8,
      delay: 0.2,
      ease: 'power3.out',
    })
      .from(
        titleRef.current,
        {
          opacity: 0,
          y: 50,
          duration: 1,
          ease: 'power3.out',
        },
        0.5
      )
      .from(
        subtitleRef.current,
        {
          opacity: 0,
          y: 30,
          duration: 0.8,
          ease: 'power3.out',
        },
        0.8
      )
      .from(
        badgeRef.current,
        {
          opacity: 0,
          scale: 0.8,
          duration: 1,
          ease: 'elastic.out(1, 0.5)',
        },
        1.2
      )
      .from(
        buttonsRef.current,
        {
          opacity: 0,
          y: 20,
          duration: 0.6,
          ease: 'power3.out',
        },
        1.5
      );

    return () => {
      tl.kill();
    };
  }, []);

  return (
    <section
      id="hero"
      ref={sectionRef}
      className="relative min-h-screen flex items-center overflow-hidden"
    >
      {/* Video Background */}
      <video
        autoPlay
        muted
        loop
        playsInline
        className="absolute inset-0 w-full h-full object-cover"
        style={{ filter: 'saturate(1.2) contrast(1.1)', zIndex: 0 }}
      >
        <source src="/videos/hero-bg.mp4" type="video/mp4" />
      </video>

      {/* Liquid Lens Shader Overlay */}
      <LiquidLensShader />

      {/* Dark Gradient Overlay */}
      <div
        className="absolute inset-0"
        style={{
          background:
            'linear-gradient(to bottom, rgba(5,5,5,0.3) 0%, rgba(5,5,5,0.6) 70%, rgba(5,5,5,1) 100%)',
          zIndex: 2,
        }}
      />

      {/* Content */}
      <div
        className="relative z-10 max-w-[1400px] mx-auto px-[5vw] w-full py-32 flex flex-col lg:flex-row items-center lg:items-start justify-between gap-12"
      >
        {/* Left Content */}
        <div className="flex-1 max-w-3xl">
          <p
            ref={overlineRef}
            className="font-mono text-xs uppercase tracking-[0.15em] text-text-muted mb-6"
          >
            TAAC 2026 &middot; GROUP MEETING REPORT
          </p>

          <h1
            ref={titleRef}
            className="text-white uppercase mb-6"
            style={{
              fontSize: 'clamp(4rem, 10vw, 10rem)',
              fontWeight: 400,
              letterSpacing: '-0.04em',
              lineHeight: 0.9,
              textShadow: '0 0 80px rgba(102,126,234,0.3)',
            }}
          >
            Token
            <br />
            Former
          </h1>

          <p
            ref={subtitleRef}
            className="text-xl md:text-2xl font-light text-text-muted tracking-wide mb-10"
            style={{ letterSpacing: '0.02em' }}
          >
            统一 Token 流架构的点击率预估方案
          </p>

          <div ref={buttonsRef} className="flex flex-wrap gap-4">
            <button
              onClick={() => scrollTo('#results')}
              className="pill-btn"
            >
              查看实验结果
              <ChevronRight size={18} />
            </button>
            <button
              onClick={() => scrollTo('#architecture')}
              className="pill-btn"
            >
              模型架构
              <ChevronRight size={18} />
            </button>
          </div>
        </div>

        {/* Right - AUC Badge */}
        <div ref={badgeRef} className="flex-shrink-0 lg:mt-[15vh]">
          <div
            className="gradient-border-badge rounded-full flex flex-col items-center justify-center text-center"
            style={{ width: '200px', height: '200px' }}
          >
            <span className="text-sm text-text-muted mb-1">🏆 最佳 AUC</span>
            <span
              className="metric-number metric-number-accent"
              style={{ fontSize: '48px' }}
            >
              0.7741
            </span>
            <span className="text-xs text-text-muted mt-1">
              腾讯广告算法大赛 2026
            </span>
          </div>
        </div>
      </div>
    </section>
  );
}
