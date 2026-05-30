import { useEffect, useRef } from 'react';
import { getLenis } from '@/hooks/useLenis';

interface ScrollVelocityWrapperProps {
  children: React.ReactNode;
}

export default function ScrollVelocityWrapper({ children }: ScrollVelocityWrapperProps) {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const velocityRef = useRef(0);
  const rafRef = useRef<number>(0);

  useEffect(() => {
    const wrapper = wrapperRef.current;
    if (!wrapper) return;

    const lenis = getLenis();
    if (!lenis) return;

    const handleScroll = () => {
      velocityRef.current = lenis.velocity;
    };

    lenis.on('scroll', handleScroll);

    let currentSkew = 0;
    let currentScale = 1;

    const animate = () => {
      rafRef.current = requestAnimationFrame(animate);

      // Damping toward 0
      velocityRef.current *= 0.95;

      // Target values with caps
      const targetSkew = Math.max(-2, Math.min(2, velocityRef.current * 0.001));
      const targetScale = Math.min(1.002, 1 + Math.abs(velocityRef.current) * 0.0001);

      // Smooth interpolation
      currentSkew += (targetSkew - currentSkew) * 0.1;
      currentScale += (targetScale - currentScale) * 0.1;

      // Apply transform
      if (Math.abs(currentSkew) > 0.001 || Math.abs(currentScale - 1) > 0.0001) {
        wrapper.style.transform = `skewY(${currentSkew}deg) scaleY(${currentScale})`;
      } else {
        wrapper.style.transform = '';
      }
    };

    animate();

    return () => {
      cancelAnimationFrame(rafRef.current);
      lenis.off('scroll', handleScroll);
    };
  }, []);

  return (
    <div ref={wrapperRef} style={{ willChange: 'transform' }}>
      {children}
    </div>
  );
}
