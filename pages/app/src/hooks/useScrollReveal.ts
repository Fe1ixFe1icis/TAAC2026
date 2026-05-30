import { useEffect, useRef } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

export function useScrollReveal<T extends HTMLElement>(
  selector?: string,
  options: {
    y?: number;
    x?: number;
    opacity?: number;
    duration?: number;
    stagger?: number;
    ease?: string;
    start?: string;
  } = {}
) {
  const containerRef = useRef<T>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const {
      y = 40,
      x = 0,
      opacity = 0,
      duration = 0.8,
      stagger = 0.1,
      ease = 'power3.out',
      start = 'top 80%',
    } = options;

    const targets = selector
      ? container.querySelectorAll(selector)
      : [container];

    const ctx = gsap.context(() => {
      gsap.from(targets, {
        y,
        x,
        opacity,
        duration,
        stagger,
        ease,
        scrollTrigger: {
          trigger: container,
          start,
          toggleActions: 'play none none none',
        },
      });
    }, container);

    return () => ctx.revert();
  }, [selector, options.y, options.x, options.opacity, options.duration, options.stagger, options.ease, options.start]);

  return containerRef;
}

export function useScrollProgress() {
  const progressRef = useRef(0);

  useEffect(() => {
    const updateProgress = () => {
      const scrollTop = window.scrollY;
      const docHeight = document.documentElement.scrollHeight - window.innerHeight;
      progressRef.current = docHeight > 0 ? scrollTop / docHeight : 0;
    };

    window.addEventListener('scroll', updateProgress, { passive: true });
    updateProgress();

    return () => window.removeEventListener('scroll', updateProgress);
  }, []);

  return progressRef;
}
