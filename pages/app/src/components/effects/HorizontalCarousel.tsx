import { useRef, useCallback, useEffect, useState } from 'react';

interface HorizontalCarouselProps {
  children: React.ReactNode;
}

export default function HorizontalCarousel({ children }: HorizontalCarouselProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const innerRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  
  const stateRef = useRef({
    isDragging: false,
    startX: 0,
    currentX: 0,
    lastX: 0,
    velocity: 0,
    lastTime: 0,
    rafId: 0,
  });

  const getCardWidth = useCallback(() => {
    const inner = innerRef.current;
    if (!inner) return 404; // 380 + 24 gap
    const firstCard = inner.children[0] as HTMLElement;
    if (!firstCard) return 404;
    return firstCard.offsetWidth + 24; // card width + gap
  }, []);

  const getMaxScroll = useCallback(() => {
    const inner = innerRef.current;
    const container = containerRef.current;
    if (!inner || !container) return 0;
    return Math.max(0, inner.scrollWidth - container.offsetWidth);
  }, []);

  const snapToBoundary = useCallback(() => {
    const state = stateRef.current;
    const cardWidth = getCardWidth();
    const maxScroll = getMaxScroll();
    
    let targetX = Math.round(state.currentX / cardWidth) * cardWidth;
    targetX = Math.max(-maxScroll, Math.min(0, targetX));
    
    const startX = state.currentX;
    const diff = targetX - startX;
    const startTime = performance.now();
    const duration = 600;

    const animate = (now: number) => {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3); // easeOutCubic
      
      state.currentX = startX + diff * eased;
      
      if (innerRef.current) {
        innerRef.current.style.transform = `translateX(${state.currentX}px)`;
      }
      
      if (progress < 1) {
        stateRef.current.rafId = requestAnimationFrame(animate);
      }
    };
    
    stateRef.current.rafId = requestAnimationFrame(animate);
  }, [getCardWidth, getMaxScroll]);

  const springAnimate = useCallback(() => {
    const state = stateRef.current;
    const maxScroll = getMaxScroll();
    
    const damping = 0.0005;
    
    // Apply spring physics
    state.velocity *= (1 - damping);
    state.currentX += state.velocity;
    
    // Boundary constraints with bounce
    if (state.currentX > 0) {
      state.currentX *= 0.8;
      state.velocity *= -0.5;
    } else if (state.currentX < -maxScroll) {
      const overshoot = state.currentX + maxScroll;
      state.currentX = -maxScroll + overshoot * 0.8;
      state.velocity *= -0.5;
    }
    
    if (Math.abs(state.velocity) > 0.1 || state.currentX > 0.5 || state.currentX < -maxScroll - 0.5) {
      if (innerRef.current) {
        innerRef.current.style.transform = `translateX(${state.currentX}px)`;
      }
      stateRef.current.rafId = requestAnimationFrame(springAnimate);
    } else {
      snapToBoundary();
    }
  }, [getMaxScroll, snapToBoundary]);

  const handlePointerDown = useCallback((e: React.PointerEvent) => {
    const state = stateRef.current;
    cancelAnimationFrame(state.rafId);
    
    state.isDragging = true;
    state.startX = e.clientX - state.currentX;
    state.lastX = state.currentX;
    state.lastTime = performance.now();
    state.velocity = 0;
    
    setIsDragging(true);
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  }, []);

  const handlePointerMove = useCallback((e: React.PointerEvent) => {
    const state = stateRef.current;
    if (!state.isDragging) return;
    
    const now = performance.now();
    const dt = now - state.lastTime;
    
    state.currentX = e.clientX - state.startX;
    
    if (dt > 0) {
      state.velocity = (state.currentX - state.lastX) / dt * 16; // normalize to ~60fps
    }
    
    state.lastX = state.currentX;
    state.lastTime = now;
    
    if (innerRef.current) {
      innerRef.current.style.transform = `translateX(${state.currentX}px)`;
    }
  }, []);

  const handlePointerUp = useCallback(() => {
    const state = stateRef.current;
    state.isDragging = false;
    setIsDragging(false);
    
    // Continue with spring physics
    stateRef.current.rafId = requestAnimationFrame(springAnimate);
  }, [springAnimate]);

  // Handle wheel events for scrolling
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();
      const state = stateRef.current;
      cancelAnimationFrame(state.rafId);
      
      state.currentX -= e.deltaX || e.deltaY;
      const maxScroll = getMaxScroll();
      state.currentX = Math.max(-maxScroll, Math.min(0, state.currentX));
      
      if (innerRef.current) {
        innerRef.current.style.transform = `translateX(${state.currentX}px)`;
      }
    };

    container.addEventListener('wheel', handleWheel, { passive: false });
    return () => container.removeEventListener('wheel', handleWheel);
  }, [getMaxScroll]);

  return (
    <div
      ref={containerRef}
      className="w-full overflow-hidden"
      style={{ cursor: isDragging ? 'grabbing' : 'grab' }}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerLeave={handlePointerUp}
    >
      <div
        ref={innerRef}
        className="flex gap-6"
        style={{ willChange: 'transform' }}
      >
        {children}
      </div>
    </div>
  );
}
