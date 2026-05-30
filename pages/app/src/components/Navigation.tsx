import { useEffect, useRef, useState } from 'react';
import TextFlipLink from './effects/TextFlipLink';
import { useScrollTo } from '@/hooks/useLenis';
import { Menu, X } from 'lucide-react';

const navLinks = [
  { label: '概述', target: '#overview' },
  { label: '架构', target: '#architecture' },
  { label: '实验', target: '#results' },
  { label: '历程', target: '#journey' },
  { label: '洞察', target: '#insights' },
  { label: '展望', target: '#future' },
];

export default function Navigation() {
  const scrollTo = useScrollTo();
  const [progress, setProgress] = useState(0);
  const [visible, setVisible] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const progressRef = useRef(0);

  useEffect(() => {
    // Entrance animation
    const timer = setTimeout(() => setVisible(true), 1000);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    const handleScroll = () => {
      const scrollTop = window.scrollY;
      const docHeight = document.documentElement.scrollHeight - window.innerHeight;
      const newProgress = docHeight > 0 ? scrollTop / docHeight : 0;
      progressRef.current = newProgress;
      setProgress(newProgress);
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll();
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleNavClick = (target: string) => {
    scrollTo(target);
    setMobileOpen(false);
  };

  return (
    <nav
      className={`fixed top-0 left-0 w-full z-50 transition-all duration-600 ${
        visible ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-5'
      }`}
      style={{
        background: 'rgba(5, 5, 5, 0.7)',
        backdropFilter: 'blur(20px) saturate(1.2)',
        WebkitBackdropFilter: 'blur(20px) saturate(1.2)',
        borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
      }}
    >
      <div className="max-w-[1400px] mx-auto px-[5vw] h-16 flex items-center justify-between">
        {/* Logo */}
        <button
          onClick={() => handleNavClick('#hero')}
          className="font-mono text-sm font-medium text-white tracking-wider hover:text-gradient transition-colors"
        >
          TF
        </button>

        {/* Desktop Nav Links */}
        <div className="hidden md:flex items-center gap-8">
          {navLinks.map((link) => (
            <TextFlipLink
              key={link.target}
              href={link.target}
              onClick={() => handleNavClick(link.target)}
            >
              {link.label}
            </TextFlipLink>
          ))}
        </div>

        {/* Mobile Menu Button */}
        <button
          className="md:hidden text-white p-2"
          onClick={() => setMobileOpen(!mobileOpen)}
        >
          {mobileOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
      </div>

      {/* Mobile Menu */}
      {mobileOpen && (
        <div
          className="md:hidden px-[5vw] pb-6"
          style={{ background: 'rgba(5, 5, 5, 0.95)' }}
        >
          <div className="flex flex-col gap-4">
            {navLinks.map((link) => (
              <button
                key={link.target}
                onClick={() => handleNavClick(link.target)}
                className="text-left text-text-secondary hover:text-white transition-colors text-sm uppercase tracking-widest font-mono"
              >
                {link.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Scroll Progress Bar */}
      <div
        className="absolute bottom-0 left-0 h-[2px]"
        style={{
          width: `${progress * 100}%`,
          background: 'linear-gradient(90deg, #667eea, #764ba2)',
          transition: 'width 0.1s linear',
        }}
      />
    </nav>
  );
}
