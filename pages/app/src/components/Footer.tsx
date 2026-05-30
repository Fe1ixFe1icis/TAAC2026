export default function Footer() {
  return (
    <footer className="w-full h-[120px] border-t border-white/5 flex items-center bg-dark-bg">
      <div className="max-w-[1400px] mx-auto px-[5vw] w-full flex flex-col md:flex-row items-center justify-between gap-4">
        <span className="font-mono text-sm text-text-secondary">
          TAAC 2026 &middot; TokenFormer
        </span>
        <span className="text-sm text-text-secondary">
          组会汇报 README
        </span>
        <span className="text-sm text-text-secondary">
          Generated with ❤️ for Research
        </span>
      </div>
    </footer>
  );
}
