interface MetricCardProps {
  value: string;
  label: string;
  isAccent?: boolean;
}

export default function MetricCard({ value, label, isAccent = false }: MetricCardProps) {
  return (
    <div className="glass-card p-6 md:p-8 flex flex-col justify-center">
      <p className={`metric-number ${isAccent ? 'metric-number-accent' : 'text-white'}`}>
        {value}
      </p>
      <p className="text-text-muted text-xs uppercase tracking-widest mt-3 font-mono">
        {label}
      </p>
    </div>
  );
}
