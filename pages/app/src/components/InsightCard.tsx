interface InsightCardProps {
  number: string;
  title: string;
  description: string;
  accentColor: string;
}

export default function InsightCard({ number, title, description, accentColor }: InsightCardProps) {
  return (
    <div
      className="glass-card p-8 md:p-10"
      style={{ borderLeft: `3px solid ${accentColor}` }}
    >
      <p
        className="metric-number mb-4"
        style={{ color: accentColor }}
      >
        {number}
      </p>
      <h3 className="text-xl font-medium text-white mb-3">{title}</h3>
      <p className="text-text-secondary text-base leading-relaxed">{description}</p>
    </div>
  );
}
