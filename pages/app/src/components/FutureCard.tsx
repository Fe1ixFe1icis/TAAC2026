interface FutureCardProps {
  icon: string;
  title: string;
  description: string;
}

export default function FutureCard({ icon, title, description }: FutureCardProps) {
  return (
    <div className="glass-card p-8 md:p-10 flex-shrink-0" style={{ width: '380px' }}>
      <span className="text-4xl mb-6 block">{icon}</span>
      <h3 className="text-2xl font-medium text-white mb-4">{title}</h3>
      <p className="text-text-secondary text-base leading-relaxed">{description}</p>
    </div>
  );
}
