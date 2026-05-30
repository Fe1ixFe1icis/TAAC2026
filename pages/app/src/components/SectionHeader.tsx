interface SectionHeaderProps {
  overline: string;
  title: string;
}

export default function SectionHeader({ overline, title }: SectionHeaderProps) {
  return (
    <div className="mb-12 md:mb-16">
      <p className="section-overline mb-4">{overline}</p>
      <h2 className="section-title">{title}</h2>
    </div>
  );
}
