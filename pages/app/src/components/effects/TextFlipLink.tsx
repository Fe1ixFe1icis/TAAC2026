interface TextFlipLinkProps {
  href: string;
  children: string;
  onClick?: () => void;
}

export default function TextFlipLink({ href, children, onClick }: TextFlipLinkProps) {
  return (
    <a href={href} onClick={onClick} className="flip-link">
      <span className="flip-link-inner relative">
        <span className="flip-link-text">{children}</span>
        <span className="flip-link-text-back">{children}</span>
      </span>
    </a>
  );
}
