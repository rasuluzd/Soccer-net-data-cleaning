import Image from "next/image";

export interface LogoProps {
  className?: string;
}

export default function Logo({ className = "h-8 w-auto" }: LogoProps) {
  return (
    <Image
      src="/forzasearch white.png"
      alt="ForzaSearch"
      width={200}
      height={32}
      className={className}
      priority
    />
  );
}
