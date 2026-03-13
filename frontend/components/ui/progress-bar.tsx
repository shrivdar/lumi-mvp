"use client";

import { cn } from "@/lib/utils";

interface ProgressBarProps {
  value: number;
  label?: string;
  showValue?: boolean;
  color?: string;
  size?: "sm" | "md" | "lg";
  animated?: boolean;
  className?: string;
}

const sizeStyles = {
  sm: "h-1.5",
  md: "h-2.5",
  lg: "h-4",
};

function interpolateColor(value: number): string {
  const clamped = Math.max(0, Math.min(1, value));
  if (clamped < 0.5) {
    const t = clamped / 0.5;
    const r = 239;
    const g = Math.round(68 + t * (178 - 68));
    const b = Math.round(68 + t * (57 - 68));
    return `rgb(${r}, ${g}, ${b})`;
  }
  const t = (clamped - 0.5) / 0.5;
  const r = Math.round(239 - t * (239 - 34));
  const g = Math.round(178 + t * (197 - 178));
  const b = Math.round(57 + t * (94 - 57));
  return `rgb(${r}, ${g}, ${b})`;
}

export function ProgressBar({
  value,
  label,
  showValue,
  color,
  size = "md",
  animated,
  className,
}: ProgressBarProps) {
  const clamped = Math.max(0, Math.min(1, value));
  const barColor = color ?? interpolateColor(clamped);
  const isTailwind = color && !color.startsWith("#") && !color.startsWith("rgb");

  return (
    <div className={cn("w-full", className)}>
      {(label || showValue) && (
        <div className="mb-1 flex items-center justify-between text-xs text-gray-400">
          {label && <span>{label}</span>}
          {showValue && <span>{Math.round(clamped * 100)}%</span>}
        </div>
      )}
      <div
        className={cn(
          "w-full overflow-hidden rounded-full bg-gray-800",
          sizeStyles[size],
        )}
      >
        <div
          className={cn(
            "h-full rounded-full transition-all duration-500 ease-out",
            isTailwind && color,
            animated && "bg-[length:200%_100%] animate-shimmer",
          )}
          style={{
            width: `${clamped * 100}%`,
            ...(!isTailwind && {
              backgroundColor: barColor,
            }),
            ...(animated &&
              !isTailwind && {
                backgroundImage: `linear-gradient(90deg, ${barColor}, ${barColor}dd, ${barColor}, ${barColor}dd, ${barColor})`,
                backgroundSize: "200% 100%",
              }),
          }}
        />
      </div>
    </div>
  );
}
