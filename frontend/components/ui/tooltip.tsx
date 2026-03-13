"use client";

import { cn } from "@/lib/utils";

interface TooltipProps {
  content: string;
  children: React.ReactNode;
  position?: "top" | "bottom";
}

export function Tooltip({ content, children, position = "top" }: TooltipProps) {
  return (
    <span className="group relative inline-flex">
      {children}
      <span
        className={cn(
          "pointer-events-none absolute left-1/2 z-50 -translate-x-1/2 scale-95 whitespace-nowrap rounded-md bg-gray-800 px-2.5 py-1.5 text-xs text-gray-200 opacity-0 shadow-lg transition-all group-hover:scale-100 group-hover:opacity-100",
          position === "top" && "bottom-full mb-2",
          position === "bottom" && "top-full mt-2",
        )}
      >
        {content}
      </span>
    </span>
  );
}
