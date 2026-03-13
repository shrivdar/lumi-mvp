"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Beaker,
  BarChart3,
  Wrench,
  Home,
  Dna,
} from "lucide-react";
import { cn } from "@/lib/utils";

const NAV_ITEMS = [
  { href: "/", label: "Dashboard", icon: Home },
  { href: "/benchmarks", label: "Benchmarks", icon: BarChart3 },
  { href: "/tools", label: "Tools", icon: Wrench },
] as const;

export default function Nav() {
  const pathname = usePathname();

  return (
    <nav className="sticky top-0 z-50 border-b border-gray-800 bg-gray-950/80 backdrop-blur-md">
      <div className="mx-auto flex h-14 max-w-7xl items-center gap-6 px-6">
        <Link href="/" className="flex items-center gap-2 font-semibold text-white">
          <Dna className="h-5 w-5 text-pathway" />
          <span>YOHAS 3.0</span>
        </Link>

        <div className="flex items-center gap-1">
          {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
            const active =
              href === "/" ? pathname === "/" : pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                className={cn(
                  "flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm transition-colors",
                  active
                    ? "bg-gray-800 text-white"
                    : "text-gray-400 hover:bg-gray-900 hover:text-gray-200",
                )}
              >
                <Icon className="h-4 w-4" />
                {label}
              </Link>
            );
          })}
        </div>

        <div className="ml-auto flex items-center gap-2">
          <span className="flex items-center gap-1.5 rounded-full bg-gray-800 px-3 py-1 text-xs text-gray-400">
            <Beaker className="h-3 w-3" />
            Autonomous Research Platform
          </span>
        </div>
      </div>
    </nav>
  );
}
