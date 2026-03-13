"use client";

import { useMemo, useState } from "react";
import { Search, Box, Plug, Container, Filter } from "lucide-react";
import { cn } from "@/lib/utils";
import type { ToolRegistryEntry, ToolSourceType } from "@/lib/types";

interface ToolRegistryProps {
  tools: ToolRegistryEntry[];
  className?: string;
}

const SOURCE_CONFIG: Record<ToolSourceType, { label: string; color: string; bg: string }> = {
  NATIVE: { label: "Native", color: "text-blue-400", bg: "bg-blue-500/10 border-blue-500/20" },
  MCP: { label: "MCP", color: "text-purple-400", bg: "bg-purple-500/10 border-purple-500/20" },
  CONTAINER: { label: "Container", color: "text-teal-400", bg: "bg-teal-500/10 border-teal-500/20" },
};

const SOURCE_ICONS: Record<ToolSourceType, React.ComponentType<{ className?: string }>> = {
  NATIVE: Box,
  MCP: Plug,
  CONTAINER: Container,
};

function SourceBadge({ sourceType }: { sourceType: ToolSourceType }) {
  const cfg = SOURCE_CONFIG[sourceType];
  const Icon = SOURCE_ICONS[sourceType];
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium",
        cfg.bg,
        cfg.color,
      )}
    >
      <Icon className="h-3 w-3" />
      {cfg.label}
    </span>
  );
}

export default function ToolRegistry({ tools, className }: ToolRegistryProps) {
  const [search, setSearch] = useState("");
  const [sourceFilter, setSourceFilter] = useState<ToolSourceType | "ALL">("ALL");
  const [categoryFilter, setCategoryFilter] = useState<string>("ALL");

  const categories = useMemo(() => {
    const set = new Set(tools.map((t) => t.category));
    return Array.from(set).sort();
  }, [tools]);

  const filtered = useMemo(() => {
    const q = search.toLowerCase();
    return tools.filter((t) => {
      if (sourceFilter !== "ALL" && t.source_type !== sourceFilter) return false;
      if (categoryFilter !== "ALL" && t.category !== categoryFilter) return false;
      if (q && !t.name.toLowerCase().includes(q) && !t.description.toLowerCase().includes(q)) {
        return false;
      }
      return true;
    });
  }, [tools, search, sourceFilter, categoryFilter]);

  const sourceCounts = useMemo(() => {
    const counts: Record<string, number> = { ALL: tools.length };
    for (const t of tools) {
      counts[t.source_type] = (counts[t.source_type] ?? 0) + 1;
    }
    return counts;
  }, [tools]);

  return (
    <div className={cn("space-y-6", className)}>
      {/* Summary bar */}
      <div className="flex flex-wrap items-center gap-3">
        {(["ALL", "NATIVE", "MCP", "CONTAINER"] as const).map((key) => {
          const count = sourceCounts[key] ?? 0;
          const active = sourceFilter === key;
          return (
            <button
              key={key}
              onClick={() => setSourceFilter(key)}
              className={cn(
                "rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors",
                active
                  ? "border-gray-600 bg-gray-800 text-white"
                  : "border-gray-800 bg-gray-900/50 text-gray-500 hover:border-gray-700 hover:text-gray-300",
              )}
            >
              {key === "ALL" ? "All" : SOURCE_CONFIG[key].label}{" "}
              <span className="ml-1 text-gray-500">{count}</span>
            </button>
          );
        })}
      </div>

      {/* Search and category filter */}
      <div className="flex flex-col gap-3 sm:flex-row">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-500" />
          <input
            type="text"
            placeholder="Search tools..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full rounded-lg border border-gray-800 bg-gray-900/50 py-2 pl-9 pr-3 text-sm text-gray-200 placeholder-gray-600 outline-none focus:border-gray-600 focus:ring-1 focus:ring-gray-600"
          />
        </div>
        <div className="relative">
          <Filter className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-500" />
          <select
            value={categoryFilter}
            onChange={(e) => setCategoryFilter(e.target.value)}
            className="appearance-none rounded-lg border border-gray-800 bg-gray-900/50 py-2 pl-9 pr-8 text-sm text-gray-200 outline-none focus:border-gray-600 focus:ring-1 focus:ring-gray-600"
          >
            <option value="ALL">All Categories</option>
            {categories.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Results count */}
      <p className="text-xs text-gray-500">
        Showing {filtered.length} of {tools.length} tools
      </p>

      {/* Tool grid */}
      {filtered.length === 0 ? (
        <div className="flex flex-col items-center justify-center rounded-xl border border-gray-800 bg-gray-900/30 py-16">
          <Search className="mb-3 h-8 w-8 text-gray-700" />
          <p className="text-sm text-gray-500">No tools match your filters</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
          {filtered.map((tool) => (
            <div
              key={tool.name}
              className={cn(
                "group rounded-xl border bg-gray-900/50 p-4 transition-colors hover:border-gray-600",
                tool.enabled ? "border-gray-800" : "border-gray-800/50 opacity-60",
              )}
            >
              <div className="mb-2 flex items-start justify-between gap-2">
                <h4 className="text-sm font-semibold text-gray-100 group-hover:text-white">
                  {tool.name}
                </h4>
                <span
                  className={cn(
                    "mt-0.5 h-2 w-2 shrink-0 rounded-full",
                    tool.enabled ? "bg-emerald-400" : "bg-gray-600",
                  )}
                  title={tool.enabled ? "Enabled" : "Disabled"}
                />
              </div>
              <p className="mb-3 text-xs leading-relaxed text-gray-500">
                {tool.description}
              </p>
              <div className="flex flex-wrap items-center gap-2">
                <SourceBadge sourceType={tool.source_type} />
                <span className="rounded-full border border-gray-800 bg-gray-900 px-2 py-0.5 text-xs text-gray-400">
                  {tool.category}
                </span>
              </div>
              {tool.mcp_server && (
                <p className="mt-2 truncate text-xs text-gray-600" title={tool.mcp_server}>
                  MCP: {tool.mcp_server}
                </p>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
