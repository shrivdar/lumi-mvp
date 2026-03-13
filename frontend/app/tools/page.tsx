"use client";

import { Wrench, Loader2 } from "lucide-react";
import { useFetch } from "@/lib/hooks";
import type { ToolRegistryEntry } from "@/lib/types";
import ToolRegistry from "@/components/tool-registry";

export default function ToolsPage() {
  const { data, loading, error } = useFetch<ToolRegistryEntry[]>("/api/v1/tools");

  return (
    <div className="mx-auto max-w-7xl px-6 py-8">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3">
          <Wrench className="h-6 w-6 text-cyan-400" />
          <h1 className="text-2xl font-bold text-white">Tool Registry</h1>
          {data && (
            <span className="rounded-full bg-gray-800 px-2.5 py-0.5 text-xs font-medium text-gray-400">
              {data.length} tools
            </span>
          )}
        </div>
        <p className="mt-2 text-sm text-gray-400">
          Browse all registered tools available to YOHAS agents, including native
          integrations, MCP servers, and containerized tools.
        </p>
      </div>

      {/* Loading */}
      {loading && (
        <div className="flex flex-col items-center justify-center py-24">
          <Loader2 className="mb-3 h-8 w-8 animate-spin text-gray-500" />
          <p className="text-sm text-gray-500">Loading tool registry...</p>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="rounded-xl border border-red-900/50 bg-red-950/20 px-6 py-4">
          <p className="text-sm text-red-400">Failed to load tools: {error}</p>
        </div>
      )}

      {/* Empty */}
      {!loading && !error && data && data.length === 0 && (
        <div className="flex flex-col items-center justify-center rounded-xl border border-gray-800 bg-gray-900/30 py-24">
          <Wrench className="mb-3 h-10 w-10 text-gray-700" />
          <p className="text-sm text-gray-500">No tools registered</p>
          <p className="mt-1 text-xs text-gray-600">
            Configure tools in the backend to see them here.
          </p>
        </div>
      )}

      {/* Registry */}
      {data && data.length > 0 && <ToolRegistry tools={data} />}
    </div>
  );
}
