"use client";

import { useState } from "react";
import { Send, Settings2, ChevronDown, ChevronUp } from "lucide-react";
import { apiFetch } from "@/lib/api";
import type { ResearchConfig, ResearchSession } from "@/lib/types";
import { cn } from "@/lib/utils";

interface ResearchFormProps {
  onSubmit?: (session: ResearchSession) => void;
  className?: string;
}

export default function ResearchForm({ onSubmit, className }: ResearchFormProps) {
  const [query, setQuery] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [config, setConfig] = useState<ResearchConfig>({
    max_mcts_iterations: 15,
    max_agents: 8,
    confidence_threshold: 0.7,
    enable_falsification: true,
    enable_hitl: true,
  });

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim() || submitting) return;

    setSubmitting(true);
    setError(null);
    try {
      // Backend returns {research_id, status} on create, not a full session
      const created = await apiFetch<{ research_id: string; status: string }>("/api/v1/research", {
        method: "POST",
        body: JSON.stringify({ query: query.trim(), config }),
      });
      // Fetch the full session object so the dashboard can navigate
      const session = await apiFetch<ResearchSession>(`/api/v1/research/${created.research_id}`);
      setQuery("");
      onSubmit?.(session);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to submit");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className={cn("space-y-3", className)}>
      <div className="flex gap-2">
        <div className="relative flex-1">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter a biomedical research question... (e.g., 'Role of B7-H3 in NSCLC immunotherapy resistance')"
            className="w-full rounded-xl border border-gray-700 bg-gray-900 px-4 py-3 pr-12 text-sm text-gray-100 placeholder-gray-500 transition-colors focus:border-pathway focus:outline-none focus:ring-1 focus:ring-pathway"
            disabled={submitting}
          />
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="absolute right-2 top-1/2 -translate-y-1/2 rounded-lg p-1.5 text-gray-500 transition-colors hover:bg-gray-800 hover:text-gray-300"
          >
            <Settings2 className="h-4 w-4" />
          </button>
        </div>
        <button
          type="submit"
          disabled={!query.trim() || submitting}
          className="flex items-center gap-2 rounded-xl bg-pathway px-5 py-3 text-sm font-medium text-white transition-all hover:bg-pathway/90 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <Send className="h-4 w-4" />
          {submitting ? "Launching..." : "Research"}
        </button>
      </div>

      {error && (
        <p className="text-sm text-red-400">{error}</p>
      )}

      {showAdvanced && (
        <div className="rounded-xl border border-gray-800 bg-gray-900/50 p-4">
          <div className="mb-3 flex items-center gap-2 text-xs font-medium uppercase tracking-wider text-gray-500">
            <Settings2 className="h-3 w-3" />
            Advanced Configuration
          </div>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
            <label className="space-y-1">
              <span className="text-xs text-gray-400">MCTS Iterations</span>
              <input
                type="number"
                min={1}
                max={50}
                value={config.max_mcts_iterations}
                onChange={(e) =>
                  setConfig({ ...config, max_mcts_iterations: +e.target.value })
                }
                className="w-full rounded-lg border border-gray-700 bg-gray-800 px-3 py-1.5 text-sm text-gray-200"
              />
            </label>
            <label className="space-y-1">
              <span className="text-xs text-gray-400">Max Agents</span>
              <input
                type="number"
                min={1}
                max={20}
                value={config.max_agents}
                onChange={(e) =>
                  setConfig({ ...config, max_agents: +e.target.value })
                }
                className="w-full rounded-lg border border-gray-700 bg-gray-800 px-3 py-1.5 text-sm text-gray-200"
              />
            </label>
            <label className="space-y-1">
              <span className="text-xs text-gray-400">Confidence Threshold</span>
              <input
                type="number"
                min={0}
                max={1}
                step={0.05}
                value={config.confidence_threshold}
                onChange={(e) =>
                  setConfig({ ...config, confidence_threshold: +e.target.value })
                }
                className="w-full rounded-lg border border-gray-700 bg-gray-800 px-3 py-1.5 text-sm text-gray-200"
              />
            </label>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={config.enable_falsification}
                onChange={(e) =>
                  setConfig({ ...config, enable_falsification: e.target.checked })
                }
                className="rounded border-gray-600 bg-gray-800"
              />
              <span className="text-xs text-gray-400">Self-Falsification</span>
            </label>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={config.enable_hitl}
                onChange={(e) =>
                  setConfig({ ...config, enable_hitl: e.target.checked })
                }
                className="rounded border-gray-600 bg-gray-800"
              />
              <span className="text-xs text-gray-400">Human-in-the-Loop</span>
            </label>
          </div>
        </div>
      )}
    </form>
  );
}
