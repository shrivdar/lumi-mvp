"use client";

import { BarChart3, Loader2 } from "lucide-react";
import { useFetch } from "@/lib/hooks";
import type { BenchmarkRun } from "@/lib/types";
import BenchmarkChart from "@/components/benchmark-chart";

export default function BenchmarksPage() {
  const { data, loading, error } = useFetch<BenchmarkRun[]>("/api/v1/benchmarks");

  return (
    <div className="mx-auto max-w-7xl px-6 py-8">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3">
          <BarChart3 className="h-6 w-6 text-emerald-400" />
          <h1 className="text-2xl font-bold text-white">Benchmark Results</h1>
        </div>
        <p className="mt-2 text-sm text-gray-400">
          Compare YOHAS 3.0 accuracy against established baselines across
          standardized biomedical research benchmarks.
        </p>
      </div>

      {/* Loading */}
      {loading && (
        <div className="flex flex-col items-center justify-center py-24">
          <Loader2 className="mb-3 h-8 w-8 animate-spin text-gray-500" />
          <p className="text-sm text-gray-500">Loading benchmark data...</p>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="rounded-xl border border-red-900/50 bg-red-950/20 px-6 py-4">
          <p className="text-sm text-red-400">Failed to load benchmarks: {error}</p>
        </div>
      )}

      {/* Empty */}
      {!loading && !error && data && data.length === 0 && (
        <div className="flex flex-col items-center justify-center rounded-xl border border-gray-800 bg-gray-900/30 py-24">
          <BarChart3 className="mb-3 h-10 w-10 text-gray-700" />
          <p className="text-sm text-gray-500">No benchmark runs yet</p>
          <p className="mt-1 text-xs text-gray-600">
            Run a benchmark suite to see results here.
          </p>
        </div>
      )}

      {/* Chart + run cards */}
      {data && data.length > 0 && (
        <div className="space-y-8">
          <BenchmarkChart runs={data} />

          {/* Individual run cards */}
          <div>
            <h2 className="mb-4 text-lg font-semibold text-gray-200">
              Individual Runs
            </h2>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
              {data.map((run) => (
                <div
                  key={run.id}
                  className="rounded-xl border border-gray-800 bg-gray-900/50 p-5"
                >
                  <h3 className="mb-1 text-sm font-semibold text-gray-100">
                    {run.benchmark_name}
                  </h3>
                  <p className="mb-3 text-xs text-gray-500">v{run.version}</p>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div>
                      <span className="text-gray-500">Accuracy</span>
                      <p className="font-medium text-emerald-400">
                        {(run.accuracy * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div>
                      <span className="text-gray-500">Tasks</span>
                      <p className="font-medium text-gray-200">
                        {run.correct_tasks}/{run.total_tasks}
                      </p>
                    </div>
                    <div>
                      <span className="text-gray-500">Started</span>
                      <p className="font-medium text-gray-300">
                        {new Date(run.started_at).toLocaleDateString()}
                      </p>
                    </div>
                    <div>
                      <span className="text-gray-500">Baselines</span>
                      <p className="font-medium text-gray-300">
                        {Object.keys(run.baseline_comparison).length}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
