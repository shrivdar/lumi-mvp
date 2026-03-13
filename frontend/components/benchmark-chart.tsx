"use client";

import { useEffect, useState } from "react";
import { TrendingUp, Target, Clock, CheckCircle } from "lucide-react";
import { cn } from "@/lib/utils";
import type { BenchmarkRun } from "@/lib/types";

interface BenchmarkChartProps {
  runs: BenchmarkRun[];
  className?: string;
}

interface ChartEntry {
  label: string;
  value: number;
  isYohas: boolean;
}

function groupedChartData(runs: BenchmarkRun[]): Record<string, ChartEntry[]> {
  const groups: Record<string, ChartEntry[]> = {};

  for (const run of runs) {
    const key = run.benchmark_name;
    const entries: ChartEntry[] = [
      { label: "YOHAS 3.0", value: run.accuracy * 100, isYohas: true },
    ];
    for (const [name, acc] of Object.entries(run.baseline_comparison)) {
      entries.push({ label: name, value: acc * 100, isYohas: false });
    }
    // Sort: YOHAS first, then baselines descending
    entries.sort((a, b) => {
      if (a.isYohas) return -1;
      if (b.isYohas) return 1;
      return b.value - a.value;
    });
    groups[key] = entries;
  }

  return groups;
}

function Bar({
  entry,
  maxValue,
  animate,
}: {
  entry: ChartEntry;
  maxValue: number;
  animate: boolean;
}) {
  const pct = maxValue > 0 ? (entry.value / maxValue) * 100 : 0;

  return (
    <div className="flex items-center gap-3">
      <span className="w-32 shrink-0 truncate text-right text-sm text-gray-400">
        {entry.label}
      </span>
      <div className="relative h-7 flex-1 overflow-hidden rounded-md bg-gray-800/60">
        <div
          className={cn(
            "absolute inset-y-0 left-0 rounded-md transition-all duration-1000 ease-out",
            entry.isYohas
              ? "bg-gradient-to-r from-emerald-500 to-cyan-400 shadow-[0_0_12px_rgba(16,185,129,0.4)]"
              : "bg-gray-600",
          )}
          style={{ width: animate ? `${pct}%` : "0%" }}
        />
        <span
          className={cn(
            "relative z-10 flex h-full items-center px-2 text-xs font-medium",
            entry.isYohas ? "text-white" : "text-gray-300",
          )}
        >
          {entry.value.toFixed(1)}%
        </span>
      </div>
    </div>
  );
}

export default function BenchmarkChart({ runs, className }: BenchmarkChartProps) {
  const [animate, setAnimate] = useState(false);

  useEffect(() => {
    const id = requestAnimationFrame(() => setAnimate(true));
    return () => cancelAnimationFrame(id);
  }, []);

  if (runs.length === 0) return null;

  const groups = groupedChartData(runs);

  // Aggregate metrics across all runs
  const totalTasks = runs.reduce((s, r) => s + r.total_tasks, 0);
  const totalCorrect = runs.reduce((s, r) => s + r.correct_tasks, 0);
  const avgAccuracy =
    runs.reduce((s, r) => s + r.accuracy, 0) / runs.length;

  const allMetrics = runs.flatMap((r) => r.metrics);

  return (
    <div className={cn("space-y-8", className)}>
      {/* Chart sections grouped by benchmark name */}
      {Object.entries(groups).map(([name, entries]) => {
        const maxValue = Math.max(...entries.map((e) => e.value), 100);

        return (
          <div key={name} className="rounded-xl border border-gray-800 bg-gray-900/50 p-6">
            <h3 className="mb-4 text-sm font-semibold uppercase tracking-wider text-gray-300">
              {name}
            </h3>
            <div className="space-y-2">
              {entries.map((entry) => (
                <Bar
                  key={entry.label}
                  entry={entry}
                  maxValue={maxValue}
                  animate={animate}
                />
              ))}
            </div>
          </div>
        );
      })}

      {/* Aggregate stats */}
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <StatCard
          icon={<Target className="h-4 w-4 text-emerald-400" />}
          label="Avg Accuracy"
          value={`${(avgAccuracy * 100).toFixed(1)}%`}
        />
        <StatCard
          icon={<CheckCircle className="h-4 w-4 text-cyan-400" />}
          label="Correct / Total"
          value={`${totalCorrect} / ${totalTasks}`}
        />
        <StatCard
          icon={<TrendingUp className="h-4 w-4 text-purple-400" />}
          label="Benchmark Runs"
          value={String(runs.length)}
        />
        <StatCard
          icon={<Clock className="h-4 w-4 text-amber-400" />}
          label="Tracked Metrics"
          value={String(allMetrics.length)}
        />
      </div>

      {/* Per-run metrics */}
      {allMetrics.length > 0 && (
        <div className="rounded-xl border border-gray-800 bg-gray-900/50 p-6">
          <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-gray-300">
            Detailed Metrics
          </h3>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
            {allMetrics.map((m, i) => (
              <div
                key={`${m.name}-${i}`}
                className="rounded-lg border border-gray-800 bg-gray-900 px-3 py-2"
              >
                <p className="text-xs text-gray-500">{m.name}</p>
                <p className="text-sm font-medium text-gray-200">
                  {m.value.toFixed(2)}{" "}
                  <span className="text-xs text-gray-500">{m.unit}</span>
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function StatCard({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="flex items-center gap-3 rounded-xl border border-gray-800 bg-gray-900/50 px-4 py-3">
      {icon}
      <div>
        <p className="text-xs text-gray-500">{label}</p>
        <p className="text-sm font-semibold text-gray-100">{value}</p>
      </div>
    </div>
  );
}
