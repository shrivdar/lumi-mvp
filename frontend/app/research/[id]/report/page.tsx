"use client";

import { useParams, useRouter } from "next/navigation";
import { ArrowLeft, Download, Network, GitBranch, Clock, Zap, FileText } from "lucide-react";
import Link from "next/link";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useFetch } from "@/lib/hooks";
import type { ResearchResult } from "@/lib/types";
import { cn } from "@/lib/utils";

export default function ReportPage() {
  const params = useParams();
  const router = useRouter();
  const id = params.id as string;

  const { data: result, loading, error } = useFetch<ResearchResult>(
    `/api/v1/research/${id}/result`,
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="h-6 w-6 animate-spin rounded-full border-2 border-gray-600 border-t-pathway" />
      </div>
    );
  }

  if (error || !result) {
    return (
      <div className="mx-auto max-w-4xl px-6 py-8">
        <div className="rounded-xl border border-red-900/50 bg-red-950/30 p-4 text-sm text-red-400">
          {error ?? "Report not available. Research may still be in progress."}
        </div>
      </div>
    );
  }

  const durationMin = Math.round(result.total_duration_ms / 60000);

  return (
    <div className="mx-auto max-w-4xl px-6 py-8">
      {/* Header */}
      <div className="mb-6 flex items-center gap-3">
        <button
          onClick={() => router.push(`/research/${id}`)}
          className="rounded-lg p-1.5 text-gray-500 hover:bg-gray-800 hover:text-gray-300"
        >
          <ArrowLeft className="h-4 w-4" />
        </button>
        <div className="flex-1">
          <h1 className="text-lg font-semibold text-white">Research Report</h1>
        </div>
        <Link
          href={`/research/${id}/graph`}
          className="flex items-center gap-1.5 rounded-lg border border-gray-700 px-3 py-1.5 text-xs text-gray-300 hover:bg-gray-800"
        >
          <Network className="h-3.5 w-3.5" /> View Graph
        </Link>
      </div>

      {/* Stats */}
      <div className="mb-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <ReportStat icon={Clock} label="Duration" value={`${durationMin}m`} />
        <ReportStat
          icon={GitBranch}
          label="Hypotheses"
          value={result.hypothesis_ranking.length}
        />
        <ReportStat icon={Zap} label="LLM Calls" value={result.total_llm_calls} />
        <ReportStat
          icon={FileText}
          label="Key Findings"
          value={result.key_findings.length}
        />
      </div>

      {/* Best hypothesis */}
      {result.best_hypothesis && (
        <div className="mb-6 rounded-xl border border-pathway/30 bg-pathway/5 p-4">
          <h2 className="mb-1 text-xs font-medium uppercase tracking-wider text-pathway">
            Best Hypothesis
          </h2>
          <p className="mb-2 text-sm text-gray-200">{result.best_hypothesis.hypothesis}</p>
          <div className="flex gap-4 text-[10px] text-gray-500">
            <span>
              Confidence: {(result.best_hypothesis.confidence * 100).toFixed(0)}%
            </span>
            <span>Visits: {result.best_hypothesis.visit_count}</span>
            <span>UCB: {result.best_hypothesis.ucb_score.toFixed(2)}</span>
          </div>
        </div>
      )}

      {/* Recommended experiments */}
      {result.recommended_experiments.length > 0 && (
        <div className="mb-6 rounded-xl border border-gray-800 bg-gray-900 p-4">
          <h2 className="mb-2 text-xs font-medium uppercase tracking-wider text-gray-500">
            Recommended Experiments
          </h2>
          <ul className="space-y-1">
            {result.recommended_experiments.map((exp, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-gray-300">
                <span className="mt-1 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-experiment" />
                {exp}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Contradictions */}
      {result.contradictions.length > 0 && (
        <div className="mb-6 rounded-xl border border-red-900/30 bg-red-950/10 p-4">
          <h2 className="mb-2 text-xs font-medium uppercase tracking-wider text-red-400">
            Contradictions Found ({result.contradictions.length})
          </h2>
          <div className="space-y-2">
            {result.contradictions.slice(0, 5).map(([e1, e2], i) => (
              <div key={i} className="text-xs text-gray-400">
                <span className="text-gray-300">{e1.relation}</span> vs{" "}
                <span className="text-gray-300">{e2.relation}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Markdown report */}
      {result.report_markdown && (
        <div className="rounded-xl border border-gray-800 bg-gray-900 p-6">
          <article className="prose prose-sm prose-invert max-w-none prose-headings:text-gray-100 prose-p:text-gray-300 prose-a:text-protein prose-strong:text-gray-200 prose-code:text-pathway prose-pre:bg-gray-950 prose-pre:border prose-pre:border-gray-800 prose-li:text-gray-300 prose-table:text-gray-300 prose-th:text-gray-200 prose-td:border-gray-800 prose-th:border-gray-700">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {result.report_markdown}
            </ReactMarkdown>
          </article>
        </div>
      )}

      {/* KG stats */}
      {Object.keys(result.kg_stats).length > 0 && (
        <div className="mt-6 rounded-xl border border-gray-800 bg-gray-900 p-4">
          <h2 className="mb-2 text-xs font-medium uppercase tracking-wider text-gray-500">
            Knowledge Graph Statistics
          </h2>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            {Object.entries(result.kg_stats).map(([k, v]) => (
              <div key={k} className="text-center">
                <p className="text-lg font-bold text-white">{v}</p>
                <p className="text-[10px] text-gray-500">{k.replace(/_/g, " ")}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function ReportStat({
  icon: Icon,
  label,
  value,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string | number;
}) {
  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900 px-3 py-2">
      <div className="flex items-center gap-1.5 text-[10px] text-gray-500">
        <Icon className="h-3 w-3" />
        {label}
      </div>
      <p className="mt-0.5 text-sm font-semibold text-white">{value}</p>
    </div>
  );
}
