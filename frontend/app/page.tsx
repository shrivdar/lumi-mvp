"use client";

import { useRouter } from "next/navigation";
import {
  Network,
  GitBranch,
  Zap,
  Beaker,
  Search,
  FlaskConical,
} from "lucide-react";
import { useResearchSessions } from "@/lib/hooks";
import type { ResearchSession } from "@/lib/types";
import ResearchForm from "@/components/research-form";
import ResearchCard from "@/components/research-card";

export default function DashboardPage() {
  const router = useRouter();
  const { data, loading, error } = useResearchSessions();

  const sessions = data?.items ?? [];
  const active = sessions.filter(
    (s) => s.status === "RUNNING" || s.status === "INITIALIZING",
  );
  const completed = sessions.filter((s) => s.status === "COMPLETED");
  const totalNodes = sessions.reduce((a, s) => a + s.total_nodes, 0);
  const totalHypotheses = sessions.reduce((a, s) => a + s.total_hypotheses, 0);

  function handleNewSession(session: ResearchSession) {
    router.push(`/research/${session.id}`);
  }

  return (
    <div className="mx-auto max-w-7xl px-6 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="mb-2 text-2xl font-bold text-white">Research Dashboard</h1>
        <p className="text-sm text-gray-400">
          Launch autonomous research sessions, monitor agent swarms, and explore knowledge graphs.
        </p>
      </div>

      {/* New Research */}
      <div className="mb-8">
        <ResearchForm onSubmit={handleNewSession} />
      </div>

      {/* Stats row */}
      <div className="mb-8 grid grid-cols-2 gap-4 sm:grid-cols-4">
        <StatCard
          icon={FlaskConical}
          label="Active Sessions"
          value={active.length}
          color="text-pathway"
        />
        <StatCard
          icon={Beaker}
          label="Completed"
          value={completed.length}
          color="text-clinical"
        />
        <StatCard
          icon={Network}
          label="Total KG Nodes"
          value={totalNodes}
          color="text-protein"
        />
        <StatCard
          icon={GitBranch}
          label="Hypotheses Explored"
          value={totalHypotheses}
          color="text-gene"
        />
      </div>

      {/* Sessions List */}
      {loading && sessions.length === 0 && (
        <div className="flex items-center justify-center py-20">
          <div className="h-6 w-6 animate-spin rounded-full border-2 border-gray-600 border-t-pathway" />
        </div>
      )}

      {error && (
        <div className="rounded-xl border border-red-900/50 bg-red-950/30 p-4 text-sm text-red-400">
          {error}
        </div>
      )}

      {!loading && sessions.length === 0 && (
        <div className="flex flex-col items-center justify-center py-20 text-center">
          <Search className="mb-4 h-12 w-12 text-gray-700" />
          <h2 className="mb-2 text-lg font-medium text-gray-300">No research sessions yet</h2>
          <p className="text-sm text-gray-500">
            Enter a biomedical research question above to launch your first autonomous investigation.
          </p>
        </div>
      )}

      {active.length > 0 && (
        <section className="mb-8">
          <h2 className="mb-4 flex items-center gap-2 text-sm font-medium uppercase tracking-wider text-gray-500">
            <span className="h-2 w-2 animate-pulse rounded-full bg-pathway" />
            Active Research
          </h2>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {active.map((s) => (
              <ResearchCard key={s.id} session={s} />
            ))}
          </div>
        </section>
      )}

      {sessions.filter((s) => s.status !== "RUNNING" && s.status !== "INITIALIZING").length > 0 && (
        <section>
          <h2 className="mb-4 text-sm font-medium uppercase tracking-wider text-gray-500">
            Past Research
          </h2>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {sessions
              .filter((s) => s.status !== "RUNNING" && s.status !== "INITIALIZING")
              .map((s) => (
                <ResearchCard key={s.id} session={s} />
              ))}
          </div>
        </section>
      )}
    </div>
  );
}

function StatCard({
  icon: Icon,
  label,
  value,
  color,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: number;
  color: string;
}) {
  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900 p-4">
      <div className="mb-2 flex items-center gap-2">
        <Icon className={`h-4 w-4 ${color}`} />
        <span className="text-xs text-gray-500">{label}</span>
      </div>
      <p className="text-2xl font-bold text-white">{value.toLocaleString()}</p>
    </div>
  );
}
