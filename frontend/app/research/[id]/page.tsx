"use client";

import { useParams, useRouter } from "next/navigation";
import { useState } from "react";
import {
  ArrowLeft,
  Network,
  FileText,
  GitBranch,
  Zap,
  Clock,
  Users,
  XCircle,
  Wifi,
  WifiOff,
} from "lucide-react";
import Link from "next/link";
import { useResearchSession, useResearchWebSocket, useFetch } from "@/lib/hooks";
import { apiFetch } from "@/lib/api";
import type { AgentInfo, HypothesisNode, ResearchEvent } from "@/lib/types";
import { STATUS_COLORS, AGENT_COLORS, AGENT_LABELS } from "@/lib/types";
import { cn } from "@/lib/utils";

export default function ResearchDetailPage() {
  const params = useParams();
  const router = useRouter();
  const id = params.id as string;

  const { data: session, error } = useResearchSession(id);
  const { events, connected } = useResearchWebSocket(
    session?.status === "RUNNING" || session?.status === "INITIALIZING" ? id : null,
  );
  const { data: agents } = useFetch<AgentInfo[]>(`/api/v1/research/${id}/agents`);
  const { data: hypotheses } = useFetch<HypothesisNode[]>(`/api/v1/research/${id}/hypotheses`);

  const [activeTab, setActiveTab] = useState<"feed" | "agents" | "hypotheses">("feed");

  if (error) {
    return (
      <div className="mx-auto max-w-7xl px-6 py-8">
        <div className="rounded-xl border border-red-900/50 bg-red-950/30 p-4 text-sm text-red-400">
          {error}
        </div>
      </div>
    );
  }

  if (!session) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="h-6 w-6 animate-spin rounded-full border-2 border-gray-600 border-t-pathway" />
      </div>
    );
  }

  const isActive = session.status === "RUNNING" || session.status === "INITIALIZING";
  const statusColor = STATUS_COLORS[session.status];

  async function handleCancel() {
    await apiFetch(`/api/v1/research/${id}/cancel`, { method: "POST" });
  }

  return (
    <div className="mx-auto max-w-7xl px-6 py-8">
      {/* Top bar */}
      <div className="mb-6 flex items-center gap-3">
        <button
          onClick={() => router.push("/")}
          className="rounded-lg p-1.5 text-gray-500 transition-colors hover:bg-gray-800 hover:text-gray-300"
        >
          <ArrowLeft className="h-4 w-4" />
        </button>
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <span
              className="inline-block h-2 w-2 rounded-full"
              style={{ backgroundColor: statusColor }}
            />
            <span className="text-xs font-medium text-gray-400">
              {session.status.replace("_", " ")}
            </span>
            {isActive && connected && (
              <span className="flex items-center gap-1 text-[10px] text-pathway">
                <Wifi className="h-3 w-3" /> Live
              </span>
            )}
            {isActive && !connected && (
              <span className="flex items-center gap-1 text-[10px] text-gray-500">
                <WifiOff className="h-3 w-3" /> Connecting...
              </span>
            )}
          </div>
          <h1 className="mt-1 text-lg font-semibold text-white">{session.query}</h1>
        </div>
        <div className="flex gap-2">
          {isActive && (
            <button
              onClick={handleCancel}
              className="flex items-center gap-1.5 rounded-lg border border-red-900/50 px-3 py-1.5 text-xs text-red-400 transition-colors hover:bg-red-950/30"
            >
              <XCircle className="h-3.5 w-3.5" /> Cancel
            </button>
          )}
          <Link
            href={`/research/${id}/graph`}
            className="flex items-center gap-1.5 rounded-lg border border-gray-700 px-3 py-1.5 text-xs text-gray-300 transition-colors hover:bg-gray-800"
          >
            <Network className="h-3.5 w-3.5" /> Knowledge Graph
          </Link>
          {session.status === "COMPLETED" && (
            <Link
              href={`/research/${id}/report`}
              className="flex items-center gap-1.5 rounded-lg bg-pathway px-3 py-1.5 text-xs font-medium text-white transition-colors hover:bg-pathway/90"
            >
              <FileText className="h-3.5 w-3.5" /> View Report
            </Link>
          )}
        </div>
      </div>

      {/* Stats */}
      <div className="mb-6 grid grid-cols-2 gap-3 sm:grid-cols-5">
        <MiniStat icon={Clock} label="Iteration" value={`${session.current_iteration}/${session.config.max_mcts_iterations ?? "?"}`} />
        <MiniStat icon={Network} label="KG Nodes" value={session.total_nodes} />
        <MiniStat icon={GitBranch} label="Hypotheses" value={session.total_hypotheses} />
        <MiniStat icon={Users} label="Agents" value={agents?.length ?? "-"} />
        <MiniStat icon={Zap} label="Tokens" value={session.total_tokens_used.toLocaleString()} />
      </div>

      {/* Progress bar for active sessions */}
      {isActive && session.config.max_mcts_iterations && (
        <div className="mb-6 h-1.5 overflow-hidden rounded-full bg-gray-800">
          <div
            className="h-full rounded-full bg-gradient-to-r from-pathway to-protein transition-all duration-700"
            style={{
              width: `${Math.min(100, (session.current_iteration / session.config.max_mcts_iterations) * 100)}%`,
            }}
          />
        </div>
      )}

      {/* Tabs */}
      <div className="mb-4 flex gap-1 border-b border-gray-800">
        {(["feed", "agents", "hypotheses"] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={cn(
              "border-b-2 px-4 py-2 text-sm font-medium transition-colors",
              activeTab === tab
                ? "border-pathway text-white"
                : "border-transparent text-gray-500 hover:text-gray-300",
            )}
          >
            {tab === "feed" ? "Live Feed" : tab === "agents" ? "Agents" : "Hypotheses"}
            {tab === "feed" && events.length > 0 && (
              <span className="ml-1.5 rounded-full bg-gray-800 px-1.5 py-0.5 text-[10px] tabular-nums">
                {events.length}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {activeTab === "feed" && <EventFeed events={events} />}
      {activeTab === "agents" && <AgentList agents={agents ?? []} />}
      {activeTab === "hypotheses" && <HypothesisList hypotheses={hypotheses ?? []} />}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function MiniStat({
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

const EVENT_ICONS: Record<string, { color: string; label: string }> = {
  session_created: { color: "#3498DB", label: "Session Created" },
  initialization_started: { color: "#3498DB", label: "Initializing" },
  hypothesis_generated: { color: "#6B5CE7", label: "Hypothesis Generated" },
  hypothesis_selected: { color: "#F39C12", label: "Hypothesis Selected" },
  mcts_iteration_start: { color: "#95A5A6", label: "MCTS Iteration" },
  agents_composed: { color: "#1ABC9C", label: "Swarm Composed" },
  agent_started: { color: "#2ECC71", label: "Agent Started" },
  node_created: { color: "#4A90D9", label: "Node Created" },
  edge_created: { color: "#1ABC9C", label: "Edge Created" },
  edge_falsified: { color: "#E74C3C", label: "Edge Falsified" },
  confidence_updated: { color: "#F39C12", label: "Confidence Updated" },
  uncertainty_aggregated: { color: "#E67E22", label: "Uncertainty Aggregated" },
  hitl_triggered: { color: "#F1C40F", label: "HITL Triggered" },
  hitl_response_received: { color: "#2ECC71", label: "HITL Response" },
  mcts_backpropagation: { color: "#95A5A6", label: "Backpropagation" },
  research_completed: { color: "#2ECC71", label: "Completed" },
};

function EventFeed({ events }: { events: ResearchEvent[] }) {
  if (events.length === 0) {
    return (
      <div className="py-12 text-center text-sm text-gray-500">
        Waiting for events... Events will appear here in real-time as the research progresses.
      </div>
    );
  }

  return (
    <div className="max-h-[500px] space-y-1 overflow-y-auto pr-2">
      {[...events].reverse().map((evt, i) => {
        const meta = EVENT_ICONS[evt.event_type] ?? { color: "#95A5A6", label: evt.event_type };
        const ago = timeAgo(evt.timestamp);
        const detail = extractDetail(evt);

        return (
          <div
            key={i}
            className={cn(
              "flex items-start gap-3 rounded-lg px-3 py-2 transition-colors",
              i === 0 && "bg-gray-900/50",
            )}
          >
            <span
              className="mt-1.5 h-2 w-2 flex-shrink-0 rounded-full"
              style={{ backgroundColor: meta.color }}
            />
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <span className="text-xs font-medium text-gray-300">{meta.label}</span>
                <span className="text-[10px] text-gray-600">{ago}</span>
              </div>
              {detail && (
                <p className="mt-0.5 truncate text-[11px] text-gray-500">{detail}</p>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function extractDetail(evt: ResearchEvent): string | null {
  const d = evt.data;
  if (d.node_name) return `Node: ${d.node_name} (${d.node_type ?? ""})`;
  if (d.relation) return `${d.source_name ?? ""} → ${d.relation} → ${d.target_name ?? ""}`;
  if (d.hypothesis) return String(d.hypothesis).slice(0, 120);
  if (d.agent_type) return `Agent: ${d.agent_type}`;
  if (d.confidence !== undefined) return `Confidence: ${((d.confidence as number) * 100).toFixed(0)}%`;
  if (d.composite !== undefined) return `Uncertainty: ${((d.composite as number) * 100).toFixed(0)}%`;
  return null;
}

function AgentList({ agents }: { agents: AgentInfo[] }) {
  // AGENT_COLORS and AGENT_LABELS imported at top of file

  if (agents.length === 0) {
    return <div className="py-12 text-center text-sm text-gray-500">No agents assigned yet.</div>;
  }

  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {agents.map((a) => (
        <div key={a.agent_id} className="rounded-xl border border-gray-800 bg-gray-900 p-4">
          <div className="mb-2 flex items-center gap-2">
            <span
              className="h-3 w-3 rounded-full"
              style={{ backgroundColor: AGENT_COLORS[a.agent_type] ?? "#95A5A6" }}
            />
            <span className="text-sm font-medium text-gray-200">
              {AGENT_LABELS[a.agent_type] ?? a.agent_type}
            </span>
            <span
              className={cn(
                "ml-auto rounded-full px-2 py-0.5 text-[10px] font-medium",
                a.status === "RUNNING" && "bg-blue-950 text-blue-400",
                a.status === "COMPLETED" && "bg-green-950 text-green-400",
                a.status === "FAILED" && "bg-red-950 text-red-400",
                a.status === "QUEUED" && "bg-gray-800 text-gray-400",
              )}
            >
              {a.status}
            </span>
          </div>
          <div className="flex gap-4 text-[11px] text-gray-500">
            <span>+{a.nodes_added} nodes</span>
            <span>+{a.edges_added} edges</span>
            <span>{a.task_count} tasks</span>
          </div>
        </div>
      ))}
    </div>
  );
}

function HypothesisList({ hypotheses }: { hypotheses: HypothesisNode[] }) {
  if (hypotheses.length === 0) {
    return <div className="py-12 text-center text-sm text-gray-500">No hypotheses generated yet.</div>;
  }

  return (
    <div className="space-y-2">
      {hypotheses.map((h) => {
        const confColor =
          h.confidence > 0.7 ? "text-green-400" :
          h.confidence > 0.4 ? "text-amber-400" : "text-red-400";
        const statusBadge =
          h.status === "CONFIRMED" ? "bg-green-950 text-green-400" :
          h.status === "REFUTED" ? "bg-red-950 text-red-400" :
          h.status === "EXPLORING" ? "bg-blue-950 text-blue-400" :
          h.status === "PRUNED" ? "bg-gray-800 text-gray-500 line-through" :
          "bg-gray-800 text-gray-400";

        return (
          <div key={h.id} className="rounded-xl border border-gray-800 bg-gray-900 p-4">
            <div className="mb-2 flex items-center gap-2">
              <span className={cn("rounded-full px-2 py-0.5 text-[10px] font-medium", statusBadge)}>
                {h.status}
              </span>
              <span className={cn("ml-auto text-xs font-medium tabular-nums", confColor)}>
                {(h.confidence * 100).toFixed(0)}%
              </span>
            </div>
            <p className="mb-2 text-sm text-gray-200">{h.hypothesis}</p>
            <div className="flex gap-4 text-[10px] text-gray-500">
              <span>Depth: {h.depth}</span>
              <span>Visits: {h.visit_count}</span>
              <span>UCB: {h.ucb_score.toFixed(2)}</span>
              <span>+{h.supporting_edges.length} / -{h.contradicting_edges.length} edges</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function timeAgo(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const secs = Math.floor(diff / 1000);
  if (secs < 5) return "just now";
  if (secs < 60) return `${secs}s ago`;
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `${mins}m ago`;
  return `${Math.floor(mins / 60)}h ago`;
}
