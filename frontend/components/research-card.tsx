"use client";

import Link from "next/link";
import {
  Clock,
  GitBranch,
  Network,
  Zap,
  ArrowRight,
} from "lucide-react";
import type { ResearchSession, SessionStatus } from "@/lib/types";
import { STATUS_COLORS } from "@/lib/types";
import { cn } from "@/lib/utils";

function statusLabel(s: SessionStatus): string {
  const map: Record<SessionStatus, string> = {
    INITIALIZING: "Initializing",
    RUNNING: "Running",
    WAITING_HITL: "Awaiting Human Input",
    COMPLETED: "Completed",
    FAILED: "Failed",
    CANCELLED: "Cancelled",
  };
  return map[s] ?? s;
}

function timeAgo(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

interface ResearchCardProps {
  session: ResearchSession;
  className?: string;
}

export default function ResearchCard({ session, className }: ResearchCardProps) {
  const isActive = session.status === "RUNNING" || session.status === "INITIALIZING";
  const color = STATUS_COLORS[session.status];

  return (
    <Link
      href={`/research/${session.id}`}
      className={cn(
        "group block rounded-xl border border-gray-800 bg-gray-900 p-5 transition-all hover:border-gray-600 hover:bg-gray-900/80",
        isActive && "border-gray-700",
        className,
      )}
    >
      <div className="mb-3 flex items-start justify-between">
        <div className="flex items-center gap-2">
          <span
            className="inline-block h-2 w-2 rounded-full"
            style={{ backgroundColor: color }}
          />
          {isActive && (
            <span
              className="inline-block h-2 w-2 animate-ping rounded-full"
              style={{ backgroundColor: color }}
            />
          )}
          <span className="text-xs font-medium text-gray-400">
            {statusLabel(session.status)}
          </span>
        </div>
        <span className="text-xs text-gray-500">
          {timeAgo(session.created_at)}
        </span>
      </div>

      <h3 className="mb-3 line-clamp-2 text-sm font-medium text-gray-100 group-hover:text-white">
        {session.query}
      </h3>

      <div className="flex flex-wrap items-center gap-3 text-xs text-gray-500">
        <span className="flex items-center gap-1">
          <Network className="h-3 w-3" />
          {session.total_nodes} nodes
        </span>
        <span className="flex items-center gap-1">
          <GitBranch className="h-3 w-3" />
          {session.total_hypotheses} hypotheses
        </span>
        <span className="flex items-center gap-1">
          <Zap className="h-3 w-3" />
          {session.total_tokens_used.toLocaleString()} tokens
        </span>
        {session.current_iteration > 0 && (
          <span className="flex items-center gap-1">
            <Clock className="h-3 w-3" />
            Iter {session.current_iteration}
            {session.config.max_mcts_iterations
              ? `/${session.config.max_mcts_iterations}`
              : ""}
          </span>
        )}
        <ArrowRight className="ml-auto h-3 w-3 text-gray-600 transition-transform group-hover:translate-x-1 group-hover:text-gray-400" />
      </div>

      {isActive && (
        <div className="mt-3 h-1 overflow-hidden rounded-full bg-gray-800">
          <div
            className="h-full rounded-full bg-gradient-to-r from-pathway to-protein transition-all duration-500"
            style={{
              width: session.config.max_mcts_iterations
                ? `${Math.min(100, (session.current_iteration / session.config.max_mcts_iterations) * 100)}%`
                : "50%",
            }}
          />
        </div>
      )}
    </Link>
  );
}
