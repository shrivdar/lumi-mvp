"use client";

import { useEffect, useRef } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle,
  CirclePlus,
  Lightbulb,
  Link,
  Play,
  Rocket,
  ShieldAlert,
  Target,
  TrendingUp,
  UserCheck,
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { ResearchEvent } from "@/lib/types";

interface LiveFeedProps {
  events: ResearchEvent[];
  maxVisible?: number;
  className?: string;
}

const EVENT_CONFIG: Record<
  string,
  { icon: React.ElementType; color: string; bg: string; label: string }
> = {
  session_created: {
    icon: Rocket,
    color: "text-blue-400",
    bg: "bg-blue-500/10",
    label: "Session Created",
  },
  hypothesis_generated: {
    icon: Lightbulb,
    color: "text-purple-400",
    bg: "bg-purple-500/10",
    label: "Hypothesis",
  },
  hypothesis_selected: {
    icon: Target,
    color: "text-amber-400",
    bg: "bg-amber-500/10",
    label: "Selected",
  },
  agent_started: {
    icon: Play,
    color: "text-green-400",
    bg: "bg-green-500/10",
    label: "Agent Started",
  },
  node_created: {
    icon: CirclePlus,
    color: "text-teal-400",
    bg: "bg-teal-500/10",
    label: "Node Created",
  },
  edge_created: {
    icon: Link,
    color: "text-blue-400",
    bg: "bg-blue-500/10",
    label: "Edge Created",
  },
  edge_falsified: {
    icon: ShieldAlert,
    color: "text-red-400",
    bg: "bg-red-500/10",
    label: "Falsified",
  },
  confidence_updated: {
    icon: TrendingUp,
    color: "text-amber-400",
    bg: "bg-amber-500/10",
    label: "Confidence",
  },
  uncertainty_aggregated: {
    icon: AlertTriangle,
    color: "text-orange-400",
    bg: "bg-orange-500/10",
    label: "Uncertainty",
  },
  hitl_triggered: {
    icon: UserCheck,
    color: "text-yellow-400",
    bg: "bg-yellow-500/10",
    label: "HITL",
  },
  research_completed: {
    icon: CheckCircle,
    color: "text-green-400",
    bg: "bg-green-500/10",
    label: "Completed",
  },
};

const DEFAULT_CONFIG = {
  icon: Activity,
  color: "text-gray-400",
  bg: "bg-gray-500/10",
  label: "Event",
};

function relativeTime(timestamp: string): string {
  const diff = Date.now() - new Date(timestamp).getTime();
  const seconds = Math.floor(diff / 1000);
  if (seconds < 5) return "just now";
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

function eventDescription(event: ResearchEvent): string {
  const d = event.data;
  switch (event.event_type) {
    case "session_created":
      return (d.query as string) ?? "Research session started";
    case "hypothesis_generated":
      return (d.hypothesis as string) ?? "New hypothesis generated";
    case "hypothesis_selected":
      return (d.hypothesis as string) ?? "Hypothesis selected for exploration";
    case "agent_started":
      return `${(d.agent_type as string) ?? "Agent"} assigned to ${(d.hypothesis as string) ?? "task"}`;
    case "node_created":
      return `${(d.node_type as string) ?? "Node"}: ${(d.name as string) ?? "unknown"}`;
    case "edge_created":
      return `${(d.relation as string) ?? "Relation"} (conf: ${typeof d.confidence === "number" ? d.confidence.toFixed(2) : "?"})`;
    case "edge_falsified":
      return `Edge falsified: ${(d.reason as string) ?? (d.edge_id as string) ?? ""}`;
    case "confidence_updated":
      return `Confidence ${typeof d.old === "number" ? d.old.toFixed(2) : "?"} -> ${typeof d.new === "number" ? (d.new as number).toFixed(2) : "?"}`;
    case "uncertainty_aggregated":
      return `Composite uncertainty: ${typeof d.composite === "number" ? d.composite.toFixed(2) : "?"}${d.is_critical ? " (CRITICAL)" : ""}`;
    case "hitl_triggered":
      return (d.reason as string) ?? "Human review requested";
    case "research_completed":
      return (d.summary as string) ?? "Research completed";
    default:
      return (d.message as string) ?? event.event_type;
  }
}

export function LiveFeed({
  events,
  maxVisible = 100,
  className,
}: LiveFeedProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const prevCountRef = useRef(0);

  // Auto-scroll when new events arrive
  useEffect(() => {
    if (events.length > prevCountRef.current && scrollRef.current) {
      scrollRef.current.scrollTop = 0;
    }
    prevCountRef.current = events.length;
  }, [events.length]);

  const visible = events.slice(-maxVisible).reverse();

  return (
    <div
      ref={scrollRef}
      className={cn(
        "overflow-y-auto bg-slate-950 rounded-lg border border-slate-800",
        className
      )}
    >
      {visible.length === 0 && (
        <div className="flex items-center justify-center h-32 text-slate-500 text-sm">
          Waiting for events...
        </div>
      )}
      {visible.map((event, i) => {
        const config = EVENT_CONFIG[event.event_type] ?? DEFAULT_CONFIG;
        const Icon = config.icon;
        const isNew = i === 0 && events.length > 1;

        return (
          <div
            key={`${event.timestamp}-${event.event_type}-${i}`}
            className={cn(
              "flex items-center gap-3 px-3 h-10 border-b border-slate-800/50 hover:bg-slate-900/60 transition-colors",
              isNew && "animate-feed-flash"
            )}
          >
            {/* Timestamp */}
            <span className="text-[11px] text-slate-500 font-mono w-16 shrink-0 text-right">
              {relativeTime(event.timestamp)}
            </span>

            {/* Icon */}
            <div
              className={cn(
                "flex items-center justify-center w-6 h-6 rounded shrink-0",
                config.bg
              )}
            >
              <Icon className={cn("w-3.5 h-3.5", config.color)} />
            </div>

            {/* Badge */}
            <span
              className={cn(
                "text-[10px] font-medium uppercase tracking-wider shrink-0 w-20 truncate",
                config.color
              )}
            >
              {config.label}
            </span>

            {/* Description */}
            <span className="text-xs text-slate-300 truncate">
              {eventDescription(event)}
            </span>
          </div>
        );
      })}
    </div>
  );
}
