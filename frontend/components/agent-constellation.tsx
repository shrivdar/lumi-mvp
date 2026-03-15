"use client";

import { useMemo } from "react";
import { Activity, AlertCircle, CheckCircle2, Circle } from "lucide-react";
import type { AgentInfo, AgentType, ResearchEvent } from "@/lib/types";
import { AGENT_COLORS, AGENT_LABELS } from "@/lib/types";
import { cn } from "@/lib/utils";

interface AgentConstellationProps {
  agents: AgentInfo[];
  query: string;
  events?: ResearchEvent[];
  className?: string;
}

const ABBREVIATIONS: Record<AgentType, string> = {
  literature_analyst: "LIT",
  protein_engineer: "PRO",
  genomics_mapper: "GEN",
  pathway_analyst: "PTH",
  drug_hunter: "DRG",
  clinical_analyst: "CLN",
  scientific_critic: "CRT",
  experiment_designer: "EXP",
  tool_creator: "TLC",
};

const ORBIT_RADII = [110, 140, 170];
const ORBIT_DURATIONS = [60, 80, 100]; // seconds per revolution

function getOrbitIndex(i: number, total: number): number {
  if (total <= 3) return 0;
  if (total <= 6) return i < 3 ? 0 : 1;
  return i < 3 ? 0 : i < 6 ? 1 : 2;
}

function activitySize(agent: AgentInfo): number {
  const activity = agent.nodes_added + agent.edges_added;
  if (activity === 0) return 16;
  if (activity < 5) return 20;
  if (activity < 15) return 24;
  return 28;
}

function truncateQuery(query: string, maxLen: number = 32): string {
  if (query.length <= maxLen) return query;
  return query.slice(0, maxLen - 1) + "\u2026";
}

/** IDs of agents that recently acted, derived from events */
function recentlyActiveAgentIds(events?: ResearchEvent[]): Set<string> {
  if (!events || events.length === 0) return new Set();
  const now = Date.now();
  const activeTypes = new Set(["agent_started", "node_created", "edge_created"]);
  const ids = new Set<string>();
  for (const ev of events) {
    if (!activeTypes.has(ev.event_type)) continue;
    const age = now - new Date(ev.timestamp).getTime();
    if (age < 10_000) {
      const aid = (ev.data.agent_id as string) ?? null;
      if (aid) ids.add(aid);
    }
  }
  return ids;
}

export default function AgentConstellation({
  agents,
  query,
  events,
  className,
}: AgentConstellationProps) {
  const activeIds = useMemo(() => recentlyActiveAgentIds(events), [events]);

  // Assign orbit positions
  const positioned = useMemo(() => {
    return agents.map((agent, i) => {
      const orbitIdx = getOrbitIndex(i, agents.length);
      // Count how many agents share this orbit
      const sameOrbit = agents.filter(
        (_, j) => getOrbitIndex(j, agents.length) === orbitIdx
      );
      const posInOrbit = sameOrbit.indexOf(agent);
      const angleOffset = (360 / sameOrbit.length) * posInOrbit;
      return { agent, orbitIdx, angleOffset };
    });
  }, [agents]);

  const svgSize = 440;
  const cx = svgSize / 2;
  const cy = svgSize / 2;

  return (
    <div className={cn("relative", className)}>
      {/* CSS keyframes for orbiting + pulse */}
      { }
      <style>{`
        @keyframes pulse-ring {
          0% { opacity: 1; r: inherit; }
          100% { opacity: 0; r: 22px; }
        }
        @keyframes particle-flow {
          0% { offset-distance: 0%; opacity: 0; }
          10% { opacity: 1; }
          90% { opacity: 1; }
          100% { offset-distance: 100%; opacity: 0; }
        }
        @keyframes glow-core {
          0%, 100% { opacity: 0.6; }
          50% { opacity: 1; }
        }
        ${ORBIT_RADII.map(
          (_, i) => `
          @keyframes orbit-${i} {
            from { transform: rotate(0deg) translateX(${ORBIT_RADII[i]}px) rotate(0deg); }
            to   { transform: rotate(360deg) translateX(${ORBIT_RADII[i]}px) rotate(-360deg); }
          }
        `
        ).join("")}
      `}</style>

      <svg
        viewBox={`0 0 ${svgSize} ${svgSize}`}
        className="w-full h-auto max-w-[440px] mx-auto"
        role="img"
        aria-label="Agent constellation visualization"
      >
        <defs>
          <radialGradient id="core-glow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#60a5fa" stopOpacity="0.9" />
            <stop offset="70%" stopColor="#3b82f6" stopOpacity="0.3" />
            <stop offset="100%" stopColor="#1e3a5f" stopOpacity="0" />
          </radialGradient>
          <filter id="blur-glow">
            <feGaussianBlur stdDeviation="4" />
          </filter>
        </defs>

        {/* Orbit rings */}
        {ORBIT_RADII.map((r, i) => (
          <circle
            key={`orbit-${i}`}
            cx={cx}
            cy={cy}
            r={r}
            fill="none"
            stroke="#334155"
            strokeWidth={0.5}
            strokeDasharray="4 6"
            opacity={0.5}
          />
        ))}

        {/* Central glow */}
        <circle cx={cx} cy={cy} r={40} fill="url(#core-glow)" opacity={0.8}>
          <animate
            attributeName="opacity"
            values="0.6;1;0.6"
            dur="3s"
            repeatCount="indefinite"
          />
        </circle>

        {/* Central node */}
        <circle cx={cx} cy={cy} r={24} fill="#1e293b" stroke="#3b82f6" strokeWidth={2} />
        <text
          x={cx}
          y={cy}
          textAnchor="middle"
          dominantBaseline="central"
          fill="#e2e8f0"
          fontSize="8"
          fontFamily="monospace"
        >
          {truncateQuery(query, 20)}
        </text>

        {/* Connection lines for recently-active agents */}
        {positioned.map(({ agent, orbitIdx, angleOffset }) => {
          if (!activeIds.has(agent.agent_id)) return null;
          const rad = ((angleOffset - 90) * Math.PI) / 180;
          const r = ORBIT_RADII[orbitIdx];
          const ax = cx + r * Math.cos(rad);
          const ay = cy + r * Math.sin(rad);
          const color = AGENT_COLORS[agent.agent_type];
          return (
            <g key={`conn-${agent.agent_id}`}>
              <line
                x1={cx}
                y1={cy}
                x2={ax}
                y2={ay}
                stroke={color}
                strokeWidth={1.5}
                strokeDasharray="3 3"
                opacity={0.6}
              >
                <animate
                  attributeName="stroke-dashoffset"
                  from="0"
                  to="-12"
                  dur="1s"
                  repeatCount="indefinite"
                />
              </line>
              {/* Particle flowing along line */}
              {[0, 0.33, 0.66].map((delay, pi) => {
                const dur = 1.5;
                return (
                  <circle key={pi} r={2} fill={color} opacity={0}>
                    <animate
                      attributeName="cx"
                      from={cx}
                      to={ax}
                      dur={`${dur}s`}
                      begin={`${delay * dur}s`}
                      repeatCount="indefinite"
                    />
                    <animate
                      attributeName="cy"
                      from={cy}
                      to={ay}
                      dur={`${dur}s`}
                      begin={`${delay * dur}s`}
                      repeatCount="indefinite"
                    />
                    <animate
                      attributeName="opacity"
                      values="0;1;1;0"
                      keyTimes="0;0.1;0.8;1"
                      dur={`${dur}s`}
                      begin={`${delay * dur}s`}
                      repeatCount="indefinite"
                    />
                  </circle>
                );
              })}
            </g>
          );
        })}

        {/* Agent nodes — each wrapped in a group that orbits */}
        {positioned.map(({ agent, orbitIdx, angleOffset }) => {
          const color = AGENT_COLORS[agent.agent_type];
          const size = activitySize(agent);
          const r = ORBIT_RADII[orbitIdx];
          const dur = ORBIT_DURATIONS[orbitIdx];
          const isRunning = agent.status === "RUNNING";
          const isFailed = agent.status === "FAILED";

          // Static position derived from angleOffset
          const rad = ((angleOffset - 90) * Math.PI) / 180;
          const baseX = cx + r * Math.cos(rad);
          const baseY = cy + r * Math.sin(rad);

          return (
            <g key={agent.agent_id}>
              {/* Orbit animation wrapper */}
              <g
                style={{
                  transformOrigin: `${cx}px ${cy}px`,
                  animation: `orbit-${orbitIdx} ${dur}s linear infinite`,
                }}
              >
                {/* Translate to static position, counter-rotate is baked into keyframes */}
                <g transform={`translate(${baseX}, ${baseY})`}>
                  {/* Running pulse ring */}
                  {isRunning && (
                    <circle
                      cx={0}
                      cy={0}
                      r={size / 2 + 2}
                      fill="none"
                      stroke={color}
                      strokeWidth={1.5}
                      opacity={0}
                    >
                      <animate
                        attributeName="r"
                        from={size / 2 + 2}
                        to={size / 2 + 10}
                        dur="1.5s"
                        repeatCount="indefinite"
                      />
                      <animate
                        attributeName="opacity"
                        values="0.8;0"
                        dur="1.5s"
                        repeatCount="indefinite"
                      />
                    </circle>
                  )}

                  {/* Agent circle */}
                  <circle
                    cx={0}
                    cy={0}
                    r={size / 2}
                    fill={`${color}22`}
                    stroke={isFailed ? "#ef4444" : color}
                    strokeWidth={isFailed ? 2.5 : 1.5}
                  />

                  {/* Abbreviation label */}
                  <text
                    x={0}
                    y={0}
                    textAnchor="middle"
                    dominantBaseline="central"
                    fill={color}
                    fontSize="7"
                    fontWeight="bold"
                    fontFamily="monospace"
                  >
                    {ABBREVIATIONS[agent.agent_type]}
                  </text>

                  {/* Status icon indicator */}
                  <g transform={`translate(${size / 2 - 2}, ${-size / 2 + 2})`}>
                    {isRunning && (
                      <circle cx={0} cy={0} r={3} fill="#22c55e">
                        <animate
                          attributeName="opacity"
                          values="1;0.3;1"
                          dur="1s"
                          repeatCount="indefinite"
                        />
                      </circle>
                    )}
                    {agent.status === "COMPLETED" && (
                      <circle cx={0} cy={0} r={3} fill="#22c55e" />
                    )}
                    {isFailed && (
                      <circle cx={0} cy={0} r={3} fill="#ef4444" />
                    )}
                  </g>
                </g>
              </g>
            </g>
          );
        })}
      </svg>

      {/* Legend */}
      <div className="flex flex-wrap justify-center gap-x-4 gap-y-1 mt-3 text-xs text-slate-400">
        {agents.map((agent) => (
          <div key={agent.agent_id} className="flex items-center gap-1.5">
            <span
              className="inline-block w-2.5 h-2.5 rounded-full"
              style={{ backgroundColor: AGENT_COLORS[agent.agent_type] }}
            />
            <span>{AGENT_LABELS[agent.agent_type]}</span>
            {agent.status === "RUNNING" && (
              <Activity className="w-3 h-3 text-green-400 animate-pulse" />
            )}
            {agent.status === "COMPLETED" && (
              <CheckCircle2 className="w-3 h-3 text-green-500" />
            )}
            {agent.status === "FAILED" && (
              <AlertCircle className="w-3 h-3 text-red-500" />
            )}
            {(agent.status === "QUEUED" || agent.status === "WAITING_HITL") && (
              <Circle className="w-3 h-3 text-slate-500" />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
