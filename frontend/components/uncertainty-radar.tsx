"use client";

import { useMemo } from "react";
import type { UncertaintyVector } from "@/lib/types";
import { cn } from "@/lib/utils";

interface UncertaintyRadarProps {
  uncertainty: UncertaintyVector;
  size?: number;
  className?: string;
}

const AXES: { key: keyof Omit<UncertaintyVector, "composite" | "is_critical">; label: string }[] = [
  { key: "input_ambiguity", label: "Ambiguity" },
  { key: "data_quality", label: "Data Qual" },
  { key: "reasoning_divergence", label: "Divergence" },
  { key: "model_disagreement", label: "Disagreement" },
  { key: "conflict_uncertainty", label: "Conflict" },
  { key: "novelty_uncertainty", label: "Novelty" },
];

const GRID_LEVELS = [0.25, 0.5, 0.75, 1.0];

/** Interpolate green -> yellow -> red based on score 0..1 */
function uncertaintyColor(score: number): string {
  if (score < 0.5) {
    // green to yellow
    const t = score / 0.5;
    const r = Math.round(34 + t * (234 - 34));
    const g = Math.round(197 + t * (179 - 197));
    const b = Math.round(94 + t * (8 - 94));
    return `rgb(${r},${g},${b})`;
  }
  // yellow to red
  const t = (score - 0.5) / 0.5;
  const r = Math.round(234 + t * (231 - 234));
  const g = Math.round(179 - t * 179);
  const b = Math.round(8 + t * (52 - 8));
  return `rgb(${r},${g},${b})`;
}

function polarToCartesian(
  cx: number,
  cy: number,
  radius: number,
  angleIndex: number,
  total: number
): { x: number; y: number } {
  const angle = (2 * Math.PI * angleIndex) / total - Math.PI / 2;
  return {
    x: cx + radius * Math.cos(angle),
    y: cy + radius * Math.sin(angle),
  };
}

export default function UncertaintyRadar({
  uncertainty,
  size = 200,
  className,
}: UncertaintyRadarProps) {
  const cx = size / 2;
  const cy = size / 2;
  const maxR = size * 0.35; // leave room for labels
  const n = AXES.length;

  const color = useMemo(
    () => uncertaintyColor(uncertainty.composite),
    [uncertainty.composite]
  );

  const dataPoints = useMemo(() => {
    return AXES.map((axis, i) => {
      const val = uncertainty[axis.key];
      const r = val * maxR;
      return polarToCartesian(cx, cy, r, i, n);
    });
  }, [uncertainty, cx, cy, maxR, n]);

  const polygonPath = dataPoints.map((p) => `${p.x},${p.y}`).join(" ");

  return (
    <div className={cn("inline-block", className)}>
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        role="img"
        aria-label={`Uncertainty radar: composite score ${uncertainty.composite.toFixed(2)}`}
      >
        {/* Grid circles */}
        {GRID_LEVELS.map((level) => (
          <polygon
            key={level}
            points={AXES.map((_, i) =>
              polarToCartesian(cx, cy, level * maxR, i, n)
            )
              .map((p) => `${p.x},${p.y}`)
              .join(" ")}
            fill="none"
            stroke="#334155"
            strokeWidth={0.5}
            opacity={0.6}
          />
        ))}

        {/* Axis lines */}
        {AXES.map((_, i) => {
          const end = polarToCartesian(cx, cy, maxR, i, n);
          return (
            <line
              key={`axis-${i}`}
              x1={cx}
              y1={cy}
              x2={end.x}
              y2={end.y}
              stroke="#475569"
              strokeWidth={0.5}
            />
          );
        })}

        {/* Data polygon */}
        <polygon
          points={polygonPath}
          fill={color}
          fillOpacity={0.2}
          stroke={color}
          strokeWidth={1.5}
        />

        {/* Data points */}
        {dataPoints.map((p, i) => (
          <circle key={`dp-${i}`} cx={p.x} cy={p.y} r={2.5} fill={color} />
        ))}

        {/* Axis labels */}
        {AXES.map((axis, i) => {
          const labelR = maxR + 18;
          const pos = polarToCartesian(cx, cy, labelR, i, n);
          return (
            <text
              key={`label-${i}`}
              x={pos.x}
              y={pos.y}
              textAnchor="middle"
              dominantBaseline="central"
              fill="#94a3b8"
              fontSize="8"
              fontFamily="sans-serif"
            >
              {axis.label}
            </text>
          );
        })}

        {/* Center composite score */}
        <text
          x={cx}
          y={cy - 6}
          textAnchor="middle"
          dominantBaseline="central"
          fill={color}
          fontSize="16"
          fontWeight="bold"
          fontFamily="monospace"
        >
          {uncertainty.composite.toFixed(2)}
        </text>
        <text
          x={cx}
          y={cy + 8}
          textAnchor="middle"
          dominantBaseline="central"
          fill="#64748b"
          fontSize="7"
          fontFamily="sans-serif"
        >
          composite
        </text>
      </svg>
    </div>
  );
}
