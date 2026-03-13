"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import cytoscape, { type Core, type EventObject } from "cytoscape";
// @ts-expect-error -- no type declarations for cytoscape-cola
import cola from "cytoscape-cola";
import { cn } from "@/lib/utils";
import { type KGNode, type KGEdge, type NodeType, NODE_COLORS } from "@/lib/types";
import { ZoomIn, ZoomOut, Maximize, LayoutGrid, Network, Search } from "lucide-react";

// Register cola layout once
if (typeof window !== "undefined") {
  try { cytoscape.use(cola); } catch { /* already registered */ }
}

interface KnowledgeGraphProps {
  nodes: KGNode[];
  edges: KGEdge[];
  timeRange?: [string, string];
  selectedHypothesisBranch?: string | null;
  onNodeSelect?: (node: KGNode) => void;
  onEdgeSelect?: (edge: KGEdge) => void;
  className?: string;
}

/** Map confidence 0-1 to #E74C3C -> #F39C12 -> #2ECC71 */
function confidenceColor(v: number): string {
  const t = Math.max(0, Math.min(1, v));
  const lerp = (a: number, b: number, p: number) => Math.round(a + (b - a) * p);
  const hex = (n: number) => n.toString(16).padStart(2, "0");
  let r: number, g: number, b: number;
  if (t < 0.5) {
    const p = t / 0.5;
    r = lerp(0xe7, 0xf3, p); g = lerp(0x4c, 0x9c, p); b = lerp(0x3c, 0x12, p);
  } else {
    const p = (t - 0.5) / 0.5;
    r = lerp(0xf3, 0x2e, p); g = lerp(0x9c, 0xcc, p); b = lerp(0x12, 0x71, p);
  }
  return `#${hex(r)}${hex(g)}${hex(b)}`;
}

/** Deterministic color from hypothesis branch ID via djb2 hash */
function branchColor(branch: string): string {
  const palette = [
    "#4A90D9", "#6B5CE7", "#E91E63", "#1ABC9C", "#F39C12",
    "#8E44AD", "#00BCD4", "#FF5722", "#009688", "#795548", "#3F51B5", "#CDDC39",
  ];
  let hash = 5381;
  for (let i = 0; i < branch.length; i++) hash = ((hash << 5) + hash + branch.charCodeAt(i)) >>> 0;
  return palette[hash % palette.length];
}

function useDebouncedCallback<T extends (...args: unknown[]) => void>(fn: T, delay: number) {
  const timer = useRef<ReturnType<typeof setTimeout>>(undefined);
  return useCallback((...args: Parameters<T>) => {
    clearTimeout(timer.current);
    timer.current = setTimeout(() => fn(...args), delay);
  }, [fn, delay]);
}

 
function buildStylesheet(): any[] {
  return [
    { selector: ":parent", style: {
      "background-opacity": 0.06, "background-color": "#6B7280",
      "border-width": 1, "border-color": "#374151", "border-opacity": 0.15,
      shape: "round-rectangle", padding: "24px",
      label: "data(label)", "font-size": "10px", color: "#9CA3AF",
      "text-valign": "top", "text-halign": "center", "text-margin-y": -4,
    } as cytoscape.Css.Node },
    { selector: "node[nodeType]", style: {
      width: "data(size)", height: "data(size)",
      "background-color": "data(bgColor)", "background-opacity": 0.92,
      "border-width": 2.5, "border-color": "data(borderColor)",
      label: "data(label)", "font-size": "9px", "font-weight": 500,
      color: "#E5E7EB", "text-valign": "bottom", "text-margin-y": 6,
      "text-wrap": "ellipsis", "text-max-width": "72px",
      "text-outline-width": 2, "text-outline-color": "#111827",
      "overlay-padding": "4px",
      "shadow-blur": "8px", "shadow-color": "data(borderColor)",
      "shadow-opacity": 0.35, "shadow-offset-x": 0, "shadow-offset-y": 0,
    } as cytoscape.Css.Node },
    { selector: "node.dimmed", style: { opacity: 0.15 } as cytoscape.Css.Node },
    { selector: "node:selected", style: {
      "border-width": 4, "border-color": "#FBBF24",
      "shadow-blur": "18px", "shadow-color": "#FBBF24", "shadow-opacity": 0.6,
    } as cytoscape.Css.Node },
    { selector: "edge", style: {
      width: "data(edgeWidth)", "line-color": "data(edgeColor)",
      "target-arrow-color": "data(edgeColor)", "target-arrow-shape": "triangle",
      "arrow-scale": 0.8, "curve-style": "bezier", opacity: 0.8,
      "line-dash-pattern": "data(dashPattern)", "line-dash-offset": 0,
      "line-style": "data(lineStyle)",
      label: "data(relation)", "font-size": "7px", color: "#9CA3AF",
      "text-rotation": "autorotate", "text-outline-width": 1.5,
      "text-outline-color": "#111827", "text-margin-y": -8, "overlay-padding": "3px",
    } as unknown as cytoscape.Css.Edge },
    { selector: "edge.falsified", style: {
      "line-color": "#E74C3C", "line-style": "dashed",
      "line-dash-pattern": [8, 6], opacity: 0.4, "target-arrow-color": "#E74C3C",
    } as cytoscape.Css.Edge },
    { selector: "edge.dimmed", style: { opacity: 0.08 } as cytoscape.Css.Edge },
  ];
}

export function KnowledgeGraph({
  nodes, edges, timeRange, selectedHypothesisBranch,
  onNodeSelect, onEdgeSelect, className,
}: KnowledgeGraphProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);
  const animFrameRef = useRef<number>(0);
  const [layout, setLayout] = useState<"cola" | "grid">("cola");
  const [tooltip, setTooltip] = useState<{
    x: number; y: number; name: string; type: string; confidence: number;
  } | null>(null);

  const nodeMap = useMemo(() => {
    const m = new Map<string, KGNode>();
    nodes.forEach((n) => m.set(n.id, n));
    return m;
  }, [nodes]);

  const edgeMap = useMemo(() => {
    const m = new Map<string, KGEdge>();
    edges.forEach((e) => m.set(e.id, e));
    return m;
  }, [edges]);

  // Temporal filtering
  const filtered = useMemo(() => {
    let fn = nodes, fe = edges;
    if (timeRange) {
      const [start, end] = timeRange.map((t) => new Date(t).getTime());
      fn = fn.filter((n) => { const ts = new Date(n.created_at).getTime(); return ts >= start && ts <= end; });
      const visibleIds = new Set(fn.map((n) => n.id));
      fe = fe.filter((e) => visibleIds.has(e.source_id) && visibleIds.has(e.target_id));
    }
    return { nodes: fn, edges: fe };
  }, [nodes, edges, timeRange]);

  // Cluster compound nodes
  const clusters = useMemo(() => {
    const map = new Map<string, string[]>();
    filtered.nodes.forEach((n) => {
      const cid = n.viz.cluster_id;
      if (cid) { if (!map.has(cid)) map.set(cid, []); map.get(cid)!.push(n.id); }
    });
    return map;
  }, [filtered.nodes]);

  // Build Cytoscape elements
  const elements = useMemo(() => {
    const els: cytoscape.ElementDefinition[] = [];
    clusters.forEach((_, cid) => {
      els.push({ data: { id: `cluster-${cid}`, label: cid }, classes: "cluster-parent" });
    });
    filtered.nodes.forEach((n) => {
      const size = 24 + n.confidence * (n.viz.visual_weight || 1) * 28;
      const borderColor = NODE_COLORS[n.type as NodeType] ?? "#6B7280";
      const bgColor = n.hypothesis_branch ? branchColor(n.hypothesis_branch) : "#374151";
      const parent = n.viz.cluster_id ? `cluster-${n.viz.cluster_id}` : undefined;
      els.push({ data: {
        id: n.id, label: n.name, nodeType: n.type, size, borderColor, bgColor,
        confidence: n.confidence, branch: n.hypothesis_branch ?? "", parent,
      }});
    });
    filtered.edges.forEach((e) => {
      const conf = e.confidence.overall;
      const speed = 1 + conf * 3;
      const dashLen = Math.round(12 / speed), dashGap = Math.round(8 / speed);
      els.push({
        data: {
          id: e.id, source: e.source_id, target: e.target_id,
          relation: e.relation.replace(/_/g, " ").toLowerCase(),
          edgeColor: e.falsified ? "#E74C3C" : confidenceColor(conf),
          edgeWidth: 1 + conf * 3,
          dashPattern: e.falsified ? [8, 6] : [dashLen, dashGap],
          lineStyle: "dashed", confidence: conf, animSpeed: speed,
        },
        classes: e.falsified ? "falsified" : undefined,
      });
    });
    return els;
  }, [filtered, clusters]);

  // Layout runner
  const runLayout = useCallback(() => {
    const cy = cyRef.current;
    if (!cy || cy.elements().length === 0) return;
    const opts = layout === "cola"
      ? { name: "cola" as const, animate: true, animationDuration: 600, nodeSpacing: 18,
          maxSimulationTime: 3000, randomize: false, convergenceThreshold: 0.01, fit: true, padding: 40 }
      : { name: "grid" as const, animate: true, animationDuration: 400, fit: true, padding: 40, avoidOverlap: true };
    cy.layout(opts).run();
  }, [layout]);

  const debouncedLayout = useDebouncedCallback(runLayout, 200);

  // Edge dash animation (evidence flow)
  const startEdgeAnimation = useCallback(() => {
    const cy = cyRef.current;
    if (!cy) return;
    let offset = 0;
    const tick = () => {
      offset += 0.6;
      cy.edges().forEach((edge) => {
        edge.style("line-dash-offset", -offset * (edge.data("animSpeed") ?? 1));
      });
      animFrameRef.current = requestAnimationFrame(tick);
    };
    animFrameRef.current = requestAnimationFrame(tick);
  }, []);

  // Init / destroy Cytoscape
  useEffect(() => {
    if (!containerRef.current) return;
    const cy = cytoscape({
      container: containerRef.current, elements, style: buildStylesheet(),
      minZoom: 0.15, maxZoom: 5, wheelSensitivity: 0.3, boxSelectionEnabled: false,
    });
    cyRef.current = cy;
    runLayout();
    startEdgeAnimation();

    cy.on("tap", "node[nodeType]", (evt: EventObject) => {
      const kgNode = nodeMap.get(evt.target.id());
      if (kgNode && onNodeSelect) onNodeSelect(kgNode);
    });
    cy.on("tap", "edge", (evt: EventObject) => {
      const kgEdge = edgeMap.get(evt.target.id());
      if (kgEdge && onEdgeSelect) onEdgeSelect(kgEdge);
    });
    cy.on("mouseover", "node[nodeType]", (evt: EventObject) => {
      const node = evt.target, pos = node.renderedPosition();
      setTooltip({
        x: pos.x, y: pos.y - node.renderedOuterHeight() / 2 - 10,
        name: node.data("label"), type: node.data("nodeType"), confidence: node.data("confidence"),
      });
      containerRef.current!.style.cursor = "pointer";
    });
    cy.on("mouseout", "node[nodeType]", () => {
      setTooltip(null);
      containerRef.current!.style.cursor = "default";
    });

    return () => { cancelAnimationFrame(animFrameRef.current); cy.destroy(); cyRef.current = null; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Sync data changes
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.json({ elements });
    debouncedLayout();
  }, [elements, debouncedLayout]);

  // Hypothesis branch dimming
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    if (selectedHypothesisBranch) {
      cy.nodes().forEach((n) => {
        const branch = n.data("branch");
        n.toggleClass("dimmed", !!branch && branch !== selectedHypothesisBranch);
      });
      cy.edges().forEach((e) => {
        e.toggleClass("dimmed", e.source().hasClass("dimmed") || e.target().hasClass("dimmed"));
      });
    } else {
      cy.elements().removeClass("dimmed");
    }
  }, [selectedHypothesisBranch]);

  // Controls
  const zoomIn = () => cyRef.current?.zoom(cyRef.current.zoom() * 1.3);
  const zoomOut = () => cyRef.current?.zoom(cyRef.current.zoom() / 1.3);
  const fitScreen = () => cyRef.current?.fit(undefined, 40);
  const toggleLayout = () => setLayout((l) => (l === "cola" ? "grid" : "cola"));

  useEffect(() => { debouncedLayout(); }, [layout, debouncedLayout]);

  // Empty state
  if (nodes.length === 0) {
    return (
      <div className={cn("relative flex items-center justify-center rounded-xl border border-zinc-800 bg-zinc-950 text-zinc-500", className)}>
        <div className="flex flex-col items-center gap-3">
          <Search className="h-10 w-10 opacity-30" />
          <p className="text-sm">No knowledge graph data yet.</p>
          <p className="text-xs text-zinc-600">Start a research session to build the graph.</p>
        </div>
      </div>
    );
  }

  return (
    <div className={cn("relative overflow-hidden rounded-xl border border-zinc-800 bg-zinc-950", className)}>
      <div ref={containerRef} className="h-full w-full" />

      {/* Tooltip */}
      {tooltip && (
        <div
          className="pointer-events-none absolute z-30 rounded-lg border border-zinc-700 bg-zinc-900/95 px-3 py-2 shadow-xl backdrop-blur-sm"
          style={{ left: tooltip.x, top: tooltip.y, transform: "translate(-50%, -100%)" }}
        >
          <p className="text-xs font-semibold text-zinc-100">{tooltip.name}</p>
          <div className="mt-0.5 flex items-center gap-2">
            <span className="inline-block h-2 w-2 rounded-full"
              style={{ backgroundColor: NODE_COLORS[tooltip.type as NodeType] ?? "#6B7280" }} />
            <span className="text-[10px] text-zinc-400">{tooltip.type}</span>
            <span className="text-[10px] text-zinc-500">|</span>
            <span className="text-[10px] font-medium" style={{ color: confidenceColor(tooltip.confidence) }}>
              {(tooltip.confidence * 100).toFixed(0)}% conf
            </span>
          </div>
        </div>
      )}

      {/* Controls */}
      <div className="absolute bottom-4 right-4 z-20 flex flex-col gap-1.5">
        {[
          { icon: ZoomIn, action: zoomIn, title: "Zoom in" },
          { icon: ZoomOut, action: zoomOut, title: "Zoom out" },
          { icon: Maximize, action: fitScreen, title: "Fit to screen" },
          { icon: layout === "cola" ? LayoutGrid : Network, action: toggleLayout,
            title: layout === "cola" ? "Switch to grid" : "Switch to cola" },
        ].map(({ icon: Icon, action, title }) => (
          <button key={title} onClick={action} title={title}
            className="flex h-8 w-8 items-center justify-center rounded-lg border border-zinc-700 bg-zinc-900/90 text-zinc-400 backdrop-blur-sm transition-colors hover:border-zinc-500 hover:text-zinc-200">
            <Icon className="h-4 w-4" />
          </button>
        ))}
      </div>

      {/* Legend */}
      <div className="absolute bottom-4 left-4 z-20 rounded-lg border border-zinc-800 bg-zinc-900/90 p-2.5 backdrop-blur-sm">
        <p className="mb-1.5 text-[10px] font-medium uppercase tracking-wider text-zinc-500">Confidence</p>
        <div className="flex items-center gap-1">
          <span className="text-[9px] text-zinc-500">0</span>
          <div className="h-2 w-24 rounded-full"
            style={{ background: "linear-gradient(to right, #E74C3C, #F39C12, #2ECC71)" }} />
          <span className="text-[9px] text-zinc-500">1</span>
        </div>
        <div className="mt-1.5 flex items-center gap-1.5">
          <div className="h-[2px] w-4 border-t-2 border-dashed border-red-500 opacity-50" />
          <span className="text-[9px] text-zinc-500">Falsified</span>
        </div>
      </div>

      {/* Node count badge */}
      <div className="absolute right-4 top-4 z-20 rounded-lg border border-zinc-800 bg-zinc-900/90 px-2.5 py-1 backdrop-blur-sm">
        <span className="text-[10px] text-zinc-400">
          {filtered.nodes.length} nodes &middot; {filtered.edges.length} edges
        </span>
      </div>
    </div>
  );
}

export default KnowledgeGraph;
