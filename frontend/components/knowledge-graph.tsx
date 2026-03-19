"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import cytoscape, { type Core, type EventObject } from "cytoscape";
// @ts-expect-error -- no type declarations for cytoscape-cola
import cola from "cytoscape-cola";
import { cn } from "@/lib/utils";
import { type KGNode, type KGEdge, type NodeType, type EdgeRelationType, NODE_COLORS } from "@/lib/types";
import { ZoomIn, ZoomOut, Maximize, LayoutGrid, Network, Search, X } from "lucide-react";

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

// ---------------------------------------------------------------------------
// Edge color / style mapping by relation type
// ---------------------------------------------------------------------------

type EdgeVisual = { color: string; dashed: boolean };

const EDGE_RELATION_STYLES: Partial<Record<EdgeRelationType, EdgeVisual>> = {
  EVIDENCE_FOR: { color: "#2ECC71", dashed: false },
  SUPPORTED_BY: { color: "#2ECC71", dashed: false },
  ACTIVATES: { color: "#2ECC71", dashed: false },
  UPREGULATES: { color: "#2ECC71", dashed: false },
  SYNERGIZES_WITH: { color: "#27AE60", dashed: false },
  EVIDENCE_AGAINST: { color: "#E74C3C", dashed: false },
  CONTRADICTS: { color: "#E74C3C", dashed: false },
  INHIBITS: { color: "#E74C3C", dashed: true },
  ANTAGONIZES: { color: "#E74C3C", dashed: true },
  DOWNREGULATES: { color: "#C0392B", dashed: true },
  ASSOCIATED_WITH: { color: "#8E8E93", dashed: false },
  CORRELATES_WITH: { color: "#8E8E93", dashed: false },
};

function edgeVisualForRelation(relation: string): EdgeVisual {
  return EDGE_RELATION_STYLES[relation as EdgeRelationType] ?? { color: "#555566", dashed: false };
}

// ---------------------------------------------------------------------------
// Confidence color (red -> amber -> green)
// ---------------------------------------------------------------------------

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

function truncate(s: string, max: number): string {
  return s.length > max ? s.slice(0, max - 1) + "\u2026" : s;
}

// ---------------------------------------------------------------------------
// Stylesheet builder
// ---------------------------------------------------------------------------

function buildStylesheet(): cytoscape.Stylesheet[] {
  return [
    // Compound (cluster parent) nodes
    {
      selector: ":parent",
      style: {
        "background-opacity": 0.04,
        "background-color": "#6B7280",
        "border-width": 1,
        "border-color": "#1e2030",
        "border-opacity": 0.3,
        shape: "round-rectangle",
        padding: "28px",
        label: "data(label)",
        "font-size": "10px",
        color: "#4a4e69",
        "text-valign": "top",
        "text-halign": "center",
        "text-margin-y": -4,
      } as cytoscape.Css.Node,
    },
    // Regular nodes
    {
      selector: "node[nodeType]",
      style: {
        width: "data(size)",
        height: "data(size)",
        "background-color": "data(bgColor)",
        "background-opacity": 0.88,
        "border-width": 2.5,
        "border-color": "data(borderColor)",
        "border-opacity": 0.9,
        label: "data(label)",
        "font-size": "9px",
        "font-weight": 600,
        "font-family": "'Inter', 'SF Pro', system-ui, sans-serif",
        color: "#d1d5e0",
        "text-valign": "bottom",
        "text-margin-y": 7,
        "text-wrap": "ellipsis",
        "text-max-width": "80px",
        "text-outline-width": 2.5,
        "text-outline-color": "#08080f",
        "text-outline-opacity": 0.9,
        "overlay-padding": "5px",
        "shadow-blur": "12px",
        "shadow-color": "data(borderColor)",
        "shadow-opacity": 0.45,
        "shadow-offset-x": 0,
        "shadow-offset-y": 0,
      } as cytoscape.Css.Node,
    },
    // Hover state
    {
      selector: "node[nodeType]:active",
      style: {
        "overlay-opacity": 0.08,
        "overlay-color": "#ffffff",
      } as cytoscape.Css.Node,
    },
    // Selected node
    {
      selector: "node:selected",
      style: {
        "border-width": 4,
        "border-color": "#FBBF24",
        "shadow-blur": "24px",
        "shadow-color": "#FBBF24",
        "shadow-opacity": 0.7,
      } as cytoscape.Css.Node,
    },
    // Highlighted neighbors
    {
      selector: "node.highlighted",
      style: {
        "border-width": 3.5,
        "border-color": "#60a5fa",
        "shadow-blur": "16px",
        "shadow-color": "#60a5fa",
        "shadow-opacity": 0.5,
      } as cytoscape.Css.Node,
    },
    // Dimmed nodes
    {
      selector: "node.dimmed",
      style: {
        opacity: 0.12,
      } as cytoscape.Css.Node,
    },
    // Edges
    {
      selector: "edge",
      style: {
        width: "data(edgeWidth)",
        "line-color": "data(edgeColor)",
        "target-arrow-color": "data(edgeColor)",
        "target-arrow-shape": "triangle",
        "arrow-scale": 0.7,
        "curve-style": "bezier",
        opacity: "data(edgeOpacity)",
        "line-style": "data(lineStyle)",
        "line-dash-pattern": "data(dashPattern)",
        "line-dash-offset": 0,
        // Hide edge labels by default -- show on hover
        label: "",
        "font-size": "7px",
        color: "#7a7e8e",
        "text-rotation": "autorotate",
        "text-outline-width": 1.5,
        "text-outline-color": "#08080f",
        "text-margin-y": -8,
        "overlay-padding": "3px",
      } as unknown as cytoscape.Css.Edge,
    },
    // Hovered edge -- show label
    {
      selector: "edge.hover-label",
      style: {
        label: "data(relation)",
        "font-size": "8px",
        color: "#c8ccd8",
        opacity: 1,
        width: "mapData(confidence, 0, 1, 2, 6)",
      } as unknown as cytoscape.Css.Edge,
    },
    // Highlighted edges (neighbors of selected node)
    {
      selector: "edge.highlighted",
      style: {
        opacity: 1,
        width: "mapData(confidence, 0, 1, 2.5, 6)",
        "z-index": 10,
      } as unknown as cytoscape.Css.Edge,
    },
    // Falsified edges
    {
      selector: "edge.falsified",
      style: {
        "line-color": "#E74C3C",
        "line-style": "dashed",
        "line-dash-pattern": [8, 6],
        opacity: 0.3,
        "target-arrow-color": "#E74C3C",
      } as cytoscape.Css.Edge,
    },
    // Dimmed edges
    {
      selector: "edge.dimmed",
      style: {
        opacity: 0.06,
      } as cytoscape.Css.Edge,
    },
  ] as cytoscape.Stylesheet[];
}

// ---------------------------------------------------------------------------
// Detail panel sub-components
// ---------------------------------------------------------------------------

interface NodeDetailProps {
  node: KGNode;
  onClose: () => void;
}

function NodeDetailPanel({ node, onClose }: NodeDetailProps) {
  const color = NODE_COLORS[node.type as NodeType] ?? "#6B7280";
  return (
    <div className="absolute left-4 top-4 z-30 w-72 rounded-xl border border-zinc-700/60 bg-[#0e0e18]/95 p-4 shadow-2xl backdrop-blur-md">
      <div className="mb-3 flex items-start justify-between">
        <div className="flex items-center gap-2">
          <span className="inline-block h-3 w-3 rounded-full shadow-lg" style={{ backgroundColor: color, boxShadow: `0 0 8px ${color}66` }} />
          <span className="text-[10px] font-semibold uppercase tracking-widest text-zinc-400">{node.type}</span>
        </div>
        <button onClick={onClose} className="rounded p-0.5 text-zinc-500 hover:text-zinc-300 transition-colors">
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
      <h3 className="mb-1 text-sm font-bold text-zinc-100">{node.name}</h3>
      {node.description && (
        <p className="mb-3 text-xs leading-relaxed text-zinc-400">{node.description}</p>
      )}
      <div className="mb-3 flex items-center gap-3">
        <div className="flex items-center gap-1.5">
          <span className="text-[10px] text-zinc-500">Confidence</span>
          <span className="text-xs font-semibold" style={{ color: confidenceColor(node.confidence) }}>
            {(node.confidence * 100).toFixed(0)}%
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-[10px] text-zinc-500">Sources</span>
          <span className="text-xs font-semibold text-zinc-300">{node.sources.length}</span>
        </div>
      </div>
      {node.sources.length > 0 && (
        <div className="max-h-32 overflow-y-auto">
          <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">Evidence</p>
          {node.sources.slice(0, 5).map((src, i) => (
            <div key={i} className="mb-1.5 rounded border border-zinc-800 bg-zinc-900/60 px-2 py-1.5">
              <p className="text-[10px] font-medium text-zinc-300">{truncate(src.claim || src.title || src.source_type, 60)}</p>
              <p className="text-[9px] text-zinc-500">{src.source_type} &middot; quality {(src.quality_score * 100).toFixed(0)}%</p>
            </div>
          ))}
          {node.sources.length > 5 && (
            <p className="text-[9px] text-zinc-600">+{node.sources.length - 5} more sources</p>
          )}
        </div>
      )}
      {Object.keys(node.external_ids).length > 0 && (
        <div className="mt-2 border-t border-zinc-800 pt-2">
          <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">External IDs</p>
          <div className="flex flex-wrap gap-1">
            {Object.entries(node.external_ids).slice(0, 4).map(([db, id]) => (
              <span key={db} className="rounded bg-zinc-800/80 px-1.5 py-0.5 text-[9px] text-zinc-400">
                {db}: {truncate(id, 16)}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

interface EdgeDetailProps {
  edge: KGEdge;
  sourceNode?: KGNode;
  targetNode?: KGNode;
  onClose: () => void;
}

function EdgeDetailPanel({ edge, sourceNode, targetNode, onClose }: EdgeDetailProps) {
  const visual = edgeVisualForRelation(edge.relation);
  return (
    <div className="absolute left-4 top-4 z-30 w-72 rounded-xl border border-zinc-700/60 bg-[#0e0e18]/95 p-4 shadow-2xl backdrop-blur-md">
      <div className="mb-3 flex items-start justify-between">
        <span className="text-[10px] font-semibold uppercase tracking-widest text-zinc-400">Edge</span>
        <button onClick={onClose} className="rounded p-0.5 text-zinc-500 hover:text-zinc-300 transition-colors">
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
      <div className="mb-2 flex items-center gap-1.5 text-xs text-zinc-200">
        <span className="font-medium">{sourceNode?.name ?? edge.source_id}</span>
        <span className="text-zinc-600">&rarr;</span>
        <span className="font-medium">{targetNode?.name ?? edge.target_id}</span>
      </div>
      <div className="mb-3 flex items-center gap-2">
        <span className="inline-block h-0.5 w-4 rounded" style={{ backgroundColor: visual.color, borderStyle: visual.dashed ? "dashed" : "solid" }} />
        <span className="text-[11px] font-semibold" style={{ color: visual.color }}>
          {edge.relation.replace(/_/g, " ")}
        </span>
        {edge.falsified && (
          <span className="rounded bg-red-900/40 px-1.5 py-0.5 text-[9px] font-bold text-red-400">FALSIFIED</span>
        )}
      </div>
      <div className="mb-3 grid grid-cols-2 gap-2">
        <div>
          <p className="text-[9px] text-zinc-500">Overall Confidence</p>
          <p className="text-sm font-bold" style={{ color: confidenceColor(edge.confidence.overall) }}>
            {(edge.confidence.overall * 100).toFixed(1)}%
          </p>
        </div>
        <div>
          <p className="text-[9px] text-zinc-500">Evidence Quality</p>
          <p className="text-sm font-bold" style={{ color: confidenceColor(edge.confidence.evidence_quality) }}>
            {(edge.confidence.evidence_quality * 100).toFixed(1)}%
          </p>
        </div>
        <div>
          <p className="text-[9px] text-zinc-500">Evidence Count</p>
          <p className="text-sm font-bold text-zinc-300">{edge.confidence.evidence_count}</p>
        </div>
        <div>
          <p className="text-[9px] text-zinc-500">Replications</p>
          <p className="text-sm font-bold text-zinc-300">{edge.confidence.replication_count}</p>
        </div>
      </div>
      {edge.confidence.falsification_attempts > 0 && (
        <div className="mb-3 rounded border border-zinc-800 bg-zinc-900/50 px-2.5 py-2">
          <p className="text-[9px] font-semibold uppercase tracking-wider text-zinc-500">Falsification</p>
          <p className="text-xs text-zinc-300">
            {edge.confidence.falsification_attempts} attempts &middot; {edge.confidence.falsification_failures} failures
          </p>
        </div>
      )}
      {edge.evidence.length > 0 && (
        <div className="max-h-28 overflow-y-auto">
          <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">Evidence</p>
          {edge.evidence.slice(0, 4).map((src, i) => (
            <div key={i} className="mb-1.5 rounded border border-zinc-800 bg-zinc-900/60 px-2 py-1.5">
              <p className="text-[10px] font-medium text-zinc-300">{truncate(src.claim || src.title || src.source_type, 60)}</p>
              <p className="text-[9px] text-zinc-500">{src.source_type}</p>
            </div>
          ))}
          {edge.evidence.length > 4 && (
            <p className="text-[9px] text-zinc-600">+{edge.evidence.length - 4} more</p>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function KnowledgeGraph({
  nodes, edges, timeRange, selectedHypothesisBranch,
  onNodeSelect, onEdgeSelect, className,
}: KnowledgeGraphProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);
  const animFrameRef = useRef<number>(0);
  const [layout, setLayout] = useState<"cose" | "cola" | "grid">("cose");
  const [selectedNode, setSelectedNode] = useState<KGNode | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<KGEdge | null>(null);
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

  // Compute degree map for node sizing
  const degreeMap = useMemo(() => {
    const m = new Map<string, number>();
    nodes.forEach((n) => m.set(n.id, 0));
    edges.forEach((e) => {
      m.set(e.source_id, (m.get(e.source_id) ?? 0) + 1);
      m.set(e.target_id, (m.get(e.target_id) ?? 0) + 1);
    });
    return m;
  }, [nodes, edges]);

  const maxDegree = useMemo(() => {
    let max = 1;
    degreeMap.forEach((d) => { if (d > max) max = d; });
    return max;
  }, [degreeMap]);

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

    // Cluster parents
    clusters.forEach((_, cid) => {
      els.push({ data: { id: `cluster-${cid}`, label: cid }, classes: "cluster-parent" });
    });

    // Nodes: sized by degree, colored by type
    filtered.nodes.forEach((n) => {
      const degree = degreeMap.get(n.id) ?? 0;
      const sizeNorm = maxDegree > 0 ? degree / maxDegree : 0;
      const size = 30 + sizeNorm * 50; // min 30px, max 80px

      const typeColor = NODE_COLORS[n.type as NodeType] ?? "#BDC3C7";
      const bgColor = n.hypothesis_branch ? branchColor(n.hypothesis_branch) : typeColor;
      const parent = n.viz.cluster_id ? `cluster-${n.viz.cluster_id}` : undefined;

      els.push({
        data: {
          id: n.id,
          label: truncate(n.name, 20),
          nodeType: n.type,
          size,
          borderColor: typeColor,
          bgColor,
          confidence: n.confidence,
          branch: n.hypothesis_branch ?? "",
          parent,
        },
      });
    });

    // Edges: styled by relation type and confidence
    filtered.edges.forEach((e) => {
      const conf = e.confidence.overall;
      const visual = edgeVisualForRelation(e.relation);

      // Width: 1-5 based on confidence
      const edgeWidth = 1 + conf * 4;
      // Opacity: 0.25 (low confidence) to 0.9 (high confidence)
      const edgeOpacity = e.falsified ? 0.25 : 0.25 + conf * 0.65;

      const lineStyle = e.falsified ? "dashed" : (visual.dashed ? "dashed" : "solid");
      const dashPattern = e.falsified ? [8, 6] : (visual.dashed ? [10, 5] : [1, 0]);
      const edgeColor = e.falsified ? "#E74C3C" : visual.color;

      // Animation speed for dash flow
      const speed = 1 + conf * 3;

      els.push({
        data: {
          id: e.id,
          source: e.source_id,
          target: e.target_id,
          relation: e.relation.replace(/_/g, " ").toLowerCase(),
          edgeColor,
          edgeWidth,
          edgeOpacity,
          dashPattern,
          lineStyle,
          confidence: conf,
          animSpeed: speed,
        },
        classes: e.falsified ? "falsified" : undefined,
      });
    });
    return els;
  }, [filtered, clusters, degreeMap, maxDegree]);

  // Layout runner
  const runLayout = useCallback(() => {
    const cy = cyRef.current;
    if (!cy || cy.elements().length === 0) return;

    let opts: cytoscape.LayoutOptions;
    if (layout === "cose") {
      opts = {
        name: "cose",
        animate: true,
        animationDuration: 800,
        fit: true,
        padding: 50,
        nodeRepulsion: () => 8000,
        idealEdgeLength: () => 100,
        edgeElasticity: () => 100,
        gravity: 0.25,
        numIter: 1000,
        randomize: false,
        componentSpacing: 80,
        nestingFactor: 1.2,
        nodeOverlap: 20,
      } as cytoscape.LayoutOptions;
    } else if (layout === "cola") {
      opts = {
        name: "cola" as const,
        animate: true,
        animationDuration: 800,
        nodeSpacing: 24,
        maxSimulationTime: 4000,
        randomize: false,
        convergenceThreshold: 0.01,
        fit: true,
        padding: 50,
      };
    } else {
      opts = {
        name: "grid" as const,
        animate: true,
        animationDuration: 400,
        fit: true,
        padding: 50,
        avoidOverlap: true,
      };
    }
    cy.layout(opts).run();
  }, [layout]);

  const debouncedLayout = useDebouncedCallback(runLayout, 200);

  // Edge dash animation (evidence flow)
  const startEdgeAnimation = useCallback(() => {
    const cy = cyRef.current;
    if (!cy) return;
    let offset = 0;
    const tick = () => {
      offset += 0.5;
      cy.edges().forEach((edge) => {
        const style = edge.data("lineStyle");
        if (style === "dashed") {
          edge.style("line-dash-offset", -offset * (edge.data("animSpeed") ?? 1));
        }
      });
      animFrameRef.current = requestAnimationFrame(tick);
    };
    animFrameRef.current = requestAnimationFrame(tick);
  }, []);

  // Neighbor highlighting
  const highlightNeighbors = useCallback((nodeId: string) => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.elements().removeClass("highlighted dimmed");
    const node = cy.getElementById(nodeId);
    const neighborhood = node.neighborhood().add(node);
    cy.elements().not(neighborhood).addClass("dimmed");
    neighborhood.addClass("highlighted");
    node.removeClass("dimmed");
  }, []);

  const clearHighlights = useCallback(() => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.elements().removeClass("highlighted dimmed");
  }, []);

  // Init / destroy Cytoscape
  useEffect(() => {
    if (!containerRef.current) return;
    const cy = cytoscape({
      container: containerRef.current,
      elements,
      style: buildStylesheet(),
      minZoom: 0.1,
      maxZoom: 6,
      wheelSensitivity: 0.25,
      boxSelectionEnabled: false,
    });
    cyRef.current = cy;
    runLayout();
    startEdgeAnimation();

    // Node click
    cy.on("tap", "node[nodeType]", (evt: EventObject) => {
      const kgNode = nodeMap.get(evt.target.id());
      if (kgNode) {
        setSelectedNode(kgNode);
        setSelectedEdge(null);
        highlightNeighbors(evt.target.id());
        if (onNodeSelect) onNodeSelect(kgNode);
      }
    });

    // Edge click
    cy.on("tap", "edge", (evt: EventObject) => {
      const kgEdge = edgeMap.get(evt.target.id());
      if (kgEdge) {
        setSelectedEdge(kgEdge);
        setSelectedNode(null);
        clearHighlights();
        if (onEdgeSelect) onEdgeSelect(kgEdge);
      }
    });

    // Background click -- deselect
    cy.on("tap", (evt: EventObject) => {
      if (evt.target === cy) {
        setSelectedNode(null);
        setSelectedEdge(null);
        clearHighlights();
      }
    });

    // Node hover -- tooltip
    cy.on("mouseover", "node[nodeType]", (evt: EventObject) => {
      const node = evt.target, pos = node.renderedPosition();
      setTooltip({
        x: pos.x, y: pos.y - node.renderedOuterHeight() / 2 - 12,
        name: node.data("label"), type: node.data("nodeType"), confidence: node.data("confidence"),
      });
      containerRef.current!.style.cursor = "pointer";
    });
    cy.on("mouseout", "node[nodeType]", () => {
      setTooltip(null);
      containerRef.current!.style.cursor = "default";
    });

    // Edge hover -- show label
    cy.on("mouseover", "edge", (evt: EventObject) => {
      evt.target.addClass("hover-label");
      containerRef.current!.style.cursor = "pointer";
    });
    cy.on("mouseout", "edge", (evt: EventObject) => {
      evt.target.removeClass("hover-label");
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
  const zoomIn = () => { const cy = cyRef.current; if (cy) cy.animate({ zoom: { level: cy.zoom() * 1.3, renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 } } }, { duration: 200 }); };
  const zoomOut = () => { const cy = cyRef.current; if (cy) cy.animate({ zoom: { level: cy.zoom() / 1.3, renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 } } }, { duration: 200 }); };
  const fitScreen = () => cyRef.current?.animate({ fit: { eles: cyRef.current.elements(), padding: 50 } }, { duration: 400 });
  const cycleLayout = () => setLayout((l) => l === "cose" ? "cola" : l === "cola" ? "grid" : "cose");

  useEffect(() => { debouncedLayout(); }, [layout, debouncedLayout]);

  // Build node type legend from visible nodes
  const visibleTypes = useMemo(() => {
    const types = new Set<NodeType>();
    filtered.nodes.forEach((n) => types.add(n.type));
    return Array.from(types).sort();
  }, [filtered.nodes]);

  // Empty state
  if (nodes.length === 0) {
    return (
      <div className={cn("relative flex items-center justify-center rounded-xl border border-zinc-800/50 bg-[#0a0a0f] text-zinc-500", className)}>
        <div className="flex flex-col items-center gap-3">
          <div className="relative">
            <Search className="h-12 w-12 opacity-20" />
            <div className="absolute inset-0 animate-ping opacity-10">
              <Search className="h-12 w-12" />
            </div>
          </div>
          <p className="text-sm font-medium">No knowledge graph data yet.</p>
          <p className="text-xs text-zinc-600">Start a research session to build the graph.</p>
        </div>
      </div>
    );
  }

  return (
    <div className={cn("relative overflow-hidden rounded-xl border border-zinc-800/40 bg-[#0a0a0f]", className)}>
      {/* Subtle gradient overlay for depth */}
      <div className="pointer-events-none absolute inset-0 z-0 bg-[radial-gradient(ellipse_at_center,_rgba(30,30,60,0.15)_0%,_transparent_70%)]" />

      {/* Cytoscape container */}
      <div ref={containerRef} className="relative z-10 h-full w-full" />

      {/* Node detail panel */}
      {selectedNode && (
        <NodeDetailPanel
          node={selectedNode}
          onClose={() => { setSelectedNode(null); clearHighlights(); }}
        />
      )}

      {/* Edge detail panel */}
      {selectedEdge && (
        <EdgeDetailPanel
          edge={selectedEdge}
          sourceNode={nodeMap.get(selectedEdge.source_id)}
          targetNode={nodeMap.get(selectedEdge.target_id)}
          onClose={() => setSelectedEdge(null)}
        />
      )}

      {/* Tooltip */}
      {tooltip && !selectedNode && !selectedEdge && (
        <div
          className="pointer-events-none absolute z-30 rounded-lg border border-zinc-700/50 bg-[#0e0e18]/95 px-3 py-2 shadow-2xl backdrop-blur-md"
          style={{ left: tooltip.x, top: tooltip.y, transform: "translate(-50%, -100%)" }}
        >
          <p className="text-xs font-semibold text-zinc-100">{tooltip.name}</p>
          <div className="mt-0.5 flex items-center gap-2">
            <span className="inline-block h-2 w-2 rounded-full shadow-sm"
              style={{ backgroundColor: NODE_COLORS[tooltip.type as NodeType] ?? "#BDC3C7", boxShadow: `0 0 4px ${NODE_COLORS[tooltip.type as NodeType] ?? "#BDC3C7"}66` }} />
            <span className="text-[10px] text-zinc-400">{tooltip.type}</span>
            <span className="text-[10px] text-zinc-600">|</span>
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
          {
            icon: layout === "grid" ? Network : LayoutGrid,
            action: cycleLayout,
            title: `Layout: ${layout}`,
          },
        ].map(({ icon: Icon, action, title }) => (
          <button key={title} onClick={action} title={title}
            className="flex h-8 w-8 items-center justify-center rounded-lg border border-zinc-700/50 bg-[#0e0e18]/80 text-zinc-400 backdrop-blur-sm transition-all hover:border-zinc-500 hover:text-zinc-200 hover:shadow-lg hover:shadow-zinc-900/50">
            <Icon className="h-4 w-4" />
          </button>
        ))}
      </div>

      {/* Legend -- node types + confidence */}
      <div className="absolute bottom-4 left-4 z-20 rounded-xl border border-zinc-800/50 bg-[#0e0e18]/90 p-3 backdrop-blur-md">
        {/* Node types */}
        <p className="mb-2 text-[9px] font-bold uppercase tracking-[0.15em] text-zinc-500">Node Types</p>
        <div className="mb-3 grid grid-cols-2 gap-x-3 gap-y-1">
          {visibleTypes.map((t) => (
            <div key={t} className="flex items-center gap-1.5">
              <span className="inline-block h-2 w-2 rounded-full shadow-sm"
                style={{ backgroundColor: NODE_COLORS[t], boxShadow: `0 0 4px ${NODE_COLORS[t]}44` }} />
              <span className="text-[9px] text-zinc-400">{t.replace(/_/g, " ")}</span>
            </div>
          ))}
        </div>
        {/* Confidence gradient */}
        <p className="mb-1 text-[9px] font-bold uppercase tracking-[0.15em] text-zinc-500">Edge Confidence</p>
        <div className="flex items-center gap-1.5">
          <span className="text-[9px] text-zinc-500">0%</span>
          <div className="h-2 w-24 rounded-full shadow-inner"
            style={{ background: "linear-gradient(to right, #E74C3C, #F39C12, #2ECC71)" }} />
          <span className="text-[9px] text-zinc-500">100%</span>
        </div>
        {/* Edge styles */}
        <div className="mt-2 flex items-center gap-3">
          <div className="flex items-center gap-1">
            <div className="h-[2px] w-4 rounded bg-green-500" />
            <span className="text-[9px] text-zinc-500">Supports</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="h-[2px] w-4 border-t-2 border-dashed border-red-500" />
            <span className="text-[9px] text-zinc-500">Inhibits</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="h-[2px] w-4 border-t-2 border-dashed border-red-500 opacity-40" />
            <span className="text-[9px] text-zinc-500">Falsified</span>
          </div>
        </div>
      </div>

      {/* Node count badge */}
      <div className="absolute right-4 top-4 z-20 flex items-center gap-2 rounded-lg border border-zinc-800/50 bg-[#0e0e18]/90 px-3 py-1.5 backdrop-blur-md">
        <div className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-500 shadow-sm shadow-emerald-500/50" />
        <span className="text-[10px] font-medium text-zinc-400">
          {filtered.nodes.length} nodes &middot; {filtered.edges.length} edges
        </span>
      </div>
    </div>
  );
}

export default KnowledgeGraph;
