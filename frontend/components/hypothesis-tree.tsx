"use client";

import { useCallback, useEffect, useMemo, useRef } from "react";
import * as d3 from "d3";
import { Check, X } from "lucide-react";
import { cn } from "@/lib/utils";
import type { HypothesisNode, HypothesisStatus } from "@/lib/types";

interface HypothesisTreeProps {
  hypotheses: HypothesisNode[];
  onSelect?: (node: HypothesisNode) => void;
  selectedId?: string;
  className?: string;
}

interface TreeDatum {
  id: string;
  node: HypothesisNode;
  children?: TreeDatum[];
}

const CONFIDENCE_COLORS = ["#E74C3C", "#F39C12", "#2ECC71"] as const;
const confidenceScale = d3
  .scaleLinear<string>()
  .domain([0, 0.5, 1])
  .range([...CONFIDENCE_COLORS])
  .clamp(true);

function nodeRadius(visitCount: number): number {
  return Math.min(60, Math.max(20, 10 + visitCount * 2));
}

function glowIntensity(visitCount: number, maxVisits: number): number {
  if (maxVisits === 0) return 0;
  return Math.min(1, visitCount / maxVisits);
}

function buildTree(nodes: HypothesisNode[]): TreeDatum | null {
  if (nodes.length === 0) return null;
  const map = new Map<string, TreeDatum>();
  for (const n of nodes) {
    map.set(n.id, { id: n.id, node: n, children: [] });
  }
  let root: TreeDatum | null = null;
  for (const n of nodes) {
    const datum = map.get(n.id)!;
    if (n.parent_id && map.has(n.parent_id)) {
      map.get(n.parent_id)!.children!.push(datum);
    } else {
      root = datum;
    }
  }
  return root;
}

function ancestorIds(
  nodeId: string,
  nodeMap: Map<string, HypothesisNode>
): Set<string> {
  const ids = new Set<string>();
  let cur = nodeId;
  while (cur) {
    ids.add(cur);
    const n = nodeMap.get(cur);
    if (!n || !n.parent_id) break;
    cur = n.parent_id;
  }
  return ids;
}

export function HypothesisTree({
  hypotheses,
  onSelect,
  selectedId,
  className,
}: HypothesisTreeProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  const nodeMap = useMemo(() => {
    const m = new Map<string, HypothesisNode>();
    for (const h of hypotheses) m.set(h.id, h);
    return m;
  }, [hypotheses]);

  const maxVisits = useMemo(
    () => Math.max(1, ...hypotheses.map((h) => h.visit_count)),
    [hypotheses]
  );

  const selectedPath = useMemo(
    () => (selectedId ? ancestorIds(selectedId, nodeMap) : new Set<string>()),
    [selectedId, nodeMap]
  );

  const render = useCallback(() => {
    const svg = d3.select(svgRef.current);
    if (!svgRef.current) return;

    const root = buildTree(hypotheses);
    if (!root) {
      svg.selectAll("*").remove();
      return;
    }

    const { width, height } = svgRef.current.getBoundingClientRect();
    const margin = { top: 40, right: 80, bottom: 40, left: 80 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const hierarchy = d3
      .hierarchy<TreeDatum>(root, (d: TreeDatum) =>
        d.children && d.children.length > 0 ? d.children : undefined
      );

    const treeLayout = d3.tree<TreeDatum>().size([innerH, innerW]);
    const treeData = treeLayout(hierarchy);

    svg.selectAll("*").remove();

    // Defs for glow filter and link gradient
    const defs = svg.append("defs");

    // Glow filters per intensity bucket (0..10)
    for (let i = 0; i <= 10; i++) {
      const f = defs
        .append("filter")
        .attr("id", `glow-${i}`)
        .attr("x", "-50%")
        .attr("y", "-50%")
        .attr("width", "200%")
        .attr("height", "200%");
      f.append("feGaussianBlur")
        .attr("stdDeviation", i * 1.5)
        .attr("result", "blur");
      const merge = f.append("feMerge");
      merge.append("feMergeNode").attr("in", "blur");
      merge.append("feMergeNode").attr("in", "SourceGraphic");
    }

    // Pulse animation style
    defs
      .append("style")
      .text(
        `@keyframes pulse { 0%,100% { stroke-opacity: 0.4; } 50% { stroke-opacity: 1; } }
         .pulse-border { animation: pulse 1.5s ease-in-out infinite; }`
      );

    // Link gradient
    const grad = defs
      .append("linearGradient")
      .attr("id", "link-grad")
      .attr("x1", "0%")
      .attr("x2", "100%");
    grad.append("stop").attr("offset", "0%").attr("stop-color", "#475569").attr("stop-opacity", 0.6);
    grad.append("stop").attr("offset", "100%").attr("stop-color", "#475569").attr("stop-opacity", 0.2);

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Links
    const linkGen = d3
      .linkHorizontal<
        d3.HierarchyPointLink<TreeDatum>,
        d3.HierarchyPointNode<TreeDatum>
      >()
      .source((d: d3.HierarchyPointLink<TreeDatum>) => d.source)
      .target((d: d3.HierarchyPointLink<TreeDatum>) => d.target)
      .x((d: d3.HierarchyPointNode<TreeDatum>) => d.y)
      .y((d: d3.HierarchyPointNode<TreeDatum>) => d.x);

    g.selectAll("path.link")
      .data(treeData.links())
      .join("path")
      .attr("class", "link")
      .attr("d", linkGen as unknown as string)
      .attr("fill", "none")
      .attr("stroke", (d) => {
        const srcId = d.source.data.id;
        const tgtId = d.target.data.id;
        return selectedPath.has(srcId) && selectedPath.has(tgtId)
          ? "#60A5FA"
          : "url(#link-grad)";
      })
      .attr("stroke-width", (d) =>
        selectedPath.has(d.source.data.id) && selectedPath.has(d.target.data.id)
          ? 3
          : 1.5
      )
      .attr("opacity", 0)
      .transition()
      .duration(500)
      .attr("opacity", 1);

    // Node groups
    const nodeGroups = g
      .selectAll<SVGGElement, d3.HierarchyPointNode<TreeDatum>>("g.node")
      .data(treeData.descendants(), (d) =>
        (d as d3.HierarchyPointNode<TreeDatum>).data.id
      )
      .join(
        (enter) => {
          const group = enter
            .append("g")
            .attr("class", "node")
            .attr("transform", (d) => `translate(${d.y},${d.x})`)
            .attr("opacity", 0)
            .style("cursor", "pointer");
          group.transition().duration(500).attr("opacity", 1);
          return group;
        },
        (update) =>
          update
            .transition()
            .duration(500)
            .attr("transform", (d) => `translate(${d.y},${d.x})`)
            .attr("opacity", 1),
        (exit) => exit.transition().duration(300).attr("opacity", 0).remove()
      );

    // Draw circles
    nodeGroups.each(function (d) {
      const el = d3.select(this);
      el.selectAll("*").remove();

      const h = d.data.node;
      const r = nodeRadius(h.visit_count);
      const gBucket = Math.round(glowIntensity(h.visit_count, maxVisits) * 10);
      const fillColor = confidenceScale(h.confidence);
      const isPruned = h.status === "PRUNED";
      const isUnexplored = h.status === "UNEXPLORED";
      const isExploring = h.status === "EXPLORING";
      const isSelected = h.id === selectedId;
      const onPath = selectedPath.has(h.id);

      // Main circle
      el.append("circle")
        .attr("r", r)
        .attr("fill", isPruned ? "#1E293B" : isUnexplored ? "none" : fillColor)
        .attr("stroke", isPruned ? "#475569" : fillColor)
        .attr("stroke-width", isSelected ? 4 : onPath ? 3 : 2)
        .attr("stroke-dasharray", isPruned ? "4 3" : "none")
        .attr("filter", `url(#glow-${gBucket})`)
        .attr("opacity", isPruned ? 0.4 : 1)
        .classed("pulse-border", isExploring);

      // Status overlays
      if (h.status === "CONFIRMED") {
        el.append("foreignObject")
          .attr("x", -8)
          .attr("y", -8)
          .attr("width", 16)
          .attr("height", 16)
          .append("xhtml:div")
          .style("color", "#FFFFFF")
          .style("display", "flex")
          .style("align-items", "center")
          .style("justify-content", "center")
          .html(
            '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>'
          );
      }

      if (h.status === "REFUTED") {
        el.append("foreignObject")
          .attr("x", -8)
          .attr("y", -8)
          .attr("width", 16)
          .attr("height", 16)
          .append("xhtml:div")
          .style("color", "#FFFFFF")
          .style("display", "flex")
          .style("align-items", "center")
          .style("justify-content", "center")
          .html(
            '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>'
          );
      }

      // UCB score label
      el.append("text")
        .attr("x", r + 6)
        .attr("y", -4)
        .attr("font-size", 10)
        .attr("fill", "#94A3B8")
        .attr("font-family", "monospace")
        .text(`UCB ${h.ucb_score.toFixed(2)}`);

      // Hypothesis label (truncated)
      el.append("text")
        .attr("x", r + 6)
        .attr("y", 10)
        .attr("font-size", 11)
        .attr("fill", isPruned ? "#64748B" : "#CBD5E1")
        .attr("text-decoration", isPruned ? "line-through" : "none")
        .text(
          h.hypothesis.length > 40
            ? h.hypothesis.slice(0, 37) + "..."
            : h.hypothesis
        );

      // Click handler
      el.on("click", (event: MouseEvent) => {
        event.stopPropagation();
        onSelect?.(h);
      });
    });
  }, [hypotheses, maxVisits, selectedId, selectedPath, onSelect]);

  useEffect(() => {
    render();
  }, [render]);

  // Resize observer
  useEffect(() => {
    if (!svgRef.current) return;
    const observer = new ResizeObserver(() => render());
    observer.observe(svgRef.current);
    return () => observer.disconnect();
  }, [render]);

  return (
    <svg
      ref={svgRef}
      className={cn("w-full h-full min-h-[400px] bg-slate-950 rounded-lg", className)}
    />
  );
}
