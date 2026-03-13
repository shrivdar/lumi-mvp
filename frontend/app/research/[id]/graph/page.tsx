"use client";

import { useParams, useRouter } from "next/navigation";
import { useState, useMemo } from "react";
import { ArrowLeft, Maximize2, GitBranch } from "lucide-react";
import Link from "next/link";
import { useFetch, useResearchSession, useResearchWebSocket } from "@/lib/hooks";
import type { KGNode, KGEdge, HypothesisNode } from "@/lib/types";
import KnowledgeGraph from "@/components/knowledge-graph";
import NodeDetailPanel from "@/components/node-detail-panel";
import TemporalSlider from "@/components/temporal-slider";
import { cn } from "@/lib/utils";

interface GraphData {
  nodes: KGNode[];
  edges: KGEdge[];
}

export default function GraphPage() {
  const params = useParams();
  const router = useRouter();
  const id = params.id as string;

  const { data: session } = useResearchSession(id);
  const { data: graph } = useFetch<GraphData>(`/api/v1/research/${id}/graph?format=json`);
  const { data: hypotheses } = useFetch<HypothesisNode[]>(`/api/v1/research/${id}/hypotheses`);

  const [selectedNode, setSelectedNode] = useState<KGNode | null>(null);
  const [timeRange, setTimeRange] = useState<[string, string] | undefined>();
  const [selectedBranch, setSelectedBranch] = useState<string | null>(null);

  const nodes = useMemo(() => graph?.nodes ?? [], [graph]);
  const edges = useMemo(() => graph?.edges ?? [], [graph]);

  const timestamps = useMemo(
    () => [
      ...nodes.map((n) => n.created_at),
      ...edges.map((e) => e.created_at),
    ],
    [nodes, edges],
  );

  const branches = useMemo(() => {
    const set = new Set<string>();
    nodes.forEach((n) => { if (n.hypothesis_branch) set.add(n.hypothesis_branch); });
    return Array.from(set);
  }, [nodes]);

  return (
    <div className="flex h-[calc(100vh-3.5rem)] flex-col">
      {/* Toolbar */}
      <div className="flex items-center gap-3 border-b border-gray-800 bg-gray-950 px-4 py-2">
        <button
          onClick={() => router.push(`/research/${id}`)}
          className="rounded-lg p-1.5 text-gray-500 hover:bg-gray-800 hover:text-gray-300"
        >
          <ArrowLeft className="h-4 w-4" />
        </button>
        <div className="flex-1">
          <h1 className="text-sm font-medium text-white">Knowledge Graph</h1>
          <p className="text-[10px] text-gray-500">
            {nodes.length} nodes, {edges.length} edges
            {session ? ` — ${session.query.slice(0, 60)}...` : ""}
          </p>
        </div>

        {/* Branch filter */}
        {branches.length > 0 && (
          <div className="flex items-center gap-1">
            <GitBranch className="h-3.5 w-3.5 text-gray-500" />
            <select
              value={selectedBranch ?? ""}
              onChange={(e) => setSelectedBranch(e.target.value || null)}
              className="rounded-lg border border-gray-700 bg-gray-900 px-2 py-1 text-xs text-gray-300"
            >
              <option value="">All branches</option>
              {branches.map((b) => (
                <option key={b} value={b}>
                  {b.slice(0, 30)}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      {/* Graph area */}
      <div className="relative flex-1">
        {nodes.length === 0 ? (
          <div className="flex h-full items-center justify-center text-sm text-gray-500">
            {graph ? "No nodes in the knowledge graph yet." : "Loading graph..."}
          </div>
        ) : (
          <KnowledgeGraph
            nodes={nodes}
            edges={edges}
            timeRange={timeRange}
            selectedHypothesisBranch={selectedBranch}
            onNodeSelect={setSelectedNode}
            className="h-full w-full"
          />
        )}

        <NodeDetailPanel
          node={selectedNode}
          onClose={() => setSelectedNode(null)}
        />
      </div>

      {/* Temporal slider */}
      {timestamps.length > 1 && (
        <div className="border-t border-gray-800 bg-gray-950 px-4 py-2">
          <TemporalSlider timestamps={timestamps} onChange={setTimeRange} />
        </div>
      )}
    </div>
  );
}
