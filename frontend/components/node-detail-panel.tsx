"use client";

import { X, ExternalLink, Shield, AlertTriangle } from "lucide-react";
import type { KGNode } from "@/lib/types";
import { NODE_COLORS } from "@/lib/types";
import { cn } from "@/lib/utils";

interface NodeDetailPanelProps {
  node: KGNode | null;
  onClose: () => void;
  className?: string;
}

export default function NodeDetailPanel({ node, onClose, className }: NodeDetailPanelProps) {
  if (!node) return null;

  const color = NODE_COLORS[node.type];

  return (
    <div
      className={cn(
        "absolute right-0 top-0 z-20 h-full w-80 overflow-y-auto border-l border-gray-800 bg-gray-950/95 p-4 backdrop-blur-sm",
        className,
      )}
    >
      <div className="mb-4 flex items-start justify-between">
        <div>
          <span
            className="mb-1 inline-block rounded-full px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider"
            style={{ backgroundColor: `${color}20`, color }}
          >
            {node.type}
          </span>
          <h3 className="mt-1 text-sm font-semibold text-white">{node.name}</h3>
        </div>
        <button
          onClick={onClose}
          className="rounded-lg p-1 text-gray-500 transition-colors hover:bg-gray-800 hover:text-gray-300"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      {node.description && (
        <p className="mb-4 text-xs leading-relaxed text-gray-400">{node.description}</p>
      )}

      {/* Confidence */}
      <div className="mb-4">
        <div className="mb-1 flex items-center justify-between text-xs">
          <span className="text-gray-500">Confidence</span>
          <span className="font-medium text-gray-300">{(node.confidence * 100).toFixed(0)}%</span>
        </div>
        <div className="h-1.5 overflow-hidden rounded-full bg-gray-800">
          <div
            className="h-full rounded-full transition-all"
            style={{
              width: `${node.confidence * 100}%`,
              backgroundColor: color,
            }}
          />
        </div>
      </div>

      {/* Aliases */}
      {node.aliases.length > 0 && (
        <div className="mb-4">
          <h4 className="mb-1 text-xs font-medium text-gray-500">Aliases</h4>
          <div className="flex flex-wrap gap-1">
            {node.aliases.map((a) => (
              <span key={a} className="rounded-md bg-gray-800 px-2 py-0.5 text-[10px] text-gray-400">
                {a}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* External IDs */}
      {Object.keys(node.external_ids).length > 0 && (
        <div className="mb-4">
          <h4 className="mb-1 text-xs font-medium text-gray-500">External IDs</h4>
          <div className="space-y-1">
            {Object.entries(node.external_ids).map(([db, id]) => (
              <div key={db} className="flex items-center justify-between text-[10px]">
                <span className="uppercase text-gray-500">{db}</span>
                <span className="font-mono text-gray-300">{id}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Sources */}
      {node.sources.length > 0 && (
        <div className="mb-4">
          <h4 className="mb-1 text-xs font-medium text-gray-500">
            Evidence Sources ({node.sources.length})
          </h4>
          <div className="space-y-2">
            {node.sources.slice(0, 5).map((src, i) => (
              <div key={i} className="rounded-lg bg-gray-900 p-2 text-[10px]">
                <div className="mb-1 flex items-center gap-1">
                  <span className="rounded bg-gray-800 px-1.5 py-0.5 text-gray-400">
                    {src.source_type}
                  </span>
                  {src.is_peer_reviewed && (
                    <Shield className="h-3 w-3 text-pathway" />
                  )}
                </div>
                {src.title && (
                  <p className="line-clamp-2 text-gray-300">{src.title}</p>
                )}
                {src.url && (
                  <a
                    href={src.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="mt-1 flex items-center gap-1 text-protein hover:underline"
                  >
                    <ExternalLink className="h-2.5 w-2.5" />
                    View source
                  </a>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Metadata */}
      <div className="text-[10px] text-gray-600">
        <p>Created by: {node.created_by || "system"}</p>
        {node.hypothesis_branch && <p>Branch: {node.hypothesis_branch}</p>}
        <p>Created: {new Date(node.created_at).toLocaleString()}</p>
      </div>
    </div>
  );
}
