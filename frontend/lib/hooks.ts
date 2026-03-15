"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { apiFetch, wsUrl } from "./api";
import type { ResearchEvent, ResearchSession } from "./types";

// ---------------------------------------------------------------------------
// Generic data fetching
// ---------------------------------------------------------------------------

interface UseFetchResult<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
  refetch: () => void;
}

export function useFetch<T>(path: string | null): UseFetchResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(!!path);

  const refetch = useCallback(() => {
    if (!path) return;
    setLoading(true);
    setError(null);
    apiFetch<T>(path)
      .then(setData)
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, [path]);

  useEffect(() => {
    if (!path) return;
    let cancelled = false;
    // Using a local async function to properly handle fetch lifecycle
    const doFetch = async () => {
      try {
        const result = await apiFetch<T>(path);
        if (!cancelled) setData(result);
      } catch (e) {
        if (!cancelled) setError((e as Error).message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    doFetch();
    return () => { cancelled = true; };
  }, [path]);

  return { data, error, loading, refetch };
}

// ---------------------------------------------------------------------------
// Polling
// ---------------------------------------------------------------------------

export function usePolling<T>(path: string | null, intervalMs = 5000): UseFetchResult<T> {
  const result = useFetch<T>(path);
  useEffect(() => {
    if (!path) return;
    const id = setInterval(result.refetch, intervalMs);
    return () => clearInterval(id);
  }, [path, intervalMs, result.refetch]);
  return result;
}

// ---------------------------------------------------------------------------
// WebSocket for live research events
// ---------------------------------------------------------------------------

interface UseWebSocketResult {
  events: ResearchEvent[];
  connected: boolean;
  error: string | null;
}

export function useResearchWebSocket(sessionId: string | null): UseWebSocketResult {
  const [events, setEvents] = useState<ResearchEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!sessionId) return;

    const url = wsUrl(`/api/v1/research/${sessionId}/ws`);
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      setError(null);
    };

    ws.onmessage = (msg) => {
      try {
        const event: ResearchEvent = JSON.parse(msg.data);
        setEvents((prev) => [...prev, event]);
      } catch {
        // ignore malformed messages
      }
    };

    ws.onerror = () => setError("WebSocket connection error");
    ws.onclose = () => setConnected(false);

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [sessionId]);

  return { events, connected, error };
}

// ---------------------------------------------------------------------------
// Research sessions
// ---------------------------------------------------------------------------

export function useResearchSessions() {
  return usePolling<{ items: ResearchSession[]; total: number }>(
    "/api/v1/research",
    10000,
  );
}

export function useResearchSession(id: string | null) {
  return usePolling<ResearchSession>(
    id ? `/api/v1/research/${id}` : null,
    5000,
  );
}

// ---------------------------------------------------------------------------
// Interval-based animation timer
// ---------------------------------------------------------------------------

export function useAnimationTimer(intervalMs = 50) {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setTick((t) => t + 1), intervalMs);
    return () => clearInterval(id);
  }, [intervalMs]);
  return tick;
}

// ---------------------------------------------------------------------------
// Monitoring hooks — wire frontend components to real API data
// ---------------------------------------------------------------------------

export interface MonitoringOverview {
  sessions: { total: number; active: number; completed: number; failed: number };
  knowledge_graph: { total_nodes: number; total_edges: number };
  tokens: { total_used: number };
  agents: { total_spawned: number };
}

export interface ResearchStats {
  research_id: string;
  status: string;
  current_iteration: number;
  agents: {
    total_spawned: number;
    max_total: number;
    type_counts: Record<string, number>;
  };
  hypothesis_tree: {
    node_count: number;
    total_visits: number;
    max_depth: number;
    confirmed_count: number;
    refuted_count: number;
    exploring_count: number;
    unexplored_count: number;
    best_hypothesis?: {
      id: string;
      hypothesis: string;
      confidence: number;
      avg_info_gain: number;
    };
  };
  knowledge_graph: {
    node_count: number;
    edge_count: number;
    avg_confidence: number;
  };
  uncertainty: {
    trend: string;
    composites: number[];
    latest: number;
    mean: number;
    hitl_triggered: boolean;
    hitl_response_count: number;
  };
  tokens: {
    session_tokens_used: number;
    session_token_budget: number;
    budget_utilization: number;
  };
}

export interface AgentConstellationData {
  research_id: string;
  agents: AgentInfo[];
  total_spawned: number;
}

export interface UncertaintyData {
  research_id: string;
  current: UncertaintyVector;
  history: UncertaintyVector[];
  trend: string;
  hitl_triggered: boolean;
  hitl_response_count: number;
}

import type { AgentInfo, UncertaintyVector } from "./types";

/** Poll global monitoring overview every 10s. */
export function useMonitoringOverview() {
  return usePolling<MonitoringOverview>("/api/v1/monitoring/overview", 10000);
}

/** Poll research-specific stats every 3s (for active sessions). */
export function useResearchStats(researchId: string | null) {
  return usePolling<ResearchStats>(
    researchId ? `/api/v1/monitoring/research/${researchId}/stats` : null,
    3000,
  );
}

/** Poll agent constellation data every 3s. */
export function useAgentConstellation(researchId: string | null) {
  return usePolling<AgentConstellationData>(
    researchId ? `/api/v1/monitoring/research/${researchId}/agents` : null,
    3000,
  );
}

/** Poll uncertainty radar data every 3s. */
export function useUncertaintyRadar(researchId: string | null) {
  return usePolling<UncertaintyData>(
    researchId ? `/api/v1/monitoring/research/${researchId}/uncertainty` : null,
    3000,
  );
}

/** Send HITL response via WebSocket. */
export function useSendHITLResponse(sessionId: string | null) {
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!sessionId) return;
    const url = wsUrl(`/api/v1/research/${sessionId}/ws`);
    const ws = new WebSocket(url);
    wsRef.current = ws;
    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [sessionId]);

  const sendResponse = useCallback(
    (response: string) => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(
          JSON.stringify({ type: "hitl_response", response }),
        );
      }
    },
    [],
  );

  return sendResponse;
}
