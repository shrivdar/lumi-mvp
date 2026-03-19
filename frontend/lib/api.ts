const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "dev-api-key-change-me";

// Use relative URLs so requests go through Next.js rewrites (server-side proxy).
// This works regardless of whether the browser is on localhost or a remote VM.
// For direct backend access (SSR or scripts), fall back to the env var.
const API_BASE = typeof window !== "undefined"
  ? ""  // browser: relative URL → Next.js rewrite proxy
  : (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000");  // server: direct

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": API_KEY,
      ...init?.headers,
    },
  });
  if (!res.ok) {
    throw new Error(`API error ${res.status}: ${await res.text()}`);
  }
  return res.json();
}

export function wsUrl(path: string): string {
  // WebSocket must use the actual host since Next.js can't proxy WS
  if (typeof window !== "undefined") {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.hostname;
    return `${proto}//${host}:8000${path}`;
  }
  const base = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";
  return `${base}${path}`;
}
