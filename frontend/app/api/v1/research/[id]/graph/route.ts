import { NextResponse } from "next/server";
import { MOCK_NODES, MOCK_EDGES } from "@/lib/mock-data";

export async function GET() {
  return NextResponse.json({ nodes: MOCK_NODES, edges: MOCK_EDGES });
}
