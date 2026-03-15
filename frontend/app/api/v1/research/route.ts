import { NextResponse } from "next/server";
import { MOCK_SESSION, MOCK_RUNNING_SESSION } from "@/lib/mock-data";

export async function GET() {
  return NextResponse.json({
    items: [MOCK_RUNNING_SESSION, MOCK_SESSION],
    total: 2,
  });
}

export async function POST(req: Request) {
  const body = await req.json();
  return NextResponse.json({
    ...MOCK_RUNNING_SESSION,
    id: `session-${Date.now()}`,
    query: body.query ?? "New research session",
  });
}
