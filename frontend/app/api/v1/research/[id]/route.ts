import { NextResponse } from "next/server";
import { MOCK_SESSION, MOCK_RUNNING_SESSION } from "@/lib/mock-data";

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  if (id === MOCK_RUNNING_SESSION.id) return NextResponse.json(MOCK_RUNNING_SESSION);
  return NextResponse.json(MOCK_SESSION);
}
