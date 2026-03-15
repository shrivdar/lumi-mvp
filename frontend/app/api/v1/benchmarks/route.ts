import { NextResponse } from "next/server";
import { MOCK_BENCHMARKS } from "@/lib/mock-data";

export async function GET() {
  return NextResponse.json(MOCK_BENCHMARKS);
}
