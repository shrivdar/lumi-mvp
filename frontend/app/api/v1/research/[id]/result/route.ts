import { NextResponse } from "next/server";
import { MOCK_SESSION } from "@/lib/mock-data";

export async function GET() {
  return NextResponse.json(MOCK_SESSION.result);
}
