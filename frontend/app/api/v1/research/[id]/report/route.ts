import { NextResponse } from "next/server";
import { MOCK_SESSION } from "@/lib/mock-data";

export async function GET() {
  return NextResponse.json({
    format: "markdown",
    content: MOCK_SESSION.result?.report_markdown ?? "",
  });
}
