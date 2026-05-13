import { NextResponse } from "next/server";
import elastic from "@/lib/elastic";

export async function GET() {
  try {
    const health = await elastic.cluster.health();
    return NextResponse.json({ status: "ok", elasticsearch: health.status });
  } catch (err) {
    return NextResponse.json({ status: "error", message: err instanceof Error ? err.message : "ES unavailable" }, { status: 503 });
  }
}
