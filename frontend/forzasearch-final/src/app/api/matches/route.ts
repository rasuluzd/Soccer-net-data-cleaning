import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

export async function GET() {
  try {
    const p = path.join(process.cwd(), "matches", "registry.json");
    const registry = JSON.parse(fs.readFileSync(p, "utf-8"));
    const matches = registry.matches.map((m: { id: string; title: string; subtitle: string; date: string }) => ({
      id: m.id, title: m.title, subtitle: m.subtitle, date: m.date,
    }));
    return NextResponse.json(matches);
  } catch {
    return NextResponse.json([]);
  }
}
