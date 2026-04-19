import { NextResponse } from "next/server";
import { verifyToken } from "@/lib/auth";

export async function GET(req: Request) {
  const auth = req.headers.get("authorization");
  const token = auth?.replace("Bearer ", "");
  if (!token) return NextResponse.json({ error: "No token" }, { status: 401 });

  const user = await verifyToken(token);
  if (!user) return NextResponse.json({ error: "Invalid token" }, { status: 401 });
  return NextResponse.json({ user });
}
