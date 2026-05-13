import { NextResponse } from "next/server";
import { loginUser } from "@/lib/auth";

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const identifier = body.identifier || body.email;
    const password = body.password;
    if (!identifier || !password) return NextResponse.json({ error: "Email/username and password required" }, { status: 400 });
    const result = await loginUser(identifier, password);
    if ("error" in result) return NextResponse.json({ error: result.error }, { status: 401 });
    return NextResponse.json(result);
  } catch {
    return NextResponse.json({ error: "Internal error" }, { status: 500 });
  }
}
