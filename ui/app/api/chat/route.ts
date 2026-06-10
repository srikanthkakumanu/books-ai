/**
 * ui/app/api/chat/route.ts
 *
 * Backend For Frontend (BFF) chat endpoint.
 *
 * Responsibilities:
 * 1. Validate the user's JWT
 * 2. Validate the request body
 * 3. Forward to the Agent Orchestrator with user context
 * 4. Stream the SSE response back to the browser
 *
 * This is the ONLY entry point from the browser to the AI backend.
 * Never call the agent, MCP server, or microservices from client components.
 */

import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import { verifyJWT } from "@/lib/auth";
import { createAgentStream } from "@/lib/stream";

const ChatRequestSchema = z.object({
  message: z.string().min(1).max(2000),
  sessionId: z.string().uuid(),
});

export async function POST(req: NextRequest): Promise<Response> {
  // ── 1. Authenticate ──────────────────────────────────────────────────────
  const authHeader = req.headers.get("Authorization");
  if (!authHeader?.startsWith("Bearer ")) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const token = authHeader.slice(7);
  const payload = await verifyJWT(token);
  if (!payload) {
    return NextResponse.json({ error: "Invalid token" }, { status: 401 });
  }

  const userId = payload.sub as string;

  // ── 2. Validate body ──────────────────────────────────────────────────────
  let body: z.infer<typeof ChatRequestSchema>;
  try {
    const raw = await req.json();
    body = ChatRequestSchema.parse(raw);
  } catch {
    return NextResponse.json({ error: "Invalid request body" }, { status: 422 });
  }

  // ── 3. Forward to Agent Orchestrator with streaming ───────────────────────
  const agentUrl = `${process.env.AGENT_URL}/chat`;

  const agentResponse = await fetch(agentUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      // Agent authenticates the MCP server using its own service-to-service JWT
      Authorization: `Bearer ${process.env.AGENT_SERVICE_TOKEN}`,
    },
    body: JSON.stringify({
      user_id: userId,
      session_id: body.sessionId,
      message: body.message,
    }),
  });

  if (!agentResponse.ok) {
    return NextResponse.json(
      { error: "Agent unavailable" },
      { status: 503 }
    );
  }

  // ── 4. Stream SSE response back to the browser ────────────────────────────
  // TransformStream passes agent SSE tokens through unchanged
  const { readable, writable } = new TransformStream();

  agentResponse.body?.pipeTo(writable);

  return new Response(readable, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}
