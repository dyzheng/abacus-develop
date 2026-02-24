import { NextResponse } from "next/server";
import { z } from "zod";

/**
 * Standardized API error response handler.
 * - ZodError → 400 with validation details
 * - Known business errors (message starts with known prefix) → 400
 * - Everything else → 500
 */
export function handleApiError(error: unknown, fallbackMessage: string) {
  if (error instanceof z.ZodError) {
    return NextResponse.json({ error: error.issues }, { status: 400 });
  }

  const message = error instanceof Error ? error.message : fallbackMessage;
  console.error(`[API Error] ${fallbackMessage}:`, error);
  return NextResponse.json({ error: message || fallbackMessage }, { status: 500 });
}
