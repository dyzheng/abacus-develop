import OpenAI from "openai";
import { db } from "@/db";
import { aiLogs } from "@/db/schema";
import { config } from "@/lib/config";
import { v4 as uuid } from "uuid";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY || "",
});

interface AIChatOptions {
  projectId?: string;
  action: string;
  systemPrompt: string;
  userPrompt: string;
  temperature?: number;
  jsonMode?: boolean;
  maxRetries?: number;
}

export async function aiChat(options: AIChatOptions): Promise<{ content: string; tokensUsed: number }> {
  const { projectId, action, systemPrompt, userPrompt, temperature = 0.3, jsonMode = true, maxRetries = 3 } = options;
  const now = new Date().toISOString();
  const logId = uuid();

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const response = await openai.chat.completions.create({
        model: config.aiModel,
        temperature,
        ...(jsonMode ? { response_format: { type: "json_object" } } : {}),
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: userPrompt },
        ],
      });

      const content = response.choices[0]?.message?.content || "";
      const tokensUsed = response.usage?.total_tokens || 0;

      await db.insert(aiLogs).values({
        id: logId,
        projectId: projectId || null,
        action,
        prompt: userPrompt.slice(0, 10000),
        response: content.slice(0, 10000),
        tokensUsed,
        model: config.aiModel,
        status: "success",
        createdAt: now,
      });

      return { content, tokensUsed };
    } catch (error: any) {
      if (attempt === maxRetries - 1) {
        await db.insert(aiLogs).values({
          id: logId,
          projectId: projectId || null,
          action,
          prompt: userPrompt.slice(0, 10000),
          response: null,
          tokensUsed: 0,
          model: config.aiModel,
          status: "error",
          errorMessage: error.message,
          createdAt: now,
        });
        throw error;
      }
      // Exponential backoff
      await new Promise((r) => setTimeout(r, 1000 * Math.pow(2, attempt)));
    }
  }

  throw new Error("AI请求失败");
}
