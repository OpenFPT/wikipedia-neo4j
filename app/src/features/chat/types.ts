import type { KnowledgeBase, KnowledgeScope } from "@/features/knowledge/types";

export type ModelId = "local-qwen" | "openai" | "recommended";

export interface ChatModel {
  badge: "connected" | "imported" | "recommended";
  id: ModelId;
  kind: "local" | "provider";
  meta: string;
  name: string;
  status: "available" | "ready";
}

export interface EvidenceSource {
  excerpt: string;
  external: boolean;
  id: string;
  knowledgeBaseId: string;
  knowledgeBaseName: string;
  location: string;
  path: string[];
  score: number;
  title: string;
}

export interface RetrievalStage {
  id: string;
  label: string;
}

export interface ChatUsage {
  elapsedMs: number;
  model: ModelId;
  sources: number;
}

export type MockChatEvent =
  | { type: "stage"; state: "running" | "complete"; stage: RetrievalStage }
  | { type: "sources"; sources: EvidenceSource[] }
  | { type: "token"; value: string }
  | { type: "done"; usage: ChatUsage };

export interface ChatMessage {
  content: string;
  id: string;
  role: "assistant" | "user";
  sources?: EvidenceSource[];
  time: string;
  usage?: ChatUsage;
}

export interface ChatRequest {
  knowledgeBases?: KnowledgeBase[];
  knowledgeName?: string;
  locale: "en" | "vi";
  message: string;
  model: ModelId;
  scope: KnowledgeScope;
}
