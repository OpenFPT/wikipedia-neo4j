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

export type StageStatus = "complete" | "pending" | "running";

export interface ChatUsage {
  elapsedMs: number;
  model: ModelId;
  sources: number;
}

export interface ChatTraceStep {
  errorType?: string;
  kind: string;
  label: string;
  rowCount?: number;
  status: string;
  topK?: number;
}

export interface ChatTrace {
  mode?: string;
  route?: string;
  steps: ChatTraceStep[];
  tier?: string;
}

export interface ChatThinkingState {
  stageStatuses: Record<string, StageStatus>;
  stages: RetrievalStage[];
}

export interface ChatBackendEvent {
  detail?: string;
  id: string;
  query?: string;
  title: string;
  type:
    | "answer_delta"
    | "citation"
    | "cypher"
    | "error"
    | "fallback"
    | "final"
    | "route"
    | "status"
    | "tool_call"
    | "tool_result";
}

export type MockChatEvent =
  | { type: "stage"; state: "running" | "complete"; stage: RetrievalStage }
  | { type: "sources"; sources: EvidenceSource[] }
  | { type: "token"; value: string }
  | { type: "done"; usage: ChatUsage };

export interface ChatMessage {
  backendEvents?: ChatBackendEvent[];
  content: string;
  id: string;
  role: "assistant" | "user";
  sources?: EvidenceSource[];
  thinking?: ChatThinkingState;
  time: string;
  trace?: ChatTrace;
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
