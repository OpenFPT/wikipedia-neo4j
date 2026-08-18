import env from "@/config/env";
import { type Locale, translate } from "@/features/i18n/i18n";
import type {
  ChatBackendEvent,
  ChatTrace,
  EvidenceSource,
  ModelId,
} from "../types";

type BackendCitation = {
  chunk_id: string;
  page_title: string;
  page_url: string;
};

type BackendChatResponse = {
  answer: string;
  citations: BackendCitation[];
  retrieval_tier?: string;
  debug?: { retrieval_tier?: string };
  trace?: {
    mode?: string;
    route?: string;
    steps?: Array<{
      error_type?: string;
      kind?: string;
      label?: string;
      name?: string;
      row_count?: number;
      status?: string;
      top_k?: number;
    }>;
    tier?: string;
  };
};

type BackendStreamBlock = {
  data: Record<string, unknown>;
  event: string;
};

const SAFE_ROUTE_LABELS: Record<string, string> = {
  local_agent: "traceRouteLocalAgent",
  "route.local_agent": "traceRouteLocalAgent",
};

const SAFE_TIER_LABELS: Record<string, string> = {
  generated: "traceTierGenerated",
  wrrf: "traceTierWrrf",
};

const SAFE_MODE_LABELS: Record<string, string> = {
  api: "traceModeApi",
  local: "traceModeLocal",
  ollama: "traceModeOllama",
  unknown: "traceModeUnknownValue",
};

const SAFE_STEP_LABELS: Record<string, string> = {
  "answer.agent_final": "traceStepAgentFinal",
  "fallback.deterministic_retrieval": "traceStepDeterministicFallback",
  "fallback.synthesized_observations": "traceStepObservationFallback",
  generated: "traceStepGeneratedRetrieval",
  generated_query_failed: "traceStepGeneratedQueryFailed",
  kg_query: "traceStepKgQuery",
  kg_schema: "traceStepKgSchema",
  path_search: "traceStepPathSearch",
  "route.local_agent": "traceStepLocalAgentRoute",
  text_search: "traceStepTextSearch",
};

const SAFE_STATUS_VALUES = new Set(["ok", "fallback", "empty"]);
const SAFE_KIND_VALUES = new Set(["answer", "ranking", "retrieval"]);
const SAFE_ERROR_LABELS: Record<string, string> = {
  cyphersyntaxerror: "traceErrorCypherSyntax",
  runtimeerror: "traceErrorRuntime",
  valueerror: "traceErrorValidation",
};

function translateOrFallback(locale: Locale, key: string | undefined, fallbackKey: string) {
  return key ? translate(locale, key as never) : translate(locale, fallbackKey as never);
}

function toChatTrace(trace: BackendChatResponse["trace"], locale: Locale): ChatTrace | undefined {
  if (!trace || !Array.isArray(trace.steps) || trace.steps.length === 0) {
    return undefined;
  }

  return {
    mode: translateOrFallback(locale, trace.mode ? SAFE_MODE_LABELS[trace.mode] : undefined, "traceModeUnknownValue"),
    route: translateOrFallback(locale, trace.route ? SAFE_ROUTE_LABELS[trace.route] : undefined, "traceRouteUnknownValue"),
    tier: translateOrFallback(locale, trace.tier ? SAFE_TIER_LABELS[trace.tier] : undefined, "traceTierUnknownValue"),
    steps: trace.steps.map((step, index) => ({
      errorType: step.error_type
        ? translateOrFallback(
            locale,
            SAFE_ERROR_LABELS[step.error_type.toLowerCase()],
            "traceErrorGeneric"
          )
        : undefined,
      kind: SAFE_KIND_VALUES.has(step.kind || "") ? (step.kind as string) : "retrieval",
      label: translateOrFallback(
        locale,
        SAFE_STEP_LABELS[step.name || step.label || ""],
        index === 0 ? "traceStepStart" : "traceStepGeneric"
      ),
      rowCount: typeof step.row_count === "number" ? step.row_count : undefined,
      status: SAFE_STATUS_VALUES.has(step.status || "") ? (step.status as string) : "ok",
      topK: typeof step.top_k === "number" ? step.top_k : undefined,
    })),
  };
}

function dedupeByUrl(citations: BackendCitation[]) {
  const seen = new Set<string>();
  const deduped: BackendCitation[] = [];
  for (const c of citations) {
    if (!c.page_url || seen.has(c.page_url)) {
      continue;
    }
    seen.add(c.page_url);
    deduped.push(c);
  }
  return deduped;
}

function toEvidenceSources(citations: BackendCitation[], locale: Locale): EvidenceSource[] {
  const kbName = locale === "vi" ? "Wikipedia tiếng Việt" : "Vietnamese Wikipedia";
  return dedupeByUrl(citations).map((c, idx) => ({
    id: c.chunk_id || `source-${idx}`,
    knowledgeBaseId: "viwiki",
    knowledgeBaseName: kbName,
    external: true,
    title: c.page_title || (locale === "vi" ? "Bài viết Wikipedia" : "Wikipedia article"),
    location: c.page_url,
    excerpt:
      locale === "vi"
        ? "Trích dẫn từ Wikipedia (mở liên kết để xem bài gốc)."
        : "Evidence from Wikipedia (open link to view the original article).",
    score: 0.95,
    path:
      locale === "vi"
        ? [kbName, "dẫn chứng", c.page_title || "bài viết"]
        : [kbName, "evidence", c.page_title || "article"],
  }));
}

function eventTitle(locale: Locale, event: string, data: Record<string, unknown>) {
  if (event === "route") {
    return translate(locale, "streamRoute", {
      value: String(data.route || data.name || "unknown"),
    });
  }
  if (event === "tool_call") {
    return translate(locale, "streamToolCall", {
      value: String(data.tool || "unknown"),
    });
  }
  if (event === "tool_result") {
    return translate(locale, "streamToolResult", {
      value: String(data.tool || "unknown"),
    });
  }
  if (event === "cypher") {
    return translate(locale, "streamCypher");
  }
  if (event === "fallback") {
    return translate(locale, "streamFallback", {
      value: String(data.name || "unknown"),
    });
  }
  if (event === "error") {
    return translate(locale, "streamError");
  }
  if (event === "final") {
    return translate(locale, "streamFinal");
  }
  if (event === "answer_delta") {
    return translate(locale, "streamAnswerDelta");
  }
  if (event === "citation") {
    return translate(locale, "streamCitation");
  }
  return translate(locale, "streamStatus");
}

function eventDetail(locale: Locale, event: string, data: Record<string, unknown>) {
  if (event === "status") {
    return String(data.state || "");
  }
  if (event === "tool_call") {
    if (typeof data.input === "object" && data.input) {
      return JSON.stringify(data.input);
    }
  }
  if (event === "tool_result") {
    const status = data.status ? String(data.status) : "";
    const rows =
      typeof data.row_count === "number"
        ? ` • ${translate(locale, "traceRows", { count: data.row_count })}`
        : "";
    return `${status}${rows}`.trim();
  }
  if (event === "fallback") {
    return data.message ? String(data.message) : undefined;
  }
  if (event === "error") {
    return String(data.message || "Unknown error");
  }
  if (event === "citation") {
    return String(data.page_title || data.chunk_id || "");
  }
  return undefined;
}

function toBackendEvent(
  locale: Locale,
  block: BackendStreamBlock,
  index: number
): ChatBackendEvent {
  return {
    detail: eventDetail(locale, block.event, block.data),
    id: `${block.event}-${index + 1}`,
    query:
      block.event === "cypher"
        ? String(block.data.query || "")
        : undefined,
    title: eventTitle(locale, block.event, block.data),
    type: ([
      "answer_delta",
      "citation",
      "cypher",
      "error",
      "fallback",
      "final",
      "route",
      "status",
      "tool_call",
      "tool_result",
    ] as const).includes(block.event as ChatBackendEvent["type"])
      ? (block.event as ChatBackendEvent["type"])
      : "status",
  };
}

export function parseSseEventBlock(block: string): BackendStreamBlock | undefined {
  const trimmed = block.trim();
  if (!trimmed) {
    return undefined;
  }

  let event = "message";
  let data = "";
  for (const line of trimmed.split("\n")) {
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      data += line.slice(5).trim();
    }
  }

  return {
    data: data ? (JSON.parse(data) as Record<string, unknown>) : {},
    event,
  };
}

export async function runBackendChat({
  message,
  locale,
  model,
  debug,
}: {
  locale: Locale;
  message: string;
  model: ModelId;
  debug?: boolean;
}): Promise<{
  answer: string;
  sources: EvidenceSource[];
  trace?: ChatTrace;
  usage: { elapsedMs: number; model: ModelId; sources: number; retrievalTier?: string };
}> {
  const url = `${env.API_URL.replace(/\/$/, "")}/chat`;
  const t0 = performance.now();

  const response = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      messages: [{ role: "user", content: message }],
      top_k: 4,
      debug: Boolean(debug),
    }),
  });

  if (!response.ok) {
    throw new Error(`Chat request failed: ${response.status}`);
  }

  const data = (await response.json()) as BackendChatResponse;
  const elapsedMs = Math.max(0, Math.round(performance.now() - t0));
  const sources = toEvidenceSources(data.citations ?? [], locale);
  const trace = toChatTrace(data.trace, locale);

  // Ensure the UI always has something readable.
  const answer = (data.answer || "").trim() || translate(locale, "simulationFailed");
  const retrievalTier = data.debug?.retrieval_tier ?? data.retrieval_tier;

  return {
    answer,
    sources,
    trace,
    usage: { elapsedMs, model, sources: sources.length, retrievalTier },
  };
}

export async function streamBackendChat({
  locale,
  message,
  model,
  onEvent,
}: {
  locale: Locale;
  message: string;
  model: ModelId;
  onEvent: (event: ChatBackendEvent, raw: BackendStreamBlock) => void;
}): Promise<{
  answer: string;
  sources: EvidenceSource[];
  trace?: ChatTrace;
  usage: { elapsedMs: number; model: ModelId; sources: number; retrievalTier?: string };
}> {
  const url = `${env.API_URL.replace(/\/$/, "")}/chat/stream`;
  const t0 = performance.now();
  const response = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      messages: [{ role: "user", content: message }],
      top_k: 4,
      debug: true,
    }),
  });

  if (!response.ok || !response.body) {
    throw new Error(`Chat stream failed: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let index = 0;
  let finalPayload: BackendChatResponse | undefined;

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() || "";

    for (const block of blocks) {
      const parsed = parseSseEventBlock(block);
      if (!parsed) {
        continue;
      }
      onEvent(toBackendEvent(locale, parsed, index), parsed);
      index += 1;
      if (parsed.event === "final") {
        finalPayload = parsed.data as unknown as BackendChatResponse;
      }
      if (parsed.event === "error") {
        throw new Error(String(parsed.data.message || "Streaming request failed"));
      }
    }

    if (done) {
      break;
    }
  }

  if (!finalPayload) {
    throw new Error("Streaming response ended without a final event");
  }

  const elapsedMs = Math.max(0, Math.round(performance.now() - t0));
  const sources = toEvidenceSources(finalPayload.citations ?? [], locale);
  const trace = toChatTrace(finalPayload.trace, locale);
  const answer =
    (finalPayload.answer || "").trim() || translate(locale, "simulationFailed");
  const retrievalTier =
    finalPayload.debug?.retrieval_tier ?? finalPayload.retrieval_tier;

  return {
    answer,
    sources,
    trace,
    usage: { elapsedMs, model, sources: sources.length, retrievalTier },
  };
}
