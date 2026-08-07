import env from "@/config/env";
import { type Locale, translate } from "@/features/i18n/i18n";
import type { EvidenceSource, ModelId } from "../types";

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
};

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

  // Ensure the UI always has something readable.
  const answer = (data.answer || "").trim() || translate(locale, "simulationFailed");
  const retrievalTier = data.debug?.retrieval_tier ?? data.retrieval_tier;

  return {
    answer,
    sources,
    usage: { elapsedMs, model, sources: sources.length, retrievalTier },
  };
}
