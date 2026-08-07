import { type Locale, translate } from "@/features/i18n/i18n";
import {
  type KnowledgeBase,
  type KnowledgeScope,
  knowledgeBaseIdFromScope,
} from "@/features/knowledge/types";
import type {
  ChatModel,
  ChatRequest,
  ChatUsage,
  EvidenceSource,
  MockChatEvent,
  RetrievalStage,
} from "../types";

const delay = (milliseconds: number) =>
  new Promise<void>((resolve) => globalThis.setTimeout(resolve, milliseconds));

export const mockModels: ChatModel[] = [
  {
    id: "recommended",
    name: "ViWiki RAG 7B",
    meta: "Q4_K_M",
    badge: "recommended",
    kind: "local",
    status: "ready",
  },
  {
    id: "local-qwen",
    name: "Qwen2.5 7B Instruct",
    meta: "GGUF",
    badge: "imported",
    kind: "local",
    status: "ready",
  },
  {
    id: "openai",
    name: "GPT-4.1 mini",
    meta: "OpenAI",
    badge: "connected",
    kind: "provider",
    status: "ready",
  },
];

interface Localized {
  en: string;
  vi: string;
}

interface SourceSeed {
  excerpt: Localized;
  external: boolean;
  id: string;
  knowledgeBaseId: string;
  knowledgeBaseName: Localized;
  location: Localized;
  path: { en: string[]; vi: string[] };
  score: number;
  title: Localized;
}

const sourceSeeds: SourceSeed[] = [
  {
    id: "thesis:architecture:42",
    knowledgeBaseId: "thesis",
    knowledgeBaseName: {
      en: "Thesis research",
      vi: "Nghiên cứu luận văn",
    },
    external: false,
    title: {
      en: "GraphRAG architecture notes",
      vi: "Ghi chú kiến trúc GraphRAG",
    },
    location: { en: "Page 18", vi: "Trang 18" },
    excerpt: {
      en: "Documents are chunked, embedded, linked to extracted entities, and retrieved with lexical, vector, and bounded graph search.",
      vi: "Tài liệu được chia đoạn, nhúng, liên kết với thực thể và truy xuất bằng tìm kiếm từ khóa, véc-tơ cùng đồ thị giới hạn.",
    },
    score: 0.94,
    path: {
      en: ["Research notes", "mentions", "GraphRAG", "uses", "Neo4j"],
      vi: ["Ghi chú nghiên cứu", "đề cập", "GraphRAG", "sử dụng", "Neo4j"],
    },
  },
  {
    id: "thesis:evaluation:11",
    knowledgeBaseId: "thesis",
    knowledgeBaseName: {
      en: "Thesis research",
      vi: "Nghiên cứu luận văn",
    },
    external: false,
    title: { en: "Retrieval evaluation", vi: "Đánh giá truy xuất" },
    location: { en: "Section 3.2", vi: "Mục 3.2" },
    excerpt: {
      en: "Weighted reciprocal-rank fusion combines independently ranked evidence without assuming backend scores are directly comparable.",
      vi: "Hợp nhất thứ hạng nghịch đảo có trọng số kết hợp bằng chứng đã xếp hạng độc lập mà không coi điểm từ các nguồn là tương đương.",
    },
    score: 0.87,
    path: {
      en: ["Evaluation", "measures", "WRRF", "ranks", "Evidence"],
      vi: ["Đánh giá", "đo lường", "WRRF", "xếp hạng", "Bằng chứng"],
    },
  },
  {
    id: "viwiki:knowledge-graph:7",
    knowledgeBaseId: "viwiki",
    knowledgeBaseName: {
      en: "Vietnamese Wikipedia",
      vi: "Wikipedia tiếng Việt",
    },
    external: true,
    title: { en: "Knowledge graph", vi: "Đồ thị tri thức" },
    location: { en: "Vietnamese Wikipedia", vi: "Wikipedia tiếng Việt" },
    excerpt: {
      en: "A knowledge graph represents entities and relationships in a form that supports connected queries and traversal.",
      vi: "Đồ thị tri thức biểu diễn thực thể và quan hệ theo cách hỗ trợ truy vấn liên kết và duyệt đồ thị.",
    },
    score: 0.91,
    path: {
      en: [
        "Knowledge graph",
        "relates to",
        "Graph database",
        "supports",
        "Traversal",
      ],
      vi: [
        "Đồ thị tri thức",
        "liên quan",
        "Cơ sở dữ liệu đồ thị",
        "hỗ trợ",
        "Duyệt đồ thị",
      ],
    },
  },
  {
    id: "viwiki:rag:3",
    knowledgeBaseId: "viwiki",
    knowledgeBaseName: {
      en: "Vietnamese Wikipedia",
      vi: "Wikipedia tiếng Việt",
    },
    external: true,
    title: {
      en: "Retrieval-augmented generation",
      vi: "Sinh tăng cường bằng truy xuất",
    },
    location: { en: "Vietnamese Wikipedia", vi: "Wikipedia tiếng Việt" },
    excerpt: {
      en: "Retrieved evidence is supplied to a language model to improve factual grounding and preserve source context.",
      vi: "Bằng chứng truy xuất được cung cấp cho mô hình ngôn ngữ để tăng tính có căn cứ và giữ ngữ cảnh nguồn.",
    },
    score: 0.84,
    path: {
      en: ["RAG", "uses", "Retriever", "returns", "Evidence"],
      vi: ["RAG", "sử dụng", "Bộ truy xuất", "trả về", "Bằng chứng"],
    },
  },
  {
    id: "product:brief:4",
    knowledgeBaseId: "product-notes",
    knowledgeBaseName: { en: "Product notes", vi: "Ghi chú sản phẩm" },
    external: false,
    title: { en: "Product brief", vi: "Bản mô tả sản phẩm" },
    location: { en: "Draft 4", vi: "Bản nháp 4" },
    excerpt: {
      en: "Knowledge bases are named contexts that can be selected individually, together, or not used for a conversation.",
      vi: "Kho tri thức là các ngữ cảnh có tên, có thể được chọn riêng, chọn tất cả hoặc không dùng trong cuộc trò chuyện.",
    },
    score: 0.82,
    path: {
      en: ["Product", "defines", "Knowledge base", "grounds", "Conversation"],
      vi: [
        "Sản phẩm",
        "định nghĩa",
        "Kho tri thức",
        "làm căn cứ",
        "Cuộc trò chuyện",
      ],
    },
  },
  {
    id: "product:privacy:2",
    knowledgeBaseId: "product-notes",
    knowledgeBaseName: { en: "Product notes", vi: "Ghi chú sản phẩm" },
    external: false,
    title: { en: "Privacy principles", vi: "Nguyên tắc riêng tư" },
    location: { en: "Section 2", vi: "Mục 2" },
    excerpt: {
      en: "Private documents remain on the device unless the user explicitly chooses a hosted knowledge base.",
      vi: "Tài liệu riêng tư ở lại trên thiết bị trừ khi người dùng chủ động chọn kho tri thức trực tuyến.",
    },
    score: 0.79,
    path: {
      en: ["Private document", "stays on", "Device", "unless", "Hosted source"],
      vi: [
        "Tài liệu riêng",
        "ở trên",
        "Thiết bị",
        "trừ khi",
        "Nguồn trực tuyến",
      ],
    },
  },
];

function localizeSource(seed: SourceSeed, locale: Locale): EvidenceSource {
  return {
    id: seed.id,
    knowledgeBaseId: seed.knowledgeBaseId,
    knowledgeBaseName: seed.knowledgeBaseName[locale],
    external: seed.external,
    title: seed.title[locale],
    location: seed.location[locale],
    excerpt: seed.excerpt[locale],
    score: seed.score,
    path: seed.path[locale],
  };
}

export function getSourcesForScope(
  scope: KnowledgeScope,
  locale: Locale,
  knowledgeName?: string,
  knowledgeBases: KnowledgeBase[] = []
) {
  if (scope === "none") {
    return [];
  }
  const localized = sourceSeeds.map((seed) => localizeSource(seed, locale));
  if (scope === "all") {
    const seededIds = new Set(
      localized.map((source) => source.knowledgeBaseId)
    );
    const dynamicSources = knowledgeBases
      .filter(
        (knowledgeBase) =>
          knowledgeBase.status === "ready" && !seededIds.has(knowledgeBase.id)
      )
      .map((knowledgeBase) => makeKnowledgeBaseSource(knowledgeBase, locale));
    return [...localized, ...dynamicSources];
  }
  const knowledgeBaseId = knowledgeBaseIdFromScope(scope);
  const matching = localized.filter(
    (source) => source.knowledgeBaseId === knowledgeBaseId
  );
  if (matching.length > 0) {
    return matching;
  }
  const name = knowledgeName ?? knowledgeBaseId ?? "Knowledge base";
  return [
    {
      id: `${knowledgeBaseId ?? "custom"}:mock:1`,
      knowledgeBaseId: knowledgeBaseId ?? "custom",
      knowledgeBaseName: name,
      external: false,
      title: locale === "vi" ? "Tài liệu mô phỏng" : "Simulated document",
      location: locale === "vi" ? "Đoạn 1" : "Passage 1",
      excerpt:
        locale === "vi"
          ? "Đoạn mô phỏng này minh họa bằng chứng sẽ được trả về từ kho tri thức mới."
          : "This simulated passage illustrates evidence returned from a newly created knowledge base.",
      score: 0.88,
      path:
        locale === "vi"
          ? [name, "chứa", "Tài liệu", "hỗ trợ", "Câu trả lời"]
          : [name, "contains", "Document", "supports", "Answer"],
    },
  ];
}

function makeKnowledgeBaseSource(
  knowledgeBase: KnowledgeBase,
  locale: Locale
): EvidenceSource {
  const name =
    locale === "vi" && knowledgeBase.nameVi
      ? knowledgeBase.nameVi
      : knowledgeBase.name;
  return {
    id: `${knowledgeBase.id}:overview`,
    knowledgeBaseId: knowledgeBase.id,
    knowledgeBaseName: name,
    external: knowledgeBase.kind === "hosted",
    title: locale === "vi" ? `Tổng quan ${name}` : `${name} overview`,
    location:
      locale === "vi"
        ? `${knowledgeBase.documentCount} tài liệu`
        : `${knowledgeBase.documentCount} documents`,
    excerpt:
      locale === "vi"
        ? `Đoạn mô phỏng này đại diện cho nội dung đã lập chỉ mục trong ${name}.`
        : `This simulated passage represents indexed content from ${name}.`,
    score: 0.82,
    path:
      locale === "vi"
        ? [name, "chứa", "Nguồn đã lập chỉ mục", "hỗ trợ", "Câu trả lời"]
        : [name, "contains", "Indexed source", "supports", "Answer"],
  };
}

export function getInitialSources(locale: Locale = "en") {
  return getSourcesForScope("all", locale).slice(0, 2);
}

export function getMockStages(
  scope: KnowledgeScope,
  locale: Locale = "en"
): RetrievalStage[] {
  if (scope === "none") {
    return [
      { id: "prepare", label: translate(locale, "prepareStage") },
      { id: "generate", label: translate(locale, "generateStage") },
    ];
  }

  const stages: RetrievalStage[] = [
    { id: "understand", label: translate(locale, "understandStage") },
    { id: "search", label: translate(locale, "searchStage") },
    { id: "graph", label: translate(locale, "graphStage") },
  ];
  if (scope === "all") {
    stages.push({ id: "fusion", label: translate(locale, "fusionStage") });
  }
  stages.push({ id: "generate", label: translate(locale, "generateStage") });
  return stages;
}

function makeAnswer(request: ChatRequest) {
  const subject = request.message.replace(/\s+/g, " ").trim();
  if (request.scope === "none") {
    return `${translate(request.locale, "answerUngrounded", { subject })}\n\n${translate(request.locale, "answerUngroundedFollowup")}`;
  }
  let knowledge = request.knowledgeName ?? "the selected knowledge base";
  if (request.scope === "all") {
    knowledge =
      request.locale === "vi"
        ? "tất cả kho tri thức khả dụng"
        : "all available knowledge bases";
  }
  return `${translate(request.locale, "answerGrounded", { knowledge, subject })}\n\n${translate(request.locale, "answerGroundedFollowup")}`;
}

export interface StreamMockChatOptions {
  pause?: (milliseconds: number) => Promise<void>;
}

export async function* streamMockChat(
  request: ChatRequest,
  options: StreamMockChatOptions = {}
): AsyncGenerator<MockChatEvent> {
  if (!request.message.trim()) {
    throw new TypeError("message is required");
  }

  const pause = options.pause ?? delay;
  const stages = getMockStages(request.scope, request.locale);

  for (const stage of stages) {
    yield { type: "stage", state: "running", stage };
    await pause(stage.id === "generate" ? 260 : 180);
    yield { type: "stage", state: "complete", stage };
  }

  const sources = getSourcesForScope(
    request.scope,
    request.locale,
    request.knowledgeName,
    request.knowledgeBases
  );
  yield { type: "sources", sources };

  for (const token of makeAnswer(request).match(/\S+\s*/g) ?? []) {
    await pause(18);
    yield { type: "token", value: token };
  }

  const usage: ChatUsage = {
    model: request.model,
    sources: sources.length,
    elapsedMs: 1240,
  };
  yield { type: "done", usage };
}
