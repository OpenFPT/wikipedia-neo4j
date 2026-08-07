export type KnowledgeBaseKind = "hosted" | "local";
export type KnowledgeBaseStatus = "indexing" | "offline" | "ready";
export type KnowledgeScope = "all" | "none" | `kb:${string}`;

export interface KnowledgeBase {
  chunks: number;
  documentCount: number;
  id: string;
  kind: KnowledgeBaseKind;
  name: string;
  nameVi?: string;
  progress?: number;
  status: KnowledgeBaseStatus;
}

export const initialKnowledgeBases: KnowledgeBase[] = [
  {
    id: "thesis",
    name: "Thesis research",
    nameVi: "Nghiên cứu luận văn",
    kind: "local",
    status: "ready",
    documentCount: 12,
    chunks: 421,
  },
  {
    id: "viwiki",
    name: "Vietnamese Wikipedia",
    nameVi: "Wikipedia tiếng Việt",
    kind: "hosted",
    status: "ready",
    documentCount: 1,
    chunks: 0,
  },
  {
    id: "product-notes",
    name: "Product notes",
    nameVi: "Ghi chú sản phẩm",
    kind: "local",
    status: "indexing",
    documentCount: 4,
    chunks: 86,
    progress: 68,
  },
];

export function scopeForKnowledgeBase(id: string): KnowledgeScope {
  return `kb:${id}`;
}

export function knowledgeBaseIdFromScope(scope: KnowledgeScope) {
  return scope.startsWith("kb:") ? scope.slice(3) : undefined;
}
