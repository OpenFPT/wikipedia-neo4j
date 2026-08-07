import {
  ArrowRight,
  FileText,
  Globe2,
  HardDrive,
  Plus,
  RefreshCw,
  Trash2,
  Upload,
} from "lucide-react";
import { useState } from "react";
import { GraphCanvas } from "@/features/chat/components/graph-canvas";
import { useI18n } from "@/features/i18n/i18n";
import { cn } from "@/lib/utils";
import { type MockDocument, simulateDocumentImport } from "../api/mock-import";
import type { KnowledgeBase, KnowledgeBaseKind } from "../types";

interface KnowledgePageProps {
  knowledgeBases: KnowledgeBase[];
  onChange: (knowledgeBases: KnowledgeBase[]) => void;
  onToast: (message: string) => void;
  onUseKnowledgeBase: (id: string) => void;
}

const initialDocuments: Record<string, MockDocument[]> = {
  thesis: [
    {
      id: "thesis-file",
      name: "thesis-architecture.pdf",
      chunks: 128,
      status: "ready",
    },
    {
      id: "retrieval-file",
      name: "retrieval-notes.md",
      chunks: 42,
      status: "ready",
    },
  ],
  "product-notes": [
    {
      id: "product-file",
      name: "product-brief.md",
      chunks: 31,
      status: "ready",
    },
  ],
};

export function KnowledgePage({
  knowledgeBases,
  onChange,
  onToast,
  onUseKnowledgeBase,
}: KnowledgePageProps) {
  const { locale, t } = useI18n();
  const [selectedId, setSelectedId] = useState(knowledgeBases[0]?.id);
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState("");
  const [kind, setKind] = useState<KnowledgeBaseKind>("local");
  const [endpoint, setEndpoint] = useState("");
  const [documents, setDocuments] = useState(initialDocuments);
  const [indexingFiles, setIndexingFiles] = useState<string[]>([]);
  const selected =
    knowledgeBases.find((knowledge) => knowledge.id === selectedId) ??
    knowledgeBases[0];

  function createKnowledgeBase() {
    const trimmedName = name.trim();
    if (!trimmedName) {
      return;
    }
    const baseId = `${trimmedName.toLowerCase().replace(/[^a-z0-9]+/g, "-")}-${Date.now()}`;
    const next: KnowledgeBase = {
      id: baseId,
      name: trimmedName,
      kind,
      status: kind === "hosted" ? "ready" : "indexing",
      documentCount: kind === "hosted" ? 1 : 0,
      chunks: 0,
      progress: kind === "local" ? 18 : undefined,
    };
    onChange([next, ...knowledgeBases]);
    setSelectedId(baseId);
    setCreating(false);
    setName("");
    setEndpoint("");
    onToast(kind === "hosted" ? t("connected") : t("createKnowledgeBase"));
  }

  function updateSelected(update: (knowledge: KnowledgeBase) => KnowledgeBase) {
    if (!selected) {
      return;
    }
    onChange(
      knowledgeBases.map((knowledge) =>
        knowledge.id === selected.id ? update(knowledge) : knowledge
      )
    );
  }

  async function handleFiles(files: FileList | null) {
    if (!(files && selected)) {
      return;
    }
    for (const file of Array.from(files)) {
      setIndexingFiles((current) => [...current, file.name]);
      try {
        const document = await simulateDocumentImport(file);
        setDocuments((current) => ({
          ...current,
          [selected.id]: [
            document,
            ...(current[selected.id] ?? []).filter(
              (item) => item.name !== document.name
            ),
          ],
        }));
        updateSelected((knowledge) => ({
          ...knowledge,
          chunks: knowledge.chunks + document.chunks,
          documentCount: knowledge.documentCount + 1,
          progress: undefined,
          status: "ready",
        }));
        onToast(t("importedFile", { name: file.name }));
      } catch {
        onToast(t("unsupportedFile"));
      } finally {
        setIndexingFiles((current) =>
          current.filter((fileName) => fileName !== file.name)
        );
      }
    }
  }

  function reindex() {
    updateSelected((knowledge) => ({
      ...knowledge,
      progress: 36,
      status: "indexing",
    }));
    globalThis.setTimeout(() => {
      if (!selected) {
        return;
      }
      onChange(
        knowledgeBases.map((knowledge) =>
          knowledge.id === selected.id
            ? { ...knowledge, progress: undefined, status: "ready" }
            : knowledge
        )
      );
    }, 900);
  }

  function removeSelected() {
    if (!selected) {
      return;
    }
    const next = knowledgeBases.filter(
      (knowledge) => knowledge.id !== selected.id
    );
    onChange(next);
    setSelectedId(next[0]?.id);
  }

  return (
    <div className="min-h-0 flex-1 overflow-y-auto px-6 pb-8 max-[720px]:px-4">
      <div className="mx-auto max-w-5xl">
        <header className="flex items-start justify-between gap-4 py-6">
          <p className="max-w-xl text-muted-foreground text-sm leading-relaxed">
            {t("knowledgeDescription")}
          </p>
          <button
            className="inline-flex h-9 shrink-0 items-center gap-2 rounded-lg bg-primary px-3.5 font-medium text-primary-foreground text-sm hover:bg-primary/90"
            onClick={() => setCreating((current) => !current)}
            type="button"
          >
            <Plus className="size-3.5" />
            {t("newKnowledgeBase")}
          </button>
        </header>

        {creating && (
          <section className="mb-5 border-y py-4">
            <div className="grid gap-3 min-[760px]:grid-cols-[minmax(0,1fr)_180px]">
              <label className="grid gap-1.5 font-medium text-xs">
                {t("knowledgeBaseName")}
                <input
                  autoFocus
                  className="h-10 rounded-lg border bg-background px-3 text-foreground text-sm outline-none focus:border-ring focus:ring-2 focus:ring-ring/20"
                  onChange={(event) => setName(event.currentTarget.value)}
                  placeholder={t("knowledgeBaseNamePlaceholder")}
                  value={name}
                />
              </label>
              <label className="grid gap-1.5 font-medium text-xs">
                {t("knowledgeBases")}
                <select
                  className="h-10 rounded-lg border bg-background px-3 text-foreground text-sm outline-none focus:border-ring"
                  onChange={(event) =>
                    setKind(event.currentTarget.value as KnowledgeBaseKind)
                  }
                  value={kind}
                >
                  <option value="local">{t("localKnowledgeBase")}</option>
                  <option value="hosted">{t("hostedKnowledgeBase")}</option>
                </select>
              </label>
            </div>
            {kind === "hosted" && (
              <label className="mt-3 grid max-w-xl gap-1.5 font-medium text-xs">
                {t("endpoint")}
                <input
                  className="h-10 rounded-lg border bg-background px-3 text-foreground text-sm outline-none focus:border-ring focus:ring-2 focus:ring-ring/20"
                  onChange={(event) => setEndpoint(event.currentTarget.value)}
                  placeholder={t("endpointPlaceholder")}
                  type="url"
                  value={endpoint}
                />
              </label>
            )}
            <div className="mt-4 flex justify-end gap-2">
              <button
                className="h-9 rounded-lg px-3 text-sm hover:bg-muted"
                onClick={() => setCreating(false)}
                type="button"
              >
                {t("cancel")}
              </button>
              <button
                className="h-9 rounded-lg bg-primary px-3.5 font-medium text-primary-foreground text-sm hover:bg-primary/90 disabled:opacity-50"
                disabled={
                  !name.trim() || (kind === "hosted" && !endpoint.trim())
                }
                onClick={createKnowledgeBase}
                type="button"
              >
                {kind === "hosted" ? t("connect") : t("create")}
              </button>
            </div>
          </section>
        )}

        {knowledgeBases.length === 0 ? (
          <div className="grid min-h-80 place-items-center border-y text-center">
            <div>
              <BookOpenEmpty />
              <h2 className="mt-3 font-medium text-sm">
                {t("noKnowledgeBases")}
              </h2>
              <p className="mt-1 text-muted-foreground text-xs">
                {t("noKnowledgeDescription")}
              </p>
            </div>
          </div>
        ) : (
          <div className="grid min-h-[560px] gap-6 min-[880px]:grid-cols-[280px_minmax(0,1fr)]">
            <div className="border-r pr-4 max-[879px]:border-r-0 max-[879px]:border-b max-[879px]:pr-0 max-[879px]:pb-4">
              <div className="grid gap-1">
                {knowledgeBases.map((knowledge) => {
                  const active = knowledge.id === selected?.id;
                  const displayName =
                    locale === "vi" && knowledge.nameVi
                      ? knowledge.nameVi
                      : knowledge.name;
                  return (
                    <button
                      className={cn(
                        "flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2.5 text-left hover:bg-muted",
                        active && "bg-muted"
                      )}
                      key={knowledge.id}
                      onClick={() => setSelectedId(knowledge.id)}
                      type="button"
                    >
                      <span className="grid size-8 shrink-0 place-items-center rounded-lg bg-background text-muted-foreground">
                        {knowledge.kind === "hosted" ? (
                          <Globe2 className="size-4" />
                        ) : (
                          <HardDrive className="size-4" />
                        )}
                      </span>
                      <span className="min-w-0 flex-1">
                        <strong className="block truncate font-medium text-sm">
                          {displayName}
                        </strong>
                        <span className="mt-0.5 block text-muted-foreground text-xs">
                          {statusLabel(knowledge, t)}
                        </span>
                      </span>
                      <span
                        className={cn(
                          "size-1.5 rounded-full bg-success",
                          knowledge.status === "indexing" && "bg-warning",
                          knowledge.status === "offline" &&
                            "bg-muted-foreground"
                        )}
                      />
                    </button>
                  );
                })}
              </div>
            </div>

            {selected && (
              <section className="min-w-0">
                <div className="flex flex-wrap items-start gap-3">
                  <div className="min-w-0 flex-1">
                    <h2 className="truncate font-medium text-base">
                      {locale === "vi" && selected.nameVi
                        ? selected.nameVi
                        : selected.name}
                    </h2>
                    <p className="mt-1 max-w-xl text-muted-foreground text-xs leading-relaxed">
                      {descriptionFor(selected.id, t)}
                    </p>
                  </div>
                  <button
                    className="inline-flex h-9 items-center gap-2 rounded-lg bg-primary px-3 font-medium text-primary-foreground text-sm hover:bg-primary/90"
                    onClick={() => onUseKnowledgeBase(selected.id)}
                    type="button"
                  >
                    {t("startChat")}
                    <ArrowRight className="size-3.5" />
                  </button>
                </div>

                {selected.status === "indexing" && (
                  <div className="mt-4 rounded-lg bg-warning/10 p-3 text-warning">
                    <div className="flex justify-between text-xs">
                      <span>{t("indexing")}</span>
                      <span>{selected.progress ?? 0}%</span>
                    </div>
                    <div
                      aria-label={t("indexing")}
                      aria-valuemax={100}
                      aria-valuemin={0}
                      aria-valuenow={selected.progress ?? 0}
                      className="mt-2 h-1 overflow-hidden rounded-full bg-warning/20"
                      role="progressbar"
                    >
                      <div
                        className="h-full rounded-full bg-primary"
                        style={{ width: `${selected.progress ?? 0}%` }}
                      />
                    </div>
                  </div>
                )}

                <div className="mt-5 grid grid-cols-2 gap-px overflow-hidden rounded-xl border bg-border">
                  <Metric
                    label={t("documents")}
                    value={selected.documentCount}
                  />
                  <Metric label={t("chunks")} value={selected.chunks} />
                </div>

                {selected.kind === "local" && (
                  <div className="mt-5">
                    <div className="mb-2 flex items-center justify-between">
                      <h3 className="font-medium text-sm">{t("documents")}</h3>
                      <label className="inline-flex cursor-pointer items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs hover:bg-muted">
                        <Upload className="size-3.5" />
                        {t("addDocuments")}
                        <input
                          accept=".pdf,.md,.txt"
                          className="sr-only"
                          multiple
                          onChange={(event) =>
                            handleFiles(event.currentTarget.files)
                          }
                          type="file"
                        />
                      </label>
                    </div>
                    <div className="divide-y rounded-xl border">
                      {indexingFiles.map((fileName) => (
                        <DocumentRow
                          key={`indexing-${fileName}`}
                          meta={t("indexingFile")}
                          name={fileName}
                          pending
                        />
                      ))}
                      {(documents[selected.id] ?? []).map((document) => (
                        <DocumentRow
                          key={document.id}
                          meta={`${document.chunks} ${t("chunks")} · ${t("updatedToday")}`}
                          name={document.name}
                        />
                      ))}
                    </div>
                  </div>
                )}

                <div className="mt-5">
                  <div className="mb-2 flex items-center justify-between">
                    <h3 className="font-medium text-sm">{t("graphPreview")}</h3>
                    <span className="rounded-full bg-secondary px-2 py-1 text-[10px] text-muted-foreground leading-none">
                      {t("demo")}
                    </span>
                  </div>
                  <GraphCanvas />
                </div>

                <div className="mt-5 flex gap-2 border-t pt-4">
                  <button
                    className="inline-flex h-9 items-center gap-2 rounded-lg px-3 text-sm hover:bg-muted"
                    onClick={reindex}
                    type="button"
                  >
                    <RefreshCw className="size-3.5" />
                    {t("reindex")}
                  </button>
                  <button
                    className="ml-auto inline-flex h-9 items-center gap-2 rounded-lg px-3 text-destructive text-sm hover:bg-destructive/10"
                    onClick={removeSelected}
                    type="button"
                  >
                    <Trash2 className="size-3.5" />
                    {t("remove")}
                  </button>
                </div>
              </section>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: number }) {
  return (
    <div className="bg-background px-4 py-3">
      <span className="block text-muted-foreground text-xs">{label}</span>
      <strong className="mt-1 block font-medium text-sm tabular-nums">
        {value.toLocaleString()}
      </strong>
    </div>
  );
}

function DocumentRow({
  name,
  meta,
  pending = false,
}: {
  meta: string;
  name: string;
  pending?: boolean;
}) {
  return (
    <div className="flex items-center gap-2.5 px-3 py-2.5">
      <span className="grid size-7 place-items-center rounded-lg bg-muted text-muted-foreground">
        {pending ? (
          <RefreshCw className="size-3.5 animate-spin" />
        ) : (
          <FileText className="size-3.5" />
        )}
      </span>
      <span className="min-w-0 flex-1">
        <strong className="block truncate font-medium text-sm">{name}</strong>
        <span className="mt-0.5 block truncate text-muted-foreground text-xs">
          {meta}
        </span>
      </span>
    </div>
  );
}

function BookOpenEmpty() {
  return (
    <span className="mx-auto grid size-10 place-items-center rounded-full bg-muted text-muted-foreground">
      <FileText className="size-4" />
    </span>
  );
}

function statusLabel(
  knowledge: KnowledgeBase,
  t: ReturnType<typeof useI18n>["t"]
) {
  if (knowledge.status === "indexing") {
    return `${t("indexing")} · ${knowledge.progress ?? 0}%`;
  }
  if (knowledge.status === "offline") {
    return t("offline");
  }
  return knowledge.kind === "hosted" ? t("connected") : t("ready");
}

function descriptionFor(id: string, t: ReturnType<typeof useI18n>["t"]) {
  if (id === "thesis") {
    return t("kbThesisDescription");
  }
  if (id === "viwiki") {
    return t("kbWikiDescription");
  }
  if (id === "product-notes") {
    return t("kbProductDescription");
  }
  return t("knowledgeDescription");
}
