import {
  Check,
  ExternalLink,
  FileText,
  LoaderCircle,
  Network,
  X,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useI18n } from "@/features/i18n/i18n";
import { cn } from "@/lib/utils";
import type { EvidenceSource, RetrievalStage } from "../types";
import { GraphCanvas } from "./graph-canvas";

export type StageStatus = "complete" | "pending" | "running";

type InspectorTab = "evidence" | "graph";

interface EvidencePanelProps {
  onClose: () => void;
  onSelectSource: (source: EvidenceSource) => void;
  open: boolean;
  selectedSourceId?: string;
  sources: EvidenceSource[];
  stageStatuses: Record<string, StageStatus>;
  stages: RetrievalStage[];
  statusText: string;
}

export function EvidencePanel({
  open,
  sources,
  stages,
  stageStatuses,
  statusText,
  selectedSourceId,
  onClose,
  onSelectSource,
}: EvidencePanelProps) {
  const { t } = useI18n();
  const [tab, setTab] = useState<InspectorTab>("evidence");
  const completeCount = stages.filter(
    (stage) => stageStatuses[stage.id] === "complete"
  ).length;
  const progress = stages.length
    ? Math.round((completeCount / stages.length) * 100)
    : 0;
  const groups = useMemo(() => groupSources(sources), [sources]);
  const selectedSource =
    sources.find((source) => source.id === selectedSourceId) ?? sources[0];

  useEffect(() => {
    if (selectedSourceId || sources.length === 0) {
      setTab("evidence");
    }
  }, [selectedSourceId, sources.length]);

  return (
    <>
      <button
        aria-label={t("close")}
        className={cn(
          "pointer-events-none fixed inset-0 z-30 bg-black/20 opacity-0 transition-opacity min-[1121px]:hidden",
          open && "pointer-events-auto opacity-100"
        )}
        onClick={onClose}
        type="button"
      />
      <aside
        aria-label={t("context")}
        className={cn(
          "min-w-0 overflow-y-auto overflow-x-hidden border-l bg-background p-4",
          "max-[1120px]:fixed max-[1120px]:inset-y-0 max-[1120px]:right-0 max-[1120px]:z-40 max-[1120px]:w-[min(360px,calc(100vw-24px))] max-[1120px]:shadow-xl max-[1120px]:transition-transform max-[1120px]:duration-200",
          open
            ? "max-[1120px]:translate-x-0"
            : "pointer-events-none max-[1120px]:translate-x-full min-[1121px]:hidden"
        )}
      >
        <header className="flex items-start justify-between pb-3">
          <div>
            <h2 className="font-medium text-base tracking-tight">
              {t("context")}
            </h2>
            <p className="mt-0.5 text-muted-foreground text-xs">
              {t("retrievalDetails")}
            </p>
          </div>
          <button
            aria-label={t("close")}
            className="grid size-8 place-items-center rounded-full text-muted-foreground hover:bg-muted hover:text-foreground"
            onClick={onClose}
            type="button"
          >
            <X className="size-4" />
          </button>
        </header>

        <div className="mb-4 grid grid-cols-2 gap-1 rounded-lg bg-muted p-0.5">
          <TabButton
            active={tab === "evidence"}
            icon={FileText}
            label={t("evidence")}
            onClick={() => setTab("evidence")}
          />
          <TabButton
            active={tab === "graph"}
            disabled={sources.length === 0}
            icon={Network}
            label={t("graph")}
            onClick={() => setTab("graph")}
          />
        </div>

        {tab === "evidence" ? (
          <div>
            <RetrievalProgress
              progress={progress}
              stageStatuses={stageStatuses}
              stages={stages}
              statusText={statusText}
            />
            <div className="mt-4">
              <div className="mb-2 flex items-center justify-between">
                <h3 className="font-medium text-xs">{t("evidence")}</h3>
                <span className="text-muted-foreground text-xs">
                  {t("found", { count: sources.length })}
                </span>
              </div>
              {sources.length === 0 ? (
                <p className="rounded-xl border border-dashed p-3 text-muted-foreground text-xs">
                  {t("sourcesWillAppear")}
                </p>
              ) : (
                <div className="grid gap-4">
                  {groups.map(([knowledgeBaseName, groupSources]) => (
                    <section key={knowledgeBaseName}>
                      <h4 className="mb-1.5 break-words font-medium text-muted-foreground text-xs">
                        {knowledgeBaseName}
                      </h4>
                      <div className="grid gap-2">
                        {groupSources.map((source) => (
                          <SourceCard
                            key={source.id}
                            onClick={() => onSelectSource(source)}
                            selected={selectedSourceId === source.id}
                            source={source}
                          />
                        ))}
                      </div>
                    </section>
                  ))}
                </div>
              )}
            </div>
          </div>
        ) : (
          <div>
            <div className="mb-3 flex items-center gap-2 text-muted-foreground text-xs">
              <span className="size-1.5 rounded-full bg-primary" />
              {t("retrievedSources", { count: sources.length })}
            </div>
            <GraphCanvas key={selectedSource?.id} source={selectedSource} />
          </div>
        )}
      </aside>
    </>
  );
}

function RetrievalProgress({
  stages,
  stageStatuses,
  statusText,
  progress,
}: {
  progress: number;
  stageStatuses: Record<string, StageStatus>;
  stages: RetrievalStage[];
  statusText: string;
}) {
  const { t } = useI18n();
  return (
    <section aria-live="polite" className="rounded-xl border bg-card p-3">
      <div className="flex items-center gap-2">
        <span className="grid size-7 place-items-center rounded-full bg-primary/10 text-primary">
          <Network className="size-3.5" />
        </span>
        <div className="min-w-0 flex-1">
          <strong className="block font-medium text-xs">
            {t("retrievalRun")}
          </strong>
          <span className="mt-0.5 block break-words text-muted-foreground text-xs">
            {statusText}
          </span>
        </div>
        <span className="text-muted-foreground text-xs tabular-nums">
          {progress}%
        </span>
      </div>
      <div
        aria-label={t("retrievalRun")}
        aria-valuemax={100}
        aria-valuemin={0}
        aria-valuenow={progress}
        className="mt-3 h-1 overflow-hidden rounded-full bg-muted"
        role="progressbar"
      >
        <div
          className="h-full rounded-full bg-primary"
          style={{ width: `${progress}%` }}
        />
      </div>
      <div className="mt-3 grid gap-2.5">
        {stages.map((stage) => {
          const status = stageStatuses[stage.id] ?? "pending";
          return (
            <div
              className={cn(
                "grid grid-cols-[18px_minmax(0,1fr)] items-center gap-2 text-muted-foreground text-xs",
                status === "running" && "font-medium text-foreground"
              )}
              key={stage.id}
            >
              <span
                className={cn(
                  "grid size-4 place-items-center rounded-full border border-border-strong",
                  status === "complete" &&
                    "border-transparent bg-primary/10 text-primary",
                  status === "running" && "border-transparent text-primary"
                )}
              >
                {status === "complete" && <Check className="size-2.5" />}
                {status === "running" && (
                  <LoaderCircle className="size-3 animate-spin" />
                )}
              </span>
              <span className="min-w-0 break-words">{stage.label}</span>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function SourceCard({
  source,
  selected,
  onClick,
}: {
  onClick: () => void;
  selected: boolean;
  source: EvidenceSource;
}) {
  return (
    <button
      aria-pressed={selected}
      className={cn(
        "w-full min-w-0 overflow-hidden rounded-xl border bg-card p-3 text-left transition-colors hover:bg-muted",
        selected && "bg-muted"
      )}
      onClick={onClick}
      type="button"
    >
      <span className="flex items-start gap-2">
        <span className="grid size-6 shrink-0 place-items-center rounded-full bg-muted text-muted-foreground">
          {source.external ? (
            <ExternalLink className="size-3" />
          ) : (
            <FileText className="size-3" />
          )}
        </span>
        <span className="min-w-0 flex-1">
          <strong className="line-clamp-2 block break-words font-medium text-sm">
            {source.title}
          </strong>
          <span className="mt-0.5 block truncate text-muted-foreground text-xs">
            {source.location}
          </span>
        </span>
        <span className="shrink-0 font-medium text-primary text-xs">
          {Math.round(source.score * 100)}%
        </span>
      </span>
      <span className="mt-2 line-clamp-3 block break-words text-muted-foreground text-xs leading-relaxed">
        {source.excerpt}
      </span>
      <span className="mt-2 line-clamp-2 block break-words text-muted-foreground text-xs leading-relaxed">
        {source.path.join(" › ")}
      </span>
    </button>
  );
}

function TabButton({
  active,
  icon: TabIcon,
  label,
  onClick,
  disabled = false,
}: {
  active: boolean;
  disabled?: boolean;
  icon: typeof FileText;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      aria-pressed={active}
      className={cn(
        "flex h-8 items-center justify-center gap-1.5 rounded-md px-2 text-foreground text-xs hover:bg-background/70 disabled:cursor-not-allowed disabled:opacity-40",
        active && "bg-background font-medium shadow-sm"
      )}
      disabled={disabled}
      onClick={onClick}
      type="button"
    >
      <TabIcon className="size-3.5" />
      {label}
    </button>
  );
}

function groupSources(sources: EvidenceSource[]) {
  const groups = new Map<string, EvidenceSource[]>();
  for (const source of sources) {
    const group = groups.get(source.knowledgeBaseName) ?? [];
    group.push(source);
    groups.set(source.knowledgeBaseName, group);
  }
  return [...groups.entries()];
}
