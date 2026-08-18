import { ChevronDown, ChevronRight, Radar, Wrench } from "lucide-react";
import { useMemo, useState } from "react";
import { useI18n } from "@/features/i18n/i18n";
import type { ChatTrace } from "../types";

function prettyValue(value?: string) {
  if (!value) {
    return undefined;
  }

  return value
    .split(/[_./-]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function statusTone(status: string) {
  if (status === "fallback") {
    return "bg-amber-500/10 text-amber-700 dark:text-amber-300";
  }
  if (status === "empty") {
    return "bg-muted text-muted-foreground";
  }
  return "bg-emerald-500/10 text-emerald-700 dark:text-emerald-300";
}

function kindLabel(kind: string, t: ReturnType<typeof useI18n>["t"]) {
  if (kind === "answer") {
    return t("traceKindAnswer");
  }
  if (kind === "ranking") {
    return t("traceKindRanking");
  }
  return t("traceKindRetrieval");
}

function statusLabel(status: string, t: ReturnType<typeof useI18n>["t"]) {
  if (status === "fallback") {
    return t("traceStatusFallback");
  }
  if (status === "empty") {
    return t("traceStatusEmpty");
  }
  return t("traceStatusOk");
}

export function TracePanel({ trace }: { trace: ChatTrace }) {
  const [open, setOpen] = useState(false);
  const { t } = useI18n();
  const panelId = `trace-panel-${trace.steps.length}-${trace.route || "unknown"}-${trace.tier || "unknown"}`;
  const metadata = useMemo(
    () =>
      [
        trace.route
          ? t("traceRoute", { value: prettyValue(trace.route) || trace.route })
          : undefined,
        trace.tier
          ? t("traceTier", { value: prettyValue(trace.tier) || trace.tier })
          : undefined,
        trace.mode
          ? t("traceMode", { value: prettyValue(trace.mode) || trace.mode })
          : undefined,
      ].filter(Boolean) as string[],
    [t, trace.mode, trace.route, trace.tier]
  );

  return (
    <section className="mt-3 rounded-xl border border-border/70 bg-muted/30">
      <button
        aria-controls={panelId}
        aria-expanded={open}
        className="flex w-full items-center justify-between gap-3 px-3 py-2.5 text-left"
        onClick={() => setOpen((current) => !current)}
        type="button"
      >
        <span className="flex min-w-0 items-center gap-2">
          <span className="grid size-7 shrink-0 place-items-center rounded-full bg-background text-muted-foreground">
            <Radar className="size-3.5" />
          </span>
          <span className="min-w-0">
            <span className="block truncate font-medium text-sm">
              {t("traceTitle")}
            </span>
            <span className="block truncate text-muted-foreground text-xs">
              {trace.steps.length === 1
                ? t("traceStepsOne", { count: trace.steps.length })
                : t("traceSteps", { count: trace.steps.length })}
            </span>
          </span>
        </span>
        {open ? (
          <ChevronDown className="size-4 shrink-0 text-muted-foreground" />
        ) : (
          <ChevronRight className="size-4 shrink-0 text-muted-foreground" />
        )}
      </button>

      {open && (
        <div
          className="space-y-3 border-border/70 border-t px-3 pb-3 pt-2"
          id={panelId}
        >
          {metadata.length > 0 && (
            <div className="flex flex-wrap gap-1.5 text-xs">
              {metadata.map((item) => (
                <span
                  className="rounded-full bg-background px-2 py-1 text-muted-foreground"
                  key={item}
                >
                  {item}
                </span>
              ))}
            </div>
          )}

          <ol className="space-y-2">
            {trace.steps.map((step, index) => (
              <li
                className="rounded-lg bg-background/80 px-3 py-2"
                key={`${step.label}-${index + 1}`}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <Wrench className="mt-0.5 size-3.5 shrink-0 text-muted-foreground" />
                      <span className="text-sm leading-6">{step.label}</span>
                    </div>
                    <p className="mt-1 text-muted-foreground text-xs">
                      {kindLabel(step.kind, t)}
                      {typeof step.rowCount === "number"
                        ? ` • ${t("traceRows", { count: step.rowCount })}`
                        : ""}
                      {typeof step.topK === "number"
                        ? ` • ${t("traceTopK", { count: step.topK })}`
                        : ""}
                      {step.errorType ? ` • ${step.errorType}` : ""}
                    </p>
                  </div>
                  <span
                    className={`shrink-0 rounded-full px-2 py-1 text-[11px] font-medium ${statusTone(step.status)}`}
                  >
                    {statusLabel(step.status, t)}
                  </span>
                </div>
              </li>
            ))}
          </ol>
        </div>
      )}
    </section>
  );
}
