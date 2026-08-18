import { Check, LoaderCircle, Sparkles } from "lucide-react";
import { useI18n } from "@/features/i18n/i18n";
import { cn } from "@/lib/utils";
import type { ChatThinkingState, StageStatus } from "../types";

function iconFor(status: StageStatus) {
  if (status === "complete") {
    return <Check className="size-3" />;
  }
  if (status === "running") {
    return <LoaderCircle className="size-3 animate-spin" />;
  }
  return <span className="size-1.5 rounded-full bg-current opacity-60" />;
}

export function ChatThinkingPanel({
  thinking,
}: {
  thinking: ChatThinkingState;
}) {
  const { t } = useI18n();

  return (
    <section
      aria-live="polite"
      className="mt-2 rounded-xl border border-border/70 bg-muted/30 p-3"
    >
      <div className="flex items-start gap-3">
        <span className="grid size-8 shrink-0 place-items-center rounded-full bg-background text-primary">
          <Sparkles className="size-4" />
        </span>
        <div className="min-w-0 flex-1">
          <h3 className="font-medium text-sm">{t("thinkingTitle")}</h3>
          <p className="mt-1 text-muted-foreground text-xs">
            {t("thinkingSubtitle")}
          </p>
        </div>
      </div>

      <ol className="mt-3 grid gap-2">
        {thinking.stages.map((stage) => {
          const status = thinking.stageStatuses[stage.id] ?? "pending";
          return (
            <li
              className={cn(
                "grid grid-cols-[18px_minmax(0,1fr)] items-center gap-2 rounded-lg bg-background/70 px-2.5 py-2 text-xs",
                status === "running" && "text-foreground",
                status !== "running" && "text-muted-foreground"
              )}
              key={stage.id}
            >
              <span
                className={cn(
                  "grid size-4 place-items-center rounded-full border border-border/70",
                  status === "complete" &&
                    "border-transparent bg-primary/10 text-primary",
                  status === "running" &&
                    "border-transparent bg-primary/10 text-primary"
                )}
              >
                {iconFor(status)}
              </span>
              <span className="min-w-0 break-words">{stage.label}</span>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
