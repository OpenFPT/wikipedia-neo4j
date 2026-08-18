import { Activity, ChevronDown, ChevronRight, Code2, Wrench } from "lucide-react";
import { useState } from "react";
import { useI18n } from "@/features/i18n/i18n";
import type { ChatBackendEvent } from "../types";

function tone(type: ChatBackendEvent["type"]) {
  if (type === "error") {
    return "border-red-200 bg-red-50/70 text-red-700 dark:border-red-900/50 dark:bg-red-950/20 dark:text-red-300";
  }
  if (type === "fallback") {
    return "border-amber-200 bg-amber-50/70 text-amber-700 dark:border-amber-900/50 dark:bg-amber-950/20 dark:text-amber-300";
  }
  return "border-border/70 bg-background/80 text-foreground";
}

export function BackendProcessingPanel({
  events,
  live,
}: {
  events: ChatBackendEvent[];
  live: boolean;
}) {
  const { t } = useI18n();
  const [open, setOpen] = useState(true);

  if (events.length === 0 && !live) {
    return null;
  }

  return (
    <section className="mt-3 rounded-xl border border-border/70 bg-muted/30">
      <button
        aria-expanded={open}
        className="flex w-full items-center justify-between gap-3 px-3 py-2.5 text-left"
        onClick={() => setOpen((current) => !current)}
        type="button"
      >
        <span className="flex min-w-0 items-center gap-2">
          <span className="grid size-7 shrink-0 place-items-center rounded-full bg-background text-primary">
            <Activity className="size-3.5" />
          </span>
          <span className="min-w-0">
            <span className="block truncate font-medium text-sm">
              {t("streamTitle")}
            </span>
            <span className="block truncate text-muted-foreground text-xs">
              {live ? t("streamLive") : t("streamComplete")}
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
        <div className="space-y-2 border-border/70 border-t px-3 pb-3 pt-2">
          {events.length === 0 ? (
            <p className="text-muted-foreground text-xs">{t("streamWaiting")}</p>
          ) : (
            <ol className="space-y-2">
              {events.map((event) => (
                <li
                  className={`rounded-lg border px-3 py-2 ${tone(event.type)}`}
                  key={event.id}
                >
                  <div className="flex items-start gap-2">
                    <span className="mt-0.5 text-muted-foreground">
                      {event.query ? (
                        <Code2 className="size-3.5" />
                      ) : (
                        <Wrench className="size-3.5" />
                      )}
                    </span>
                    <div className="min-w-0 flex-1">
                      <p className="text-sm leading-6">{event.title}</p>
                      {event.detail && (
                        <p className="mt-1 break-words text-muted-foreground text-xs">
                          {event.detail}
                        </p>
                      )}
                      {event.query && (
                        <pre className="mt-2 overflow-x-auto rounded-md bg-black px-3 py-2 text-[11px] leading-5 text-green-200">
                          <code>{event.query}</code>
                        </pre>
                      )}
                    </div>
                  </div>
                </li>
              ))}
            </ol>
          )}
        </div>
      )}
    </section>
  );
}
