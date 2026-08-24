import { ArrowUp, Mic, Plus, Square } from "lucide-react";
import { useEffect, useRef } from "react";
import { useI18n } from "@/features/i18n/i18n";
import { KnowledgeSelector } from "@/features/knowledge/components/knowledge-selector";
import type { KnowledgeBase, KnowledgeScope } from "@/features/knowledge/types";

interface ComposerProps {
  busy: boolean;
  hasMessages: boolean;
  knowledgeBases: KnowledgeBase[];
  onAttach: () => void;
  onChange: (value: string) => void;
  onToggleRecording: () => void;
  onScopeChange: (scope: KnowledgeScope) => void;
  onSubmit: () => void;
  recording: boolean;
  scope: KnowledgeScope;
  speechBusy: boolean;
  speechStatus: string;
  value: string;
}

export function Composer({
  scope,
  value,
  busy,
  hasMessages,
  knowledgeBases,
  onChange,
  onToggleRecording,
  onScopeChange,
  onSubmit,
  onAttach,
  recording,
  speechBusy,
  speechStatus,
}: ComposerProps) {
  const { t } = useI18n();
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const placeholder = t(
    hasMessages ? "followupPlaceholder" : "messagePlaceholder"
  );

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) {
      return;
    }
    textarea.style.height = "auto";
    if (value) {
      textarea.style.height = `${Math.min(textarea.scrollHeight, 180)}px`;
    }
  }, [value]);

  return (
    <div className="mx-auto w-full max-w-3xl shrink-0 px-4 pb-4 max-[720px]:pb-3">
      <form
        className="overflow-visible rounded-[22px] border border-input bg-card pb-2 shadow-[0_8px_30px_rgb(0_0_0_/_5%)] transition focus-within:border-ring/50 focus-within:ring-1 focus-within:ring-ring/20 dark:shadow-none"
        onSubmit={(event) => {
          event.preventDefault();
          onSubmit();
        }}
      >
        <textarea
          aria-label={placeholder}
          className="block max-h-[180px] min-h-[68px] w-full resize-none border-0 bg-transparent px-4 pt-4 pb-2 text-sm leading-6 outline-none placeholder:text-muted-foreground"
          onChange={(event) => onChange(event.currentTarget.value)}
          onKeyDown={(event) => {
            if (
              event.key === "Enter" &&
              !event.shiftKey &&
              !event.nativeEvent.isComposing
            ) {
              event.preventDefault();
              onSubmit();
            }
          }}
          placeholder={placeholder}
          ref={textareaRef}
          rows={1}
          value={value}
        />
        <div className="flex items-center gap-1.5 px-2">
          <button
            aria-label={t("attachDocument")}
            className="grid size-8 place-items-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            onClick={onAttach}
            type="button"
          >
            <Plus className="size-4" />
          </button>
          <button
            aria-label={recording ? t("stopRecording") : t("startRecording")}
            className="grid size-8 place-items-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground disabled:cursor-not-allowed disabled:opacity-40"
            disabled={busy || speechBusy}
            onClick={onToggleRecording}
            type="button"
          >
            {recording ? (
              <Square className="size-4 fill-current" />
            ) : (
              <Mic className="size-4" />
            )}
          </button>
          <KnowledgeSelector
            knowledgeBases={knowledgeBases}
            onChange={onScopeChange}
            scope={scope}
          />
          <span className="ml-auto" />
          <button
            aria-label={t("sendMessage")}
            className="grid size-9 place-items-center rounded-full bg-primary text-primary-foreground transition-colors hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-40"
            disabled={busy || !value.trim()}
            type="submit"
          >
            <ArrowUp className="size-[15px]" />
          </button>
        </div>
        {speechStatus ? (
          <p className="px-4 pt-2 text-muted-foreground text-xs">{speechStatus}</p>
        ) : null}
      </form>
    </div>
  );
}
