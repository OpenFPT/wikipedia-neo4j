import {
  Check,
  Copy as CopyIcon,
  ExternalLink,
  LoaderCircle,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useI18n } from "@/features/i18n/i18n";
import type { ChatMessage, EvidenceSource } from "../types";
import { BackendProcessingPanel } from "./backend-processing-panel";
import { ChatThinkingPanel } from "./chat-thinking-panel";
import { TracePanel } from "./trace-panel";

interface ConversationProps {
  messages: ChatMessage[];
  onSelectSource: (source: EvidenceSource) => void;
  streaming: boolean;
}

export function Conversation({
  messages,
  streaming,
  onSelectSource,
}: ConversationProps) {
  const { t } = useI18n();
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (messages.length > 0 || streaming) {
      bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
    }
  }, [messages, streaming]);

  return (
    <section
      aria-label={t("assistantName")}
      className="min-h-0 flex-1 overflow-y-auto scroll-smooth"
    >
      {messages.length === 0 ? (
        <div className="mx-auto flex h-full w-full max-w-3xl flex-col items-center justify-center px-4 pb-6 text-center">
          <h1 className="font-medium text-2xl tracking-tight">
            {t("welcomeTitle")}
          </h1>
          <p className="mt-2 text-muted-foreground text-sm">
            {t("welcomeDescription")}
          </p>
        </div>
      ) : (
        <div className="mx-auto flex w-full max-w-3xl flex-col gap-7 px-4 py-6">
          {messages.map((message) => (
            <Message
              key={message.id}
              message={message}
              onSelectSource={onSelectSource}
            />
          ))}
          <div ref={bottomRef} />
        </div>
      )}
    </section>
  );
}

function Message({
  message,
  onSelectSource,
}: {
  message: ChatMessage;
  onSelectSource: (source: EvidenceSource) => void;
}) {
  const { t } = useI18n();
  const assistant = message.role === "assistant";
  const [copied, setCopied] = useState(false);

  async function copyAnswer() {
    await navigator.clipboard.writeText(message.content);
    setCopied(true);
    globalThis.setTimeout(() => setCopied(false), 1400);
  }

  if (!assistant) {
    return (
      <article className="flex justify-end">
        <p className="max-w-[80%] whitespace-pre-wrap rounded-lg bg-secondary px-3 py-2.5 text-secondary-foreground text-sm leading-6">
          {message.content}
        </p>
      </article>
    );
  }

  return (
    <article className="group/message min-w-0">
      {message.content ? (
        <p className="whitespace-pre-wrap text-foreground text-sm leading-7">
          {message.content}
        </p>
      ) : message.backendEvents && message.backendEvents.length > 0 ? (
        <BackendProcessingPanel events={message.backendEvents} live />
      ) : message.thinking ? (
        <ChatThinkingPanel thinking={message.thinking} />
      ) : (
        <p className="flex items-center gap-2 text-muted-foreground text-sm">
          <LoaderCircle className="size-3.5 animate-spin" />
          {t("generating")}
        </p>
      )}
      {message.backendEvents && message.backendEvents.length > 0 && message.content && (
        <BackendProcessingPanel events={message.backendEvents} live={false} />
      )}
      {message.sources && message.sources.length > 0 && (
        <fieldset className="mt-3 flex flex-wrap gap-1.5">
          <legend className="sr-only">{t("evidence")}</legend>
          {message.sources.map((source, index) => (
            <button
              className="inline-flex h-7 max-w-56 items-center gap-1.5 rounded-full bg-muted px-2 text-xs transition-colors hover:bg-accent"
              key={source.id}
              onClick={() => onSelectSource(source)}
              title={source.title}
              type="button"
            >
              <span className="font-medium tabular-nums">{index + 1}</span>
              <span className="truncate">{source.title}</span>
              {source.external && <ExternalLink className="size-3 shrink-0" />}
            </button>
          ))}
        </fieldset>
      )}
      {message.trace && <TracePanel trace={message.trace} />}
      {message.content && (
        <div className="mt-1 flex h-8 items-center text-muted-foreground opacity-100 focus-within:opacity-100 min-[721px]:opacity-0 min-[721px]:transition-opacity min-[721px]:group-hover/message:opacity-100">
          <button
            aria-label={copied ? t("copied") : t("copy")}
            className="inline-flex h-8 items-center gap-1.5 rounded-md px-2 text-xs hover:bg-muted hover:text-foreground"
            onClick={copyAnswer}
            type="button"
          >
            {copied ? (
              <Check className="size-3.5" />
            ) : (
              <CopyIcon className="size-3.5" />
            )}
            {copied ? t("copied") : t("copy")}
          </button>
        </div>
      )}
    </article>
  );
}
