import { ChevronDown, CircleSlash2, Database, Layers3 } from "lucide-react";
import { useState } from "react";
import { useI18n } from "@/features/i18n/i18n";
import { useDismissableLayer } from "@/hooks/use-dismissable-layer";
import { cn } from "@/lib/utils";
import {
  type KnowledgeBase,
  type KnowledgeScope,
  knowledgeBaseIdFromScope,
  scopeForKnowledgeBase,
} from "../types";

interface KnowledgeSelectorProps {
  knowledgeBases: KnowledgeBase[];
  onChange: (scope: KnowledgeScope) => void;
  scope: KnowledgeScope;
}

export function KnowledgeSelector({
  knowledgeBases,
  scope,
  onChange,
}: KnowledgeSelectorProps) {
  const { locale, t } = useI18n();
  const [open, setOpen] = useState(false);
  const rootRef = useDismissableLayer(open, setOpen);
  const selectedId = knowledgeBaseIdFromScope(scope);
  const selected = knowledgeBases.find(
    (knowledge) => knowledge.id === selectedId
  );
  let label = t("noKnowledge");
  if (scope === "all") {
    label = t("allKnowledgeBases");
  } else if (selected) {
    label =
      locale === "vi" && selected.nameVi ? selected.nameVi : selected.name;
  }

  function select(nextScope: KnowledgeScope) {
    onChange(nextScope);
    setOpen(false);
  }

  return (
    <div className="relative" ref={rootRef}>
      <button
        aria-expanded={open}
        className="flex h-8 max-w-56 items-center gap-1.5 rounded-full bg-secondary px-2.5 text-secondary-foreground text-xs transition-colors hover:bg-muted"
        onClick={() => setOpen((current) => !current)}
        title={label}
        type="button"
      >
        <Database className="size-3.5 shrink-0" />
        <span className="truncate">{label}</span>
        <ChevronDown className="size-3 shrink-0 text-muted-foreground" />
      </button>
      {open && (
        <div className="absolute bottom-full left-0 z-50 mb-2 w-[min(300px,calc(100vw-32px))] overflow-hidden rounded-xl border bg-popover p-1.5 text-popover-foreground shadow-xl">
          <Option
            icon={CircleSlash2}
            label={t("noKnowledge")}
            onSelect={() => select("none")}
            selected={scope === "none"}
          />
          <Option
            icon={Layers3}
            label={t("allKnowledgeBases")}
            onSelect={() => select("all")}
            selected={scope === "all"}
          />
          <div className="my-1 border-t" />
          {knowledgeBases.map((knowledge) => {
            const knowledgeScope = scopeForKnowledgeBase(knowledge.id);
            const name =
              locale === "vi" && knowledge.nameVi
                ? knowledge.nameVi
                : knowledge.name;
            let meta = t("local");
            if (knowledge.status === "offline") {
              meta = t("unavailable");
            } else if (knowledge.status === "indexing") {
              meta = t("indexing");
            } else if (knowledge.kind === "hosted") {
              meta = t("hosted");
            }
            return (
              <Option
                disabled={knowledge.status !== "ready"}
                icon={Database}
                key={knowledge.id}
                label={name}
                meta={meta}
                onSelect={() => select(knowledgeScope)}
                selected={scope === knowledgeScope}
              />
            );
          })}
        </div>
      )}
    </div>
  );
}

function Option({
  icon: OptionIcon,
  label,
  meta,
  selected,
  disabled = false,
  onSelect,
}: {
  disabled?: boolean;
  icon: typeof Database;
  label: string;
  meta?: string;
  onSelect: () => void;
  selected: boolean;
}) {
  return (
    <button
      aria-pressed={selected}
      className={cn(
        "mb-1 flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-left text-xs transition-colors last:mb-0 hover:bg-muted disabled:cursor-not-allowed disabled:opacity-45",
        selected && "bg-muted font-medium"
      )}
      disabled={disabled}
      onClick={onSelect}
      type="button"
    >
      <OptionIcon className="size-3.5 shrink-0 text-muted-foreground" />
      <span className="min-w-0 flex-1 truncate">{label}</span>
      {meta && <span className="text-muted-foreground text-xs">{meta}</span>}
    </button>
  );
}
