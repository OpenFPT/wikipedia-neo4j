import {
  ChevronsUpDown,
  Cloud,
  HardDrive,
  type LucideIcon,
  Settings2,
} from "lucide-react";
import { useState } from "react";
import type { ChatModel, ModelId } from "@/features/chat/types";
import { useI18n } from "@/features/i18n/i18n";
import { useDismissableLayer } from "@/hooks/use-dismissable-layer";
import { cn } from "@/lib/utils";

interface ModelSelectorProps {
  models: ChatModel[];
  onManage: () => void;
  onSelect: (modelId: ModelId) => void;
  selectedModel: ModelId;
}

export function ModelSelector({
  models,
  selectedModel,
  onSelect,
  onManage,
}: ModelSelectorProps) {
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  const rootRef = useDismissableLayer(open, setOpen);
  const activeModel =
    models.find((model) => model.id === selectedModel) ?? models[0];

  if (!activeModel) {
    return null;
  }

  const groups = [
    {
      icon: HardDrive,
      label: t("onDevice"),
      models: models.filter((model) => model.kind === "local"),
    },
    {
      icon: Cloud,
      label: t("providers"),
      models: models.filter((model) => model.kind === "provider"),
    },
  ].filter((group) => group.models.length > 0);

  return (
    <div className="relative z-30" ref={rootRef}>
      <button
        aria-expanded={open}
        className="flex min-w-0 max-w-56 items-center gap-2 rounded-full border bg-background px-3 py-1.5 text-left transition-colors hover:bg-muted"
        onClick={() => setOpen((current) => !current)}
        type="button"
      >
        <ModelMark local={activeModel.kind === "local"} />
        <span className="truncate font-medium text-sm">{activeModel.name}</span>
        <ChevronsUpDown className="size-3.5 shrink-0 text-muted-foreground" />
      </button>

      {open && (
        <div className="absolute top-full left-0 mt-2 w-[min(320px,calc(100vw-24px))] overflow-hidden rounded-xl border bg-popover text-popover-foreground shadow-xl">
          <div className="border-b px-3 py-2.5">
            <strong className="font-medium text-xs">{t("models")}</strong>
            <span className="mt-0.5 block text-muted-foreground text-xs">
              {t("modelDescription")}
            </span>
          </div>
          <div className="max-h-72 overflow-y-auto p-1.5">
            {groups.map((group) => (
              <ModelGroup
                icon={group.icon}
                key={group.label}
                label={group.label}
                models={group.models}
                onSelect={(modelId) => {
                  onSelect(modelId);
                  setOpen(false);
                }}
                selectedModel={selectedModel}
              />
            ))}
          </div>
          <button
            className="flex w-full items-center gap-2 border-t px-3 py-2.5 text-left text-sm transition-colors hover:bg-muted"
            onClick={() => {
              setOpen(false);
              onManage();
            }}
            type="button"
          >
            <Settings2 className="size-3.5" />
            {t("manageModels")}
          </button>
        </div>
      )}
    </div>
  );
}

function ModelGroup({
  icon: GroupIcon,
  label,
  models,
  selectedModel,
  onSelect,
}: {
  icon: LucideIcon;
  label: string;
  models: ChatModel[];
  onSelect: (modelId: ModelId) => void;
  selectedModel: ModelId;
}) {
  const { t } = useI18n();
  return (
    <section className="mb-1 last:mb-0">
      <div className="flex items-center gap-1.5 px-2 py-1.5 font-medium text-muted-foreground text-xs">
        <GroupIcon className="size-3" />
        {label}
      </div>
      {models.map((model) => {
        const selected = model.id === selectedModel;
        let badge = t("connected");
        if (model.badge === "recommended") {
          badge = t("recommended");
        } else if (model.badge === "imported") {
          badge = t("imported");
        }
        return (
          <button
            aria-pressed={selected}
            className={cn(
              "mb-1 flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-left transition-colors last:mb-0 hover:bg-muted",
              selected && "bg-muted font-medium"
            )}
            key={model.id}
            onClick={() => onSelect(model.id)}
            type="button"
          >
            <span className="min-w-0 flex-1">
              <strong className="block truncate font-medium text-sm">
                {model.name}
              </strong>
              <span className="block truncate text-muted-foreground text-xs">
                {model.meta}
              </span>
            </span>
            <span className="text-[10px] text-muted-foreground leading-none">
              {badge}
            </span>
          </button>
        );
      })}
    </section>
  );
}

function ModelMark({ local }: { local: boolean }) {
  return (
    <span
      className={cn(
        "grid size-6 shrink-0 place-items-center rounded-full bg-secondary text-muted-foreground",
        local && "bg-primary/10 text-primary"
      )}
    >
      {local ? <HardDrive className="size-3" /> : <Cloud className="size-3" />}
    </span>
  );
}
