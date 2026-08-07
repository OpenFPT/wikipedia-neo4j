import { Cloud, Download, FileUp, HardDrive, KeyRound } from "lucide-react";
import { useState } from "react";
import type { ChatModel, ModelId } from "@/features/chat/types";
import { useI18n } from "@/features/i18n/i18n";
import { cn } from "@/lib/utils";

interface ModelsPageProps {
  models: ChatModel[];
  onSelect: (modelId: ModelId) => void;
  onToast: (message: string) => void;
  selectedModel: ModelId;
}

export function ModelsPage({
  models,
  selectedModel,
  onSelect,
  onToast,
}: ModelsPageProps) {
  const { t } = useI18n();
  const [providerSaved, setProviderSaved] = useState(true);
  const [apiKey, setApiKey] = useState("");
  const [baseUrl, setBaseUrl] = useState("https://api.openai.com/v1");
  const localModels = models.filter((model) => model.kind === "local");
  const providerModels = models.filter((model) => model.kind === "provider");

  return (
    <div className="min-h-0 flex-1 overflow-y-auto px-6 pb-8 max-[720px]:px-4">
      <div className="mx-auto max-w-4xl">
        <header className="flex items-start justify-between gap-4 py-6">
          <p className="max-w-xl text-muted-foreground text-sm leading-relaxed">
            {t("modelDescription")}
          </p>
          <button
            className="inline-flex h-9 shrink-0 items-center gap-2 rounded-lg border px-3 font-medium text-sm hover:bg-muted"
            onClick={() => onToast(t("imported"))}
            type="button"
          >
            <FileUp className="size-3.5" />
            {t("addLocalModel")}
          </button>
        </header>

        <section className="border-t py-5">
          <div className="mb-3 flex items-center gap-2">
            <HardDrive className="size-4 text-muted-foreground" />
            <h2 className="font-medium text-sm">{t("onDevice")}</h2>
          </div>
          <div className="divide-y rounded-xl border">
            {localModels.map((model) => (
              <ModelRow
                key={model.id}
                model={model}
                onSelect={onSelect}
                selected={selectedModel === model.id}
              />
            ))}
            <div className="flex items-center gap-3 px-4 py-3.5">
              <span className="grid size-9 place-items-center rounded-lg bg-muted text-muted-foreground">
                <Download className="size-4" />
              </span>
              <span className="min-w-0 flex-1">
                <strong className="block font-medium text-sm">
                  ViWiki RAG 14B
                </strong>
                <span className="mt-0.5 block text-muted-foreground text-xs">
                  8.2 GB · {t("modelAvailable")}
                </span>
              </span>
              <button
                className="h-8 rounded-lg border px-3 text-sm hover:bg-muted"
                onClick={() => onToast(t("demo"))}
                type="button"
              >
                {t("download")}
              </button>
            </div>
          </div>
        </section>

        <section className="border-t py-5">
          <div className="mb-3 flex items-center gap-2">
            <Cloud className="size-4 text-muted-foreground" />
            <h2 className="font-medium text-sm">{t("providerConnections")}</h2>
          </div>
          <div className="grid gap-5 rounded-xl border p-4 min-[760px]:grid-cols-[220px_minmax(0,1fr)]">
            <div className="grid content-start gap-1">
              {providerModels.map((model) => (
                <button
                  className={cn(
                    "flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2.5 text-left hover:bg-muted",
                    selectedModel === model.id && "bg-muted"
                  )}
                  key={model.id}
                  onClick={() => onSelect(model.id)}
                  type="button"
                >
                  <span className="grid size-8 place-items-center rounded-lg bg-background text-muted-foreground">
                    <Cloud className="size-4" />
                  </span>
                  <span className="min-w-0 flex-1">
                    <strong className="block truncate font-medium text-sm">
                      {model.name}
                    </strong>
                    <span className="mt-0.5 block text-muted-foreground text-xs">
                      {providerSaved ? t("connected") : t("offline")}
                    </span>
                  </span>
                </button>
              ))}
            </div>
            <form
              className="grid gap-3 border-t pt-4 min-[760px]:border-t-0 min-[760px]:border-l min-[760px]:pt-0 min-[760px]:pl-5"
              onSubmit={(event) => {
                event.preventDefault();
                setProviderSaved(true);
                onToast(t("connectionSaved"));
              }}
            >
              <label className="grid gap-1.5 font-medium text-xs">
                {t("apiKey")}
                <span className="flex h-10 items-center gap-2 rounded-lg border bg-background px-3 focus-within:border-ring focus-within:ring-2 focus-within:ring-ring/20">
                  <KeyRound className="size-3.5" />
                  <input
                    className="min-w-0 flex-1 bg-transparent text-foreground text-sm outline-none"
                    onChange={(event) => {
                      setApiKey(event.currentTarget.value);
                      setProviderSaved(false);
                    }}
                    placeholder={t("apiKeyPlaceholder")}
                    type="password"
                    value={apiKey}
                  />
                </span>
              </label>
              <label className="grid gap-1.5 font-medium text-xs">
                {t("baseUrl")}
                <input
                  className="h-10 rounded-lg border bg-background px-3 text-foreground text-sm outline-none focus:border-ring focus:ring-2 focus:ring-ring/20"
                  onChange={(event) => {
                    setBaseUrl(event.currentTarget.value);
                    setProviderSaved(false);
                  }}
                  type="url"
                  value={baseUrl}
                />
              </label>
              <button
                className="mt-1 h-9 justify-self-end rounded-lg bg-primary px-3.5 font-medium text-primary-foreground text-sm hover:bg-primary/90"
                type="submit"
              >
                {t("saveConnection")}
              </button>
            </form>
          </div>
        </section>
      </div>
    </div>
  );
}

function ModelRow({
  model,
  selected,
  onSelect,
}: {
  model: ChatModel;
  onSelect: (modelId: ModelId) => void;
  selected: boolean;
}) {
  const { t } = useI18n();
  return (
    <div className="flex items-center gap-3 px-4 py-3.5">
      <span className="grid size-9 place-items-center rounded-lg bg-primary/10 text-primary">
        <HardDrive className="size-4" />
      </span>
      <span className="min-w-0 flex-1">
        <strong className="block truncate font-medium text-sm">
          {model.name}
        </strong>
        <span className="mt-0.5 block text-muted-foreground text-xs">
          {model.meta} · {t("modelReady")}
        </span>
      </span>
      {model.badge === "recommended" && (
        <span className="rounded-full bg-primary/10 px-2 py-1 text-[10px] text-primary leading-none">
          {t("recommended")}
        </span>
      )}
      <button
        className={cn(
          "h-8 rounded-lg border px-3 text-sm hover:bg-muted",
          selected && "bg-muted font-medium"
        )}
        onClick={() => onSelect(model.id)}
        type="button"
      >
        {selected ? t("ready") : t("load")}
      </button>
    </div>
  );
}
