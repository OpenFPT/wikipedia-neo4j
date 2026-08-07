import {
  ArrowLeft,
  ArrowRight,
  Cloud,
  Database,
  HardDrive,
  Languages,
} from "lucide-react";
import { useState } from "react";
import type { ChatModel, ModelId } from "@/features/chat/types";
import { useI18n } from "@/features/i18n/i18n";
import type { KnowledgeBase, KnowledgeScope } from "@/features/knowledge/types";
import { scopeForKnowledgeBase } from "@/features/knowledge/types";
import { cn } from "@/lib/utils";

interface OnboardingPageProps {
  knowledgeBases: KnowledgeBase[];
  models: ChatModel[];
  onComplete: (scope: KnowledgeScope) => void;
  onModelChange: (modelId: ModelId) => void;
  selectedModel: ModelId;
}

export function OnboardingPage({
  knowledgeBases,
  models,
  selectedModel,
  onModelChange,
  onComplete,
}: OnboardingPageProps) {
  const { locale, setLocale, t } = useI18n();
  const [step, setStep] = useState(1);
  const [scope, setScope] = useState<KnowledgeScope>("none");

  return (
    <div className="min-h-0 flex-1 overflow-y-auto px-6 py-8 max-[720px]:px-4">
      <div className="mx-auto flex min-h-full max-w-2xl flex-col justify-center">
        <div className="mb-8">
          <span className="text-muted-foreground text-xs">
            {t("stepOf", { step, total: 3 })}
          </span>
          <div className="mt-2 grid grid-cols-3 gap-1">
            {[1, 2, 3].map((item) => (
              <span
                className={cn(
                  "h-1 rounded-full bg-muted",
                  item <= step && "bg-primary"
                )}
                key={item}
              />
            ))}
          </div>
        </div>

        {step === 1 && (
          <section>
            <Languages className="mb-4 size-5 text-primary" />
            <h1 className="font-medium text-2xl tracking-tight">
              {t("welcomeToKnowledge")}
            </h1>
            <p className="mt-2 text-muted-foreground text-sm leading-relaxed">
              {t("onboardingIntro")}
            </p>
            <h2 className="mt-8 mb-2 font-medium text-sm">
              {t("chooseLanguage")}
            </h2>
            <div className="divide-y rounded-xl border">
              <OnboardingOption
                icon={Languages}
                label="English"
                onClick={() => setLocale("en")}
                selected={locale === "en"}
              />
              <OnboardingOption
                icon={Languages}
                label="Tiếng Việt"
                onClick={() => setLocale("vi")}
                selected={locale === "vi"}
              />
            </div>
          </section>
        )}

        {step === 2 && (
          <section>
            <HardDrive className="mb-4 size-5 text-primary" />
            <h1 className="font-medium text-2xl tracking-tight">
              {t("chooseModel")}
            </h1>
            <p className="mt-2 text-muted-foreground text-sm leading-relaxed">
              {t("modelStepDescription")}
            </p>
            <div className="mt-8 divide-y rounded-xl border">
              {models.map((model) => (
                <OnboardingOption
                  description={`${model.meta} · ${model.kind === "local" ? t("onDevice") : t("providers")}`}
                  icon={model.kind === "local" ? HardDrive : Cloud}
                  key={model.id}
                  label={model.name}
                  onClick={() => onModelChange(model.id)}
                  selected={selectedModel === model.id}
                />
              ))}
            </div>
          </section>
        )}

        {step === 3 && (
          <section>
            <Database className="mb-4 size-5 text-primary" />
            <h1 className="font-medium text-2xl tracking-tight">
              {t("addFirstKnowledge")}
            </h1>
            <p className="mt-2 text-muted-foreground text-sm leading-relaxed">
              {t("knowledgeStepDescription")}
            </p>
            <div className="mt-8 divide-y rounded-xl border">
              {knowledgeBases
                .filter((knowledge) => knowledge.status === "ready")
                .slice(0, 2)
                .map((knowledge) => {
                  const knowledgeScope = scopeForKnowledgeBase(knowledge.id);
                  const displayName =
                    locale === "vi" && knowledge.nameVi
                      ? knowledge.nameVi
                      : knowledge.name;
                  return (
                    <OnboardingOption
                      description={
                        knowledge.kind === "hosted"
                          ? t("hostedKnowledgeBase")
                          : t("localKnowledgeBase")
                      }
                      icon={knowledge.kind === "hosted" ? Cloud : HardDrive}
                      key={knowledge.id}
                      label={displayName}
                      onClick={() => setScope(knowledgeScope)}
                      selected={scope === knowledgeScope}
                    />
                  );
                })}
              <OnboardingOption
                icon={Database}
                label={t("continueWithoutKnowledge")}
                onClick={() => setScope("none")}
                selected={scope === "none"}
              />
            </div>
          </section>
        )}

        <div className="mt-8 flex items-center justify-between border-t pt-5">
          {step > 1 ? (
            <button
              className="inline-flex h-9 items-center gap-2 rounded-lg px-3 text-sm hover:bg-muted"
              onClick={() => setStep((current) => current - 1)}
              type="button"
            >
              <ArrowLeft className="size-3.5" />
              {t("back")}
            </button>
          ) : (
            <span />
          )}
          <button
            className="inline-flex h-9 items-center gap-2 rounded-lg bg-primary px-3.5 font-medium text-primary-foreground text-sm hover:bg-primary/90"
            onClick={() => {
              if (step < 3) {
                setStep((current) => current + 1);
              } else {
                onComplete(scope);
              }
            }}
            type="button"
          >
            {step === 3 ? t("startChat") : t("continue")}
            <ArrowRight className="size-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
}

function OnboardingOption({
  icon: OptionIcon,
  label,
  description,
  selected,
  onClick,
}: {
  description?: string;
  icon: typeof Languages;
  label: string;
  onClick: () => void;
  selected: boolean;
}) {
  return (
    <button
      aria-pressed={selected}
      className={cn(
        "flex w-full items-center gap-3 px-4 py-3.5 text-left hover:bg-muted",
        selected && "bg-muted"
      )}
      onClick={onClick}
      type="button"
    >
      <span
        className={cn(
          "grid size-9 place-items-center rounded-lg bg-background text-muted-foreground",
          selected && "bg-primary/10 text-primary"
        )}
      >
        <OptionIcon className="size-4" />
      </span>
      <span className="min-w-0 flex-1">
        <strong className="block truncate font-medium text-sm">{label}</strong>
        {description && (
          <span className="mt-0.5 block truncate text-muted-foreground text-xs">
            {description}
          </span>
        )}
      </span>
    </button>
  );
}
