import { Menu, PanelLeftOpen, PanelRight } from "lucide-react";
import { useEffect, useState } from "react";
import { type Locale, useI18n } from "@/features/i18n/i18n";
import { KnowledgePage } from "@/features/knowledge/components/knowledge-page";
import {
  initialKnowledgeBases,
  type KnowledgeBase,
  type KnowledgeScope,
  scopeForKnowledgeBase,
} from "@/features/knowledge/types";
import { ModelSelector } from "@/features/models/components/model-selector";
import { ModelsPage } from "@/features/models/components/models-page";
import { OnboardingPage } from "@/features/onboarding/components/onboarding-page";
import { SettingsPage } from "@/features/settings/components/settings-page";
import { ThemeToggle } from "@/features/theme/components/theme-toggle";
import {
  Sidebar,
  type WorkspaceView,
} from "@/features/workspace/components/sidebar";
import { cn } from "@/lib/utils";
import {
  getInitialSources,
  getMockStages,
  mockModels,
} from "../api/mock-chat";
import { runBackendChat } from "../api/backend-chat";
import type {
  ChatMessage,
  EvidenceSource,
  ModelId,
  RetrievalStage,
} from "../types";
import { Composer } from "./composer";
import { Conversation } from "./conversation";
import { EvidencePanel, type StageStatus } from "./evidence-panel";

const onboardingKey = "knowledge-onboarding-complete";

type RunStatus =
  | { kind: "completed"; seconds: string }
  | { kind: "failed" | "ready" | "retrieving" };

export function ChatPage() {
  const { locale, t } = useI18n();
  const [view, setView] = useState<WorkspaceView>("chat");
  const [showOnboarding, setShowOnboarding] = useState(
    () => localStorage.getItem(onboardingKey) !== "true"
  );
  const [scope, setScope] = useState<KnowledgeScope>("none");
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBase[]>(
    initialKnowledgeBases
  );
  const [modelId, setModelId] = useState<ModelId>("recommended");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sources, setSources] = useState<EvidenceSource[]>([]);
  const [stages, setStages] = useState<RetrievalStage[]>(() =>
    getMockStages("none", locale)
  );
  const [stageStatuses, setStageStatuses] = useState<
    Record<string, StageStatus>
  >({});
  const [runStatus, setRunStatus] = useState<RunStatus>({ kind: "ready" });
  const [selectedSourceId, setSelectedSourceId] = useState<string>();
  const [prompt, setPrompt] = useState("");
  const [busy, setBusy] = useState(false);
  const [evidenceOpen, setEvidenceOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [activeThread, setActiveThread] = useState("new");
  const [toast, setToast] = useState("");
  const [debug, setDebug] = useState(false);
  const inspectorVisible = view === "chat" && !showOnboarding && evidenceOpen;
  let statusText = t("ready");
  if (runStatus.kind === "retrieving") {
    statusText = t("searchKnowledge");
  } else if (runStatus.kind === "completed") {
    statusText = t("completedIn", { seconds: runStatus.seconds });
  } else if (runStatus.kind === "failed") {
    statusText = t("simulationFailed");
  }

  useEffect(() => {
    setStages(getMockStages(scope, locale));
    if (activeThread === "architecture") {
      const localizedSources = getInitialSources(locale);
      setSources(localizedSources);
      setMessages(getSavedMessages(locale, localizedSources));
    }
  }, [activeThread, locale, scope]);

  function showToast(message: string) {
    setToast(message);
    globalThis.setTimeout(() => setToast(""), 2400);
  }

  function handleScopeChange(nextScope: KnowledgeScope) {
    if (busy) {
      return;
    }
    setScope(nextScope);
    setSources([]);
    setSelectedSourceId(undefined);
    setStages(getMockStages(nextScope, locale));
    setStageStatuses({});
    setRunStatus({ kind: "ready" });
  }

  async function handleSend() {
    const message = prompt.trim();
    if (!message || busy) {
      return;
    }

    const assistantId = `assistant-${Date.now()}`;
    const requestScope = scope;
    const requestModel = modelId;
    setPrompt("");
    setBusy(true);
    setActiveThread("new");
    setSources([]);
    setStages(getMockStages(requestScope, locale));
    setStageStatuses({});
    setRunStatus({ kind: "retrieving" });
    setMessages((current) => [
      ...current,
      {
        id: `user-${Date.now()}`,
        role: "user",
        content: message,
        time: timeNow(locale),
      },
      {
        id: assistantId,
        role: "assistant",
        content: "",
        time: timeNow(locale),
      },
    ]);

    try {
      // Lightweight, deterministic progress UI (no token streaming).
      const stagesToRun = getMockStages(requestScope, locale);
      setStages(stagesToRun);
      for (const stage of stagesToRun) {
        setStageStatuses((current) => ({ ...current, [stage.id]: "running" }));
        // Small delay so the user sees progress.
        // eslint-disable-next-line no-await-in-loop
        await new Promise((resolve) => globalThis.setTimeout(resolve, 120));
        setStageStatuses((current) => ({ ...current, [stage.id]: "complete" }));
      }

      const result = await runBackendChat({
        message,
        locale,
        model: requestModel,
        debug,
      });

      setSources(result.sources);
      updateMessage(assistantId, (current) => ({
        ...current,
        content: result.answer,
        sources: result.sources,
        usage: result.usage,
      }));
      setRunStatus({
        kind: "completed",
        seconds: (result.usage.elapsedMs / 1000).toFixed(2),
      });
    } catch {
      updateMessage(assistantId, (current) => ({
        ...current,
        content: t("simulationFailed"),
      }));
      setRunStatus({ kind: "failed" });
    } finally {
      setBusy(false);
    }
  }

  function updateMessage(
    messageId: string,
    update: (message: ChatMessage) => ChatMessage
  ) {
    setMessages((current) =>
      current.map((message) =>
        message.id === messageId ? update(message) : message
      )
    );
  }

  function handleNewChat() {
    setShowOnboarding(false);
    setView("chat");
    setMessages([]);
    setSources([]);
    setSelectedSourceId(undefined);
    setScope("none");
    setStageStatuses({});
    setRunStatus({ kind: "ready" });
    setActiveThread("new");
    setEvidenceOpen(false);
    setSidebarOpen(false);
  }

  function handleThreadSelect(threadId: string) {
    setShowOnboarding(false);
    setView("chat");
    setActiveThread(threadId);
    setSidebarOpen(false);
    setEvidenceOpen(false);
    if (threadId === "architecture") {
      const savedSources = getInitialSources(locale);
      const savedStages = getMockStages("all", locale);
      setScope("all");
      setMessages(getSavedMessages(locale, savedSources));
      setSources(savedSources);
      setStages(savedStages);
      setStageStatuses(
        Object.fromEntries(
          savedStages.map((stage) => [stage.id, "complete"])
        ) as Record<string, StageStatus>
      );
      setRunStatus({ kind: "completed", seconds: "1.24" });
      return;
    }
    setMessages([]);
    setSources([]);
    setSelectedSourceId(undefined);
    setScope("none");
    setStageStatuses({});
    setRunStatus({ kind: "ready" });
  }

  function navigate(nextView: WorkspaceView) {
    setShowOnboarding(false);
    setView(nextView);
    setEvidenceOpen(false);
  }

  function useKnowledgeBase(id: string) {
    setShowOnboarding(false);
    setScope(scopeForKnowledgeBase(id));
    setMessages([]);
    setSources([]);
    setView("chat");
    setEvidenceOpen(false);
  }

  function completeOnboarding(nextScope: KnowledgeScope) {
    localStorage.setItem(onboardingKey, "true");
    setScope(nextScope);
    setShowOnboarding(false);
    setView("chat");
  }

  function handleSelectSource(source: EvidenceSource) {
    setSelectedSourceId(source.id);
    setEvidenceOpen(true);
  }

  return (
    <div
      className={cn(
        "grid h-full min-h-0 overflow-hidden bg-background transition-[grid-template-columns] duration-200 max-[720px]:block",
        shellColumns(sidebarCollapsed, inspectorVisible)
      )}
    >
      <Sidebar
        activeThread={activeThread}
        activeView={view}
        collapsed={sidebarCollapsed}
        onClose={() => setSidebarOpen(false)}
        onCollapse={() => setSidebarCollapsed(true)}
        onNavigate={navigate}
        onNewChat={handleNewChat}
        onThreadSelect={handleThreadSelect}
        open={sidebarOpen}
      />

      <main className="flex h-full min-h-0 min-w-0 flex-col bg-background">
        <header className="flex h-15 shrink-0 items-center gap-1 px-4">
          <button
            aria-label={t("openNavigation")}
            className="hidden size-8 place-items-center rounded-full text-muted-foreground hover:bg-muted max-[720px]:grid"
            onClick={() => setSidebarOpen(true)}
            type="button"
          >
            <Menu className="size-4" />
          </button>
          {sidebarCollapsed && (
            <button
              aria-label={t("expandSidebar")}
              className="grid size-8 place-items-center rounded-full text-muted-foreground hover:bg-muted max-[720px]:hidden"
              onClick={() => setSidebarCollapsed(false)}
              title={t("expandSidebar")}
              type="button"
            >
              <PanelLeftOpen className="size-4" />
            </button>
          )}
          {view === "chat" && !showOnboarding && (
            <ModelSelector
              models={mockModels}
              onManage={() => navigate("models")}
              onSelect={setModelId}
              selectedModel={modelId}
            />
          )}
          {showOnboarding && (
            <span className="truncate font-medium text-sm">{t("appName")}</span>
          )}
          {!showOnboarding && view !== "chat" && (
            <h1 className="truncate font-medium text-base">
              {viewTitle(view, t)}
            </h1>
          )}
          <span className="flex-1" />
          {view === "chat" && !showOnboarding && (
            <label className="mr-1 flex select-none items-center gap-2 text-xs text-muted-foreground">
              <input
                checked={debug}
                className="h-4 w-4"
                onChange={(e) => setDebug(e.target.checked)}
                type="checkbox"
              />
              Debug
            </label>
          )}
          <ThemeToggle />
          {view === "chat" && !showOnboarding && (
            <button
              aria-label={t("context")}
              className="relative grid size-8 shrink-0 place-items-center rounded-full text-muted-foreground hover:bg-muted hover:text-foreground"
              onClick={() => setEvidenceOpen((current) => !current)}
              title={t("context")}
              type="button"
            >
              <PanelRight className="size-4" />
              {sources.length > 0 && (
                <span className="absolute top-0.5 right-0 grid h-4 min-w-4 place-items-center rounded-full bg-primary px-1 text-[10px] text-primary-foreground leading-none">
                  {sources.length}
                </span>
              )}
            </button>
          )}
        </header>

        {showOnboarding && (
          <OnboardingPage
            knowledgeBases={knowledgeBases}
            models={mockModels}
            onComplete={completeOnboarding}
            onModelChange={setModelId}
            selectedModel={modelId}
          />
        )}
        {!showOnboarding && view === "chat" && (
          <div className="flex min-h-0 flex-1 flex-col">
            <Conversation
              messages={messages}
              onSelectSource={handleSelectSource}
              streaming={busy}
            />
            <Composer
              busy={busy}
              hasMessages={messages.length > 0}
              knowledgeBases={knowledgeBases}
              onAttach={() => navigate("knowledge")}
              onChange={setPrompt}
              onScopeChange={handleScopeChange}
              onSubmit={handleSend}
              scope={scope}
              value={prompt}
            />
          </div>
        )}
        {!showOnboarding && view === "knowledge" && (
          <KnowledgePage
            knowledgeBases={knowledgeBases}
            onChange={setKnowledgeBases}
            onToast={showToast}
            onUseKnowledgeBase={useKnowledgeBase}
          />
        )}
        {!showOnboarding && view === "models" && (
          <ModelsPage
            models={mockModels}
            onSelect={setModelId}
            onToast={showToast}
            selectedModel={modelId}
          />
        )}
        {!showOnboarding && view === "settings" && <SettingsPage />}
      </main>

      {view === "chat" && !showOnboarding && (
        <EvidencePanel
          onClose={() => setEvidenceOpen(false)}
          onSelectSource={handleSelectSource}
          open={evidenceOpen}
          selectedSourceId={selectedSourceId}
          sources={sources}
          stageStatuses={stageStatuses}
          stages={stages}
          statusText={statusText}
        />
      )}

      <div
        className={cn(
          "fixed right-5 bottom-5 z-50 max-w-[300px] rounded-lg bg-foreground px-3.5 py-2.5 text-background text-xs shadow-lg transition",
          toast
            ? "translate-y-0 opacity-100"
            : "pointer-events-none translate-y-2 opacity-0"
        )}
        role="status"
      >
        {toast}
      </div>
    </div>
  );
}

function shellColumns(sidebarCollapsed: boolean, inspectorOpen: boolean) {
  if (sidebarCollapsed && inspectorOpen) {
    return "grid-cols-[0px_minmax(0,1fr)_360px] max-[1120px]:grid-cols-[0px_minmax(0,1fr)]";
  }
  if (sidebarCollapsed) {
    return "grid-cols-[0px_minmax(0,1fr)]";
  }
  if (inspectorOpen) {
    return "grid-cols-[256px_minmax(0,1fr)_360px] max-[1120px]:grid-cols-[256px_minmax(0,1fr)]";
  }
  return "grid-cols-[256px_minmax(0,1fr)]";
}

function getSavedMessages(
  locale: Locale,
  sources: EvidenceSource[]
): ChatMessage[] {
  const content =
    locale === "vi"
      ? "Ứng dụng tìm trong các kho tri thức đã chọn, theo các liên kết thực thể mạnh nhất và kết hợp bằng chứng đã xếp hạng trước khi tạo câu trả lời. Tài liệu riêng tư vẫn ở trên thiết bị.\n\nMỗi nhận định có thể liên kết lại đoạn nguồn và đường dẫn đồ thị đã đưa nó vào ngữ cảnh."
      : "The app searches the selected knowledge bases, follows the strongest entity connections, and combines ranked evidence before generating an answer. Private documents remain on the device.\n\nEach statement can link back to the source passage and graph path that brought it into context.";
  return [
    {
      id: "initial-user",
      role: "user",
      content:
        locale === "vi"
          ? "Ứng dụng kết hợp GraphRAG và Neo4j như thế nào?"
          : "How does the app combine GraphRAG with Neo4j?",
      time: timeNow(locale),
    },
    {
      id: "initial-assistant",
      role: "assistant",
      content,
      time: timeNow(locale),
      sources,
      usage: { model: "recommended", sources: 2, elapsedMs: 1240 },
    },
  ];
}

function viewTitle(view: WorkspaceView, t: ReturnType<typeof useI18n>["t"]) {
  if (view === "knowledge") {
    return t("knowledgeBases");
  }
  if (view === "models") {
    return t("modelsAndProviders");
  }
  return t("settings");
}

function timeNow(locale: Locale) {
  return new Intl.DateTimeFormat(locale === "vi" ? "vi-VN" : "en", {
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date());
}
