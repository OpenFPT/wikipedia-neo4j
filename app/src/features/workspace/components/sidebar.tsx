import {
  BookOpen,
  MessageSquare,
  MoreHorizontal,
  PanelLeftClose,
  Pencil,
  Plus,
  Search,
  Settings,
  SlidersHorizontal,
  Trash2,
  X,
} from "lucide-react";
import {
  type ReactNode,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { createPortal } from "react-dom";
import { useI18n } from "@/features/i18n/i18n";
import { cn } from "@/lib/utils";

export type WorkspaceView = "chat" | "knowledge" | "models" | "settings";

interface SidebarProps {
  activeThread: string;
  activeView: WorkspaceView;
  collapsed: boolean;
  onClose: () => void;
  onCollapse: () => void;
  onNavigate: (view: WorkspaceView) => void;
  onNewChat: () => void;
  onThreadSelect: (threadId: string) => void;
  open: boolean;
}

const navItem =
  "flex h-8 w-full items-center gap-2 rounded-md px-2 text-left text-sm text-sidebar-foreground transition-colors hover:bg-sidebar-accent [&>svg]:text-muted-foreground";

const threadTitles = {
  architecture: "threadArchitecture",
  evaluation: "threadEvaluation",
  model: "threadModel",
} as const;

interface SidebarThread {
  id: string;
  title: string;
}

export function Sidebar({
  open,
  collapsed,
  activeThread,
  activeView,
  onClose,
  onCollapse,
  onNavigate,
  onNewChat,
  onThreadSelect,
}: SidebarProps) {
  const { t } = useI18n();
  const [query, setQuery] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);
  const [renamedThreads, setRenamedThreads] = useState<Record<string, string>>(
    {}
  );
  const [deletedThreadIds, setDeletedThreadIds] = useState<string[]>([]);
  const [renameTarget, setRenameTarget] = useState<SidebarThread>();
  const [deleteTarget, setDeleteTarget] = useState<"all" | SidebarThread>();
  const threads = Object.entries(threadTitles)
    .filter(([id]) => !deletedThreadIds.includes(id))
    .map(([id, titleKey]) => ({
      id,
      title: renamedThreads[id] ?? t(titleKey),
    }));
  const filteredThreads = threads.filter((thread) =>
    thread.title.toLowerCase().includes(query.trim().toLowerCase())
  );

  function navigate(view: WorkspaceView) {
    onNavigate(view);
    onClose();
  }

  return (
    <>
      <button
        aria-label={t("closeNavigation")}
        className={cn(
          "pointer-events-none fixed inset-0 z-20 bg-black/20 opacity-0 transition-opacity min-[721px]:hidden",
          open && "pointer-events-auto opacity-100"
        )}
        onClick={onClose}
        type="button"
      />
      <aside
        className={cn(
          "relative flex h-full min-h-0 flex-col border-sidebar-border border-r bg-sidebar p-2 transition-[transform,opacity] duration-200 max-[720px]:fixed max-[720px]:inset-y-0 max-[720px]:left-0 max-[720px]:z-30 max-[720px]:w-64 max-[720px]:-translate-x-[110%] max-[720px]:shadow-xl",
          collapsed &&
            "min-[721px]:pointer-events-none min-[721px]:-translate-x-[110%] min-[721px]:opacity-0",
          open && "max-[720px]:translate-x-0 max-[720px]:opacity-100"
        )}
      >
        <div className="mb-1 flex h-8 items-center pl-2">
          <strong className="font-medium text-sidebar-foreground text-sm">
            {t("appName")}
          </strong>
          <button
            aria-label={t("collapseSidebar")}
            className="ml-auto grid size-8 place-items-center rounded-full text-muted-foreground hover:bg-sidebar-accent max-[720px]:hidden"
            onClick={onCollapse}
            title={t("collapseSidebar")}
            type="button"
          >
            <PanelLeftClose className="size-4" />
          </button>
          <button
            aria-label={t("closeNavigation")}
            className="ml-auto hidden size-8 place-items-center rounded-full text-muted-foreground hover:bg-sidebar-accent max-[720px]:grid"
            onClick={onClose}
            type="button"
          >
            <X className="size-4" />
          </button>
        </div>

        <nav aria-label={t("appName")} className="grid gap-1">
          <button className={navItem} onClick={onNewChat} type="button">
            <Plus className="size-4" />
            {t("newChat")}
          </button>
          <button
            className={navItem}
            onClick={() => navigate("knowledge")}
            type="button"
          >
            <BookOpen className="size-4" />
            {t("knowledgeBases")}
          </button>
          <button
            className={navItem}
            onClick={() => setSearchOpen(true)}
            type="button"
          >
            <Search className="size-4" />
            {t("searchChats")}
          </button>
        </nav>

        <ChatSectionHeader
          disabled={threads.length === 0}
          onDeleteAll={() => setDeleteTarget("all")}
        />
        <div className="grid min-h-0 gap-1 overflow-y-auto overflow-x-hidden">
          {threads.map((thread) => (
            <ThreadRow
              active={activeView === "chat" && activeThread === thread.id}
              key={thread.id}
              onDelete={() => setDeleteTarget(thread)}
              onRename={() => setRenameTarget(thread)}
              onSelect={() => onThreadSelect(thread.id)}
              thread={thread}
            />
          ))}
          {threads.length === 0 && (
            <p className="px-2 py-3 text-muted-foreground text-xs">
              {t("noChats")}
            </p>
          )}
        </div>

        <div className="mt-auto grid gap-1">
          <button
            className={navItem}
            onClick={() => navigate("models")}
            type="button"
          >
            <SlidersHorizontal className="size-4" />
            {t("modelsAndProviders")}
          </button>
          <button
            className={navItem}
            onClick={() => navigate("settings")}
            type="button"
          >
            <Settings className="size-4" />
            {t("settings")}
          </button>
        </div>
      </aside>
      {renameTarget && (
        <RenameChatDialog
          onClose={() => setRenameTarget(undefined)}
          onSave={(title) => {
            setRenamedThreads((current) => ({
              ...current,
              [renameTarget.id]: title,
            }));
            setRenameTarget(undefined);
          }}
          thread={renameTarget}
        />
      )}
      {deleteTarget && (
        <DeleteChatDialog
          all={deleteTarget === "all"}
          onClose={() => setDeleteTarget(undefined)}
          onDelete={() => {
            if (deleteTarget === "all") {
              setDeletedThreadIds(Object.keys(threadTitles));
              onNewChat();
            } else {
              setDeletedThreadIds((current) => [...current, deleteTarget.id]);
              if (activeThread === deleteTarget.id) {
                onNewChat();
              }
            }
            setDeleteTarget(undefined);
          }}
          title={deleteTarget === "all" ? undefined : deleteTarget.title}
        />
      )}
      {searchOpen && (
        <SearchDialog
          onClose={() => {
            setSearchOpen(false);
            setQuery("");
          }}
          onQueryChange={setQuery}
          onSelect={(threadId) => {
            setSearchOpen(false);
            setQuery("");
            onThreadSelect(threadId);
          }}
          query={query}
          threads={filteredThreads}
        />
      )}
    </>
  );
}

function ChatSectionHeader({
  disabled,
  onDeleteAll,
}: {
  disabled: boolean;
  onDeleteAll: () => void;
}) {
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  const triggerRef = useRef<HTMLButtonElement>(null);
  return (
    <div className="relative mt-3">
      <div className="flex h-8 items-center px-2 text-muted-foreground text-xs">
        {t("chats")}
      </div>
      <button
        aria-expanded={open}
        aria-haspopup="menu"
        aria-label={t("more")}
        className="absolute top-0 right-0 grid size-8 place-items-center rounded-md text-sidebar-foreground hover:bg-sidebar-accent disabled:opacity-40"
        disabled={disabled}
        onClick={() => setOpen((current) => !current)}
        ref={triggerRef}
        type="button"
      >
        <MoreHorizontal className="size-4" />
      </button>
      <FloatingMenu
        anchor={triggerRef.current}
        estimatedHeight={40}
        onClose={() => setOpen(false)}
        open={open}
      >
        <button
          className="flex h-8 w-full items-center gap-2 rounded-md px-2 text-left text-destructive text-sm hover:bg-destructive/10"
          onClick={() => {
            setOpen(false);
            onDeleteAll();
          }}
          role="menuitem"
          type="button"
        >
          <Trash2 className="size-4" />
          {t("deleteAllChats")}
        </button>
      </FloatingMenu>
    </div>
  );
}

function ThreadRow({
  thread,
  active,
  onSelect,
  onRename,
  onDelete,
}: {
  active: boolean;
  onDelete: () => void;
  onRename: () => void;
  onSelect: () => void;
  thread: SidebarThread;
}) {
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  const triggerRef = useRef<HTMLButtonElement>(null);
  return (
    <div className="group relative min-w-0">
      <button
        aria-current={active ? "page" : undefined}
        className={cn(
          "flex h-8 w-full min-w-0 items-center rounded-md px-2 pr-8 text-left text-sidebar-foreground text-sm transition-colors hover:bg-sidebar-accent",
          active && "bg-sidebar-accent font-medium"
        )}
        onClick={onSelect}
        title={thread.title}
        type="button"
      >
        <span className="truncate">{thread.title}</span>
      </button>
      <button
        aria-expanded={open}
        aria-haspopup="menu"
        aria-label={`${t("more")}: ${thread.title}`}
        className={cn(
          "absolute top-0 right-0 grid size-8 place-items-center rounded-md text-sidebar-foreground opacity-0 transition-opacity hover:bg-sidebar-accent focus:opacity-100 group-hover:opacity-100",
          open && "opacity-100"
        )}
        onClick={() => setOpen((current) => !current)}
        ref={triggerRef}
        type="button"
      >
        <MoreHorizontal className="size-4" />
      </button>
      <FloatingMenu
        anchor={triggerRef.current}
        estimatedHeight={72}
        onClose={() => setOpen(false)}
        open={open}
      >
        <button
          className="flex h-8 w-full items-center gap-2 rounded-md px-2 text-left text-sm hover:bg-muted"
          onClick={() => {
            setOpen(false);
            onRename();
          }}
          role="menuitem"
          type="button"
        >
          <Pencil className="size-4 text-muted-foreground" />
          {t("renameChat")}
        </button>
        <button
          className="flex h-8 w-full items-center gap-2 rounded-md px-2 text-left text-destructive text-sm hover:bg-destructive/10"
          onClick={() => {
            setOpen(false);
            onDelete();
          }}
          role="menuitem"
          type="button"
        >
          <Trash2 className="size-4" />
          {t("deleteChat")}
        </button>
      </FloatingMenu>
    </div>
  );
}

function FloatingMenu({
  anchor,
  children,
  estimatedHeight,
  open,
  onClose,
}: {
  anchor: HTMLButtonElement | null;
  children: ReactNode;
  estimatedHeight: number;
  onClose: () => void;
  open: boolean;
}) {
  const menuRef = useRef<HTMLDivElement>(null);
  const width = 176;
  const [position, setPosition] = useState({ left: -9999, top: -9999 });

  useLayoutEffect(() => {
    if (!(open && anchor)) {
      return;
    }
    const rect = anchor.getBoundingClientRect();
    const rightSide = rect.right + 6;
    const left =
      rightSide + width <= window.innerWidth - 8
        ? rightSide
        : Math.max(8, rect.left - width - 6);
    const top = Math.min(
      Math.max(8, rect.top - 4),
      window.innerHeight - estimatedHeight - 8
    );
    setPosition({ left, top });
  }, [anchor, estimatedHeight, open]);

  useEffect(() => {
    if (!open) {
      return;
    }
    function dismiss(event: PointerEvent) {
      const target = event.target as Node;
      if (!(menuRef.current?.contains(target) || anchor?.contains(target))) {
        onClose();
      }
    }
    function dismissOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape") {
        onClose();
      }
    }
    function dismissOnLayoutChange() {
      onClose();
    }
    document.addEventListener("pointerdown", dismiss);
    document.addEventListener("keydown", dismissOnEscape);
    window.addEventListener("resize", dismissOnLayoutChange);
    window.addEventListener("scroll", dismissOnLayoutChange, true);
    return () => {
      document.removeEventListener("pointerdown", dismiss);
      document.removeEventListener("keydown", dismissOnEscape);
      window.removeEventListener("resize", dismissOnLayoutChange);
      window.removeEventListener("scroll", dismissOnLayoutChange, true);
    };
  }, [anchor, onClose, open]);

  if (!open) {
    return null;
  }

  return createPortal(
    <div
      className="fixed z-[60] grid gap-0.5 rounded-lg border bg-popover p-1 text-popover-foreground shadow-xl"
      ref={menuRef}
      role="menu"
      style={{ left: position.left, top: position.top, width }}
    >
      {children}
    </div>,
    document.body
  );
}

function RenameChatDialog({
  thread,
  onClose,
  onSave,
}: {
  onClose: () => void;
  onSave: (title: string) => void;
  thread: SidebarThread;
}) {
  const { t } = useI18n();
  const [title, setTitle] = useState(thread.title);
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center overflow-y-auto p-4">
      <button
        aria-label={t("close")}
        className="absolute inset-0 bg-black/25"
        onClick={onClose}
        type="button"
      />
      <form
        aria-label={t("renameChat")}
        aria-modal="true"
        className="relative w-full max-w-sm rounded-xl border bg-popover p-4 text-popover-foreground shadow-xl"
        onSubmit={(event) => {
          event.preventDefault();
          const nextTitle = title.trim();
          if (nextTitle) {
            onSave(nextTitle);
          }
        }}
        role="dialog"
      >
        <h2 className="font-medium text-base">{t("renameChat")}</h2>
        <p className="mt-1 text-muted-foreground text-xs">
          {t("renameChatDescription")}
        </p>
        <label className="mt-4 grid gap-1.5 font-medium text-xs">
          {t("chatName")}
          <input
            autoFocus
            className="h-10 rounded-lg border bg-background px-3 text-sm outline-none focus:border-ring focus:ring-2 focus:ring-ring/20"
            maxLength={80}
            onChange={(event) => setTitle(event.currentTarget.value)}
            onKeyDown={(event) => {
              if (event.key === "Escape") {
                onClose();
              }
            }}
            value={title}
          />
        </label>
        <div className="mt-4 flex justify-end gap-2">
          <button
            className="h-9 rounded-lg px-3 text-sm hover:bg-muted"
            onClick={onClose}
            type="button"
          >
            {t("cancel")}
          </button>
          <button
            className="h-9 rounded-lg bg-primary px-3.5 font-medium text-primary-foreground text-sm hover:bg-primary/90 disabled:opacity-50"
            disabled={!title.trim()}
            type="submit"
          >
            {t("save")}
          </button>
        </div>
      </form>
    </div>
  );
}

function DeleteChatDialog({
  all,
  title,
  onClose,
  onDelete,
}: {
  all: boolean;
  onClose: () => void;
  onDelete: () => void;
  title?: string;
}) {
  const { t } = useI18n();
  const heading = all ? t("deleteAllChats") : t("deleteChat");
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center overflow-y-auto p-4">
      <button
        aria-label={t("close")}
        className="absolute inset-0 bg-black/25"
        onClick={onClose}
        type="button"
      />
      <section
        aria-label={heading}
        aria-modal="true"
        className="relative w-full max-w-sm rounded-xl border bg-popover p-4 text-popover-foreground shadow-xl"
        role="dialog"
      >
        <h2 className="font-medium text-base">{heading}</h2>
        <p className="mt-1 text-muted-foreground text-xs leading-relaxed">
          {all
            ? t("deleteAllChatsDescription")
            : t("deleteChatDescription", { title: title ?? "" })}
        </p>
        <div className="mt-4 flex justify-end gap-2">
          <button
            className="h-9 rounded-lg px-3 text-sm hover:bg-muted"
            onClick={onClose}
            type="button"
          >
            {t("cancel")}
          </button>
          <button
            className="h-9 rounded-lg bg-destructive px-3.5 font-medium text-sm text-white hover:bg-destructive/90"
            onClick={onDelete}
            type="button"
          >
            {t("delete")}
          </button>
        </div>
      </section>
    </div>
  );
}

function SearchDialog({
  query,
  threads,
  onClose,
  onQueryChange,
  onSelect,
}: {
  onClose: () => void;
  onQueryChange: (query: string) => void;
  onSelect: (threadId: string) => void;
  query: string;
  threads: { id: string; title: string }[];
}) {
  const { t } = useI18n();
  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center px-4 pt-[12vh]">
      <button
        aria-label={t("close")}
        className="absolute inset-0 bg-black/25"
        onClick={onClose}
        type="button"
      />
      <section
        aria-label={t("searchChats")}
        aria-modal="true"
        className="relative w-full max-w-xl overflow-hidden rounded-xl border bg-popover text-popover-foreground shadow-xl"
        role="dialog"
      >
        <div className="flex h-12 items-center gap-2 border-b px-3">
          <Search className="size-4 shrink-0 text-muted-foreground" />
          <input
            aria-label={t("searchChats")}
            autoFocus
            className="min-w-0 flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
            onChange={(event) => onQueryChange(event.currentTarget.value)}
            onKeyDown={(event) => {
              if (event.key === "Escape") {
                onClose();
              } else if (event.key === "Enter" && threads[0]) {
                onSelect(threads[0].id);
              }
            }}
            placeholder={t("searchChats")}
            value={query}
          />
          <button
            aria-label={t("close")}
            className="grid size-8 place-items-center rounded-full text-muted-foreground hover:bg-muted hover:text-foreground"
            onClick={onClose}
            type="button"
          >
            <X className="size-4" />
          </button>
        </div>
        <div className="max-h-80 overflow-y-auto p-2">
          {threads.length > 0 ? (
            threads.map((thread) => (
              <button
                className="flex h-9 w-full items-center gap-2 rounded-md px-3 text-left text-sm hover:bg-muted"
                key={thread.id}
                onClick={() => onSelect(thread.id)}
                type="button"
              >
                <MessageSquare className="size-4 shrink-0 text-muted-foreground" />
                <span className="truncate">{thread.title}</span>
              </button>
            ))
          ) : (
            <div className="grid min-h-32 place-items-center px-4 text-center">
              <div>
                <Search className="mx-auto size-5 text-muted-foreground" />
                <p className="mt-2 text-muted-foreground text-xs">
                  {t("noMatchingChats")}
                </p>
              </div>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
