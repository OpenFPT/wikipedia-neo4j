import { Languages, type LucideIcon, Sun } from "lucide-react";
import type { ReactNode } from "react";
import { type Locale, useI18n } from "@/features/i18n/i18n";
import { type Theme, useTheme } from "@/features/theme/use-theme";
import { cn } from "@/lib/utils";

export function SettingsPage() {
  const { locale, setLocale, t } = useI18n();
  const { theme, setTheme } = useTheme();

  return (
    <div className="min-h-0 flex-1 overflow-y-auto px-6 pb-8 max-[720px]:px-4">
      <div className="mx-auto max-w-3xl pt-4">
        <section className="border-t py-5">
          <div className="grid gap-5 min-[680px]:grid-cols-[200px_minmax(0,1fr)]">
            <div>
              <h2 className="font-medium text-sm">{t("appearance")}</h2>
            </div>
            <div className="grid gap-4">
              <SettingRow icon={Sun} label={t("appearance")}>
                <SegmentedControl
                  onChange={(value) => setTheme(value as Theme)}
                  options={[
                    { label: t("light"), value: "light" },
                    { label: t("dark"), value: "dark" },
                  ]}
                  value={theme}
                />
              </SettingRow>
              <SettingRow icon={Languages} label={t("interfaceLanguage")}>
                <SegmentedControl
                  onChange={(value) => setLocale(value as Locale)}
                  options={[
                    { label: t("english"), value: "en" },
                    { label: t("vietnamese"), value: "vi" },
                  ]}
                  value={locale}
                />
              </SettingRow>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

function SettingRow({
  icon: RowIcon,
  label,
  children,
}: {
  children: ReactNode;
  icon: LucideIcon;
  label: string;
}) {
  return (
    <div className="flex flex-wrap items-center gap-3">
      <span className="grid size-8 place-items-center rounded-lg bg-muted text-muted-foreground">
        <RowIcon className="size-4" />
      </span>
      <span className="min-w-28 flex-1 font-medium text-sm">{label}</span>
      {children}
    </div>
  );
}

function SegmentedControl({
  options,
  value,
  onChange,
}: {
  onChange: (value: string) => void;
  options: { label: string; value: string }[];
  value: string;
}) {
  return (
    <div className="inline-flex rounded-lg bg-muted p-0.5">
      {options.map((option) => (
        <button
          aria-pressed={option.value === value}
          className={cn(
            "rounded-md px-3 py-1.5 text-muted-foreground text-xs hover:text-foreground",
            option.value === value &&
              "bg-background font-medium text-foreground shadow-sm"
          )}
          key={option.value}
          onClick={() => onChange(option.value)}
          type="button"
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}
