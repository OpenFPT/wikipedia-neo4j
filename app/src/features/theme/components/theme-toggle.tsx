import { Moon, Sun } from "lucide-react";
import { useI18n } from "@/features/i18n/i18n";
import { useTheme } from "../use-theme";

export function ThemeToggle() {
  const { t } = useI18n();
  const { theme, toggleTheme } = useTheme();
  const dark = theme === "dark";
  const label = dark ? t("useLightTheme") : t("useDarkTheme");

  return (
    <button
      aria-label={label}
      className="grid size-8 shrink-0 place-items-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
      onClick={toggleTheme}
      title={label}
      type="button"
    >
      {dark ? <Sun className="size-4" /> : <Moon className="size-4" />}
    </button>
  );
}
