import { type ReactNode, Suspense } from "react";
import { ErrorBoundary } from "react-error-boundary";
import { TooltipProvider } from "@/components/ui/tooltip";
import AppErrorPage from "@/features/errors/app-error";
import { I18nProvider } from "@/features/i18n/i18n";
import { ThemeProvider } from "@/features/theme/use-theme";

export default function AppProvider({ children }: { children: ReactNode }) {
  return (
    <I18nProvider>
      <ThemeProvider>
        <Suspense fallback={null}>
          <ErrorBoundary FallbackComponent={AppErrorPage}>
            <TooltipProvider>{children}</TooltipProvider>
          </ErrorBoundary>
        </Suspense>
      </ThemeProvider>
    </I18nProvider>
  );
}
