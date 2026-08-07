import { relaunch } from "@tauri-apps/plugin-process";
import { Button } from "@/components/ui/button";
import {
  ErrorActions,
  ErrorDescription,
  ErrorHeader,
  ErrorView,
} from "@/features/errors/error-base";
import { useI18n } from "@/features/i18n/i18n";

export default function AppErrorPage() {
  const { t } = useI18n();
  return (
    <ErrorView>
      <ErrorHeader>{t("somethingWentWrong")}</ErrorHeader>
      <ErrorDescription>{t("errorDescription")}</ErrorDescription>
      <ErrorActions>
        <Button onClick={relaunch} size="lg">
          {t("relaunchApp")}
        </Button>
      </ErrorActions>
    </ErrorView>
  );
}
