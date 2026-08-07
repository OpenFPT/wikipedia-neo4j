import { useNavigate } from "react-router";
import { Button } from "@/components/ui/button";
import {
  ErrorActions,
  ErrorDescription,
  ErrorHeader,
  ErrorView,
} from "@/features/errors/error-base";
import { useI18n } from "@/features/i18n/i18n";

export default function NotFoundErrorPage() {
  const navigate = useNavigate();
  const { t } = useI18n();
  return (
    <ErrorView>
      <ErrorHeader>{t("pageNotFound")}</ErrorHeader>
      <ErrorDescription>{t("pageNotFoundDescription")}</ErrorDescription>
      <ErrorActions>
        <Button onClick={() => navigate(-1)} size="lg">
          {t("back")}
        </Button>
      </ErrorActions>
    </ErrorView>
  );
}

// Necessary for react router to lazy load.
export const Component = NotFoundErrorPage;
