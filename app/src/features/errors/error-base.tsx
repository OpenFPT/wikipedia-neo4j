import type { ReactNode } from "react";
import { useI18n } from "@/features/i18n/i18n";
import { cn } from "@/lib/utils";

export function ErrorView({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  const { t } = useI18n();
  return (
    <main
      className={cn(
        "flex h-full flex-col items-center justify-center bg-background p-8 text-center text-foreground",
        className
      )}
    >
      <div className="max-w-md text-center">
        <p className="font-medium text-primary text-xs">{t("error")}</p>
        {children}
      </div>
    </main>
  );
}

export function ErrorHeader({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <h1
      className={cn(
        "mt-3 font-medium text-2xl text-foreground tracking-tight",
        className
      )}
    >
      {children}
    </h1>
  );
}

export function ErrorDescription({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <p
      className={cn("mt-4 text-muted-foreground text-sm leading-6", className)}
    >
      {children}
    </p>
  );
}

export function ErrorActions({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn("mt-8 flex items-center justify-center gap-x-3", className)}
    >
      {children}
    </div>
  );
}
