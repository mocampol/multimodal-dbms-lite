import { useCallback, useEffect, useState } from "react";
import { MotionConfig } from "motion/react";
import {
  Activity,
  RefreshCw,
  ScanFace,
  UserPlus,
  Users,
} from "lucide-react";

import { EnrollView } from "@/components/EnrollView";
import { IdentitiesView } from "@/components/IdentitiesView";
import { RecognizeView } from "@/components/RecognizeView";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { getHealth, listIdentities } from "@/lib/api";
import { cn } from "@/lib/utils";
import type {
  HealthResponse,
  IdentityRecord,
  RecognizeParams,
} from "@/types/face";

type TabId = "recognize" | "enroll" | "identities";

const DEFAULT_PARAMS: RecognizeParams = {
  k: 5,
  threshold: 0.45,
  index_type: "hnsw",
  metric: "cosine",
  ef_search: 64,
  n_probe: 4,
  aggregation: "max",
};

interface NavItem {
  id: TabId;
  label: string;
  icon: typeof ScanFace;
}

const NAV_ITEMS: NavItem[] = [
  { id: "recognize", label: "Buscar", icon: ScanFace },
  { id: "enroll", label: "Registrar", icon: UserPlus },
  { id: "identities", label: "Identidades", icon: Users },
];

export function App() {
  const [activeTab, setActiveTab] = useState<TabId>("recognize");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [identities, setIdentities] = useState<IdentityRecord[]>([]);
  const [loadingIdentities, setLoadingIdentities] = useState<boolean>(true);
  const [paramsInitialized, setParamsInitialized] = useState<boolean>(false);
  const [params, setParams] = useState<RecognizeParams>(DEFAULT_PARAMS);
  const [inspectedFile, setInspectedFile] = useState<File | null>(null);

  const loadAppState = useCallback(async () => {
    setLoadingIdentities(true);
    const [healthRes, idsRes] = await Promise.allSettled([
      getHealth(),
      listIdentities(),
    ]);

    if (healthRes.status === "fulfilled") {
      const h = healthRes.value;
      setHealth(h);
      setHealthError(null);
      if (!paramsInitialized) {
        setParams((prev) => ({
          ...prev,
          k: h.defaults.k,
          threshold: h.defaults.threshold,
          index_type: h.defaults.index_type,
          metric: h.defaults.metric,
        }));
        setParamsInitialized(true);
      }
    } else {
      setHealth(null);
      setHealthError(
        healthRes.reason instanceof Error
          ? healthRes.reason.message
          : "No fue posible conectar con el backend de reconocimiento facial."
      );
    }

    if (idsRes.status === "fulfilled") {
      setIdentities(idsRes.value.identities);
    }
    setLoadingIdentities(false);
  }, [paramsInitialized]);

  const refreshIdentities = useCallback(async () => {
    setLoadingIdentities(true);
    try {
      const res = await listIdentities();
      setIdentities(res.identities);
    } catch {
      // Keep existing state if background refresh fails; health indicator surfaces offline status.
    } finally {
      setLoadingIdentities(false);
    }
  }, []);

  useEffect(() => {
    void loadAppState();
  }, [loadAppState]);

  const activeCount = identities.filter(
    (item) => item.consent_status !== "withdrawn"
  ).length;

  const statusBadge = (() => {
    if (!health) {
      return {
        variant: "destructive" as const,
        shortLabel: "Sin conexión",
        fullLabel: "Servicio desconectado",
      };
    }
    if (health.status === "ok") {
      return {
        variant: "verified" as const,
        shortLabel: `${health.embedding_dim}d`,
        fullLabel: `En línea · ${health.model} (${health.embedding_dim}-d)`,
      };
    }
    return {
      variant: "unknown" as const,
      shortLabel: "Degradado",
      fullLabel: `DBMS degradado · ${health.model}`,
    };
  })();

  return (
    <MotionConfig reducedMotion="user">
      <div className="min-h-dvh bg-background text-foreground">
        <header className="sticky top-0 z-30 border-b border-border bg-card/95 backdrop-blur-sm">
          <div className="mx-auto flex h-16 max-w-5xl items-center justify-between gap-3 px-4 md:px-6">
            <div className="flex min-w-0 items-center gap-3">
              <div
                aria-hidden="true"
                className="flex size-9 shrink-0 items-center justify-center rounded-md bg-primary text-primary-foreground"
              >
                <ScanFace className="size-5" />
              </div>
              <div className="min-w-0">
                <h1 className="truncate text-base font-semibold tracking-tight text-foreground">
                  Verificación Facial
                </h1>
                <p className="truncate text-xs text-muted-foreground">
                  Índice vectorial independiente
                </p>
              </div>
            </div>

            <nav
              aria-label="Navegación principal"
              className="hidden items-center gap-1 rounded-md border border-border bg-background p-1 md:flex"
            >
              {NAV_ITEMS.map((item) => {
                const Icon = item.icon;
                const active = activeTab === item.id;
                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => setActiveTab(item.id)}
                    aria-current={active ? "page" : undefined}
                    className={cn(
                      "inline-flex h-9 items-center gap-2 rounded-sm px-3 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                      active
                        ? "bg-primary text-primary-foreground shadow-xs"
                        : "text-muted-foreground hover:bg-muted hover:text-foreground"
                    )}
                  >
                    <Icon className="size-4 shrink-0" />
                    <span>{item.label}</span>
                    {item.id === "identities" && (
                      <span
                        className={cn(
                          "tabular-metrics rounded-full px-1.5 py-0.5 text-xs",
                          active
                            ? "bg-primary-foreground/20 text-primary-foreground"
                            : "bg-muted text-foreground"
                        )}
                      >
                        {activeCount}
                      </span>
                    )}
                  </button>
                );
              })}
            </nav>

            <div className="flex shrink-0 items-center gap-2">
              <Badge
                variant={statusBadge.variant}
                title={statusBadge.fullLabel}
                className="gap-1.5"
              >
                <Activity className="size-3.5 shrink-0" />
                <span className="sm:hidden">{statusBadge.shortLabel}</span>
                <span className="hidden sm:inline">{statusBadge.fullLabel}</span>
              </Badge>
            </div>
          </div>
        </header>

        <main className="mx-auto max-w-5xl px-4 pt-5 pb-24 md:px-6 md:pt-8 md:pb-12">
          {healthError && (
            <div
              role="alert"
              className="mb-5 flex flex-col gap-3 rounded-lg border border-destructive/30 bg-destructive-soft p-4 text-sm text-destructive sm:flex-row sm:items-center sm:justify-between"
            >
              <div>
                <p className="font-semibold">
                  Backend de reconocimiento no disponible
                </p>
                <p className="mt-0.5 text-xs text-destructive/90">
                  {healthError}. Verifica que el DBMS (puerto 8000) y la API de
                  rostros (puerto 8001) estén activos.
                </p>
              </div>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => void loadAppState()}
                className="shrink-0 border-destructive/40 bg-card text-destructive hover:bg-destructive/10"
              >
                <RefreshCw className="size-4" />
                Reintentar
              </Button>
            </div>
          )}

          {health && health.status === "degraded" && (
            <div
              role="status"
              className="mb-5 flex flex-col gap-3 rounded-lg border border-unknown/30 bg-unknown-soft p-4 text-sm text-unknown sm:flex-row sm:items-center sm:justify-between"
            >
              <div>
                <p className="font-semibold">
                  La conexión con el DBMS principal está degradada
                </p>
                <p className="mt-0.5 text-xs text-unknown/90">
                  El servicio de rostros respondió, pero el motor de vectores no
                  confirmó estado saludable.
                </p>
              </div>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => void loadAppState()}
                className="shrink-0 border-unknown/40 bg-card text-unknown hover:bg-unknown/10"
              >
                <RefreshCw className="size-4" />
                Comprobar
              </Button>
            </div>
          )}

          {activeTab === "recognize" && (
            <RecognizeView
              params={params}
              onParamsChange={setParams}
              initialFile={inspectedFile}
              onConsumedInitialFile={() => setInspectedFile(null)}
            />
          )}

          {activeTab === "enroll" && (
            <EnrollView
              identities={identities}
              recognizeParams={params}
              onEnrolled={async () => {
                await refreshIdentities();
              }}
              onInspectInRecognize={(file) => {
                setInspectedFile(file);
                setActiveTab("recognize");
              }}
            />
          )}

          {activeTab === "identities" && (
            <IdentitiesView
              identities={identities}
              loading={loadingIdentities}
              onRefresh={refreshIdentities}
            />
          )}
        </main>

        <nav
          aria-label="Navegación móvil"
          className="safe-bottom-bar fixed inset-x-0 bottom-0 z-30 border-t border-border bg-card/95 backdrop-blur-sm md:hidden"
        >
          <div className="mx-auto grid max-w-lg grid-cols-3 gap-1 px-2 pt-1.5">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              const active = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => setActiveTab(item.id)}
                  aria-current={active ? "page" : undefined}
                  className={cn(
                    "relative flex min-h-[50px] flex-col items-center justify-center gap-0.5 rounded-md px-2 py-1.5 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                    active
                      ? "bg-primary/10 text-primary font-semibold"
                      : "text-muted-foreground hover:bg-muted hover:text-foreground"
                  )}
                >
                  <div className="relative">
                    <Icon className="size-5" />
                    {item.id === "identities" && activeCount > 0 && (
                      <span className="tabular-metrics absolute -top-1.5 -right-3 rounded-full bg-primary px-1 text-[10px] leading-4 font-semibold text-primary-foreground">
                        {activeCount}
                      </span>
                    )}
                  </div>
                  <span>{item.label}</span>
                </button>
              );
            })}
          </div>
        </nav>
      </div>
    </MotionConfig>
  );
}

export default App;
