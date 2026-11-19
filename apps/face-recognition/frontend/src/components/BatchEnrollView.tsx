import * as React from "react";
import {
  AlertCircle,
  CheckCircle2,
  Download,
  FolderOpen,
  HelpCircle,
  Images,
  Loader2,
  RotateCcw,
  ScanSearch,
  Search,
  Square,
  Trash2,
  Upload,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { createIdentity, enrollFace, recognizeFace } from "@/lib/api";
import {
  buildBatchGroups,
  buildCsvReport,
  buildJsonReport,
  entriesFromFileList,
  extractEntriesFromDataTransfer,
  extractEntriesFromDirectoryHandle,
  filterValidImageEntries,
  inferGroupingStrategy,
  mapBatchError,
  RawBatchEntry,
  resolveItemGrouping,
  triggerTextDownload,
} from "@/lib/batchFiles";
import { cn } from "@/lib/utils";
import {
  BatchEnrollItem,
  BatchExistingMatchPolicy,
  BatchGroupingStrategy,
  BatchItemStatus,
  FacePolicy,
  IdentityRecord,
  RecognizeParams,
} from "@/types/face";

interface BatchEnrollViewProps {
  identities: IdentityRecord[];
  recognizeParams: RecognizeParams;
  onEnrolled: () => Promise<void> | void;
  onInspectInRecognize?: (file: File) => void;
}

type StatusFilter = "all" | "registered" | "similar_found" | "error";

const CONCURRENCY_LIMIT = 2;
const LARGE_BATCH_WARNING_THRESHOLD = 100;
const PAGE_SIZE = 50;

function LazyThumb({ file, alt }: { file: File; alt: string }) {
  const [src, setSrc] = React.useState<string>("");

  React.useEffect(() => {
    const url = URL.createObjectURL(file);
    setSrc(url);
    return () => {
      URL.revokeObjectURL(url);
    };
  }, [file]);

  if (!src) {
    return <div className="h-12 w-12 shrink-0 rounded-md border border-border bg-slate-900" />;
  }

  return (
    <img
      src={src}
      alt={alt}
      loading="lazy"
      className="h-12 w-12 shrink-0 rounded-md border border-border object-cover bg-slate-900"
    />
  );
}

export function BatchEnrollView({
  identities,
  recognizeParams,
  onEnrolled,
  onInspectInRecognize,
}: BatchEnrollViewProps) {
  const activeIdentities = React.useMemo(
    () => identities.filter((p) => p.source === "participant" && p.consent_status === "granted"),
    [identities],
  );

  const folderInputRef = React.useRef<HTMLInputElement | null>(null);
  const filesInputRef = React.useRef<HTMLInputElement | null>(null);
  const abortRef = React.useRef<AbortController | null>(null);
  const groupIdentityMapRef = React.useRef<Map<string, number>>(new Map());

  const setFolderInputRef = React.useCallback((node: HTMLInputElement | null) => {
    folderInputRef.current = node;
    if (node) {
      // React 19 asigna propiedades booleanas; webkitdirectory="" se evaluaría como false.
      node.webkitdirectory = true;
      node.multiple = true;
      node.setAttribute("webkitdirectory", "");
      node.setAttribute("directory", "");
    }
  }, []);

  const [items, setItems] = React.useState<BatchEnrollItem[]>([]);
  const [skippedFilesCount, setSkippedFilesCount] = React.useState(0);
  const [dragActive, setDragActive] = React.useState(false);
  const [scanningFolder, setScanningFolder] = React.useState(false);

  const [strategy, setStrategy] = React.useState<BatchGroupingStrategy>("subfolder");
  const [existingPolicy, setExistingPolicy] = React.useState<BatchExistingMatchPolicy>("reuse");
  const [singleMode, setSingleMode] = React.useState<"new" | "existing">("new");
  const [singleLabel, setSingleLabel] = React.useState("");
  const [singleExistingId, setSingleExistingId] = React.useState("");
  const [purpose, setPurpose] = React.useState("control_acceso_demo");
  const [consentGranted, setConsentGranted] = React.useState(false);
  const [facePolicy, setFacePolicy] = React.useState<FacePolicy>("largest");
  const [checkDuplicates, setCheckDuplicates] = React.useState(false);

  const [running, setRunning] = React.useState(false);
  const [batchDurationMs, setBatchDurationMs] = React.useState<number | null>(null);
  const [formError, setFormError] = React.useState<string | null>(null);
  const [statusFilter, setStatusFilter] = React.useState<StatusFilter>("all");
  const [queryFilter, setQueryFilter] = React.useState("");
  const [visibleLimit, setVisibleLimit] = React.useState(PAGE_SIZE);

  React.useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  React.useEffect(() => {
    setVisibleLimit(PAGE_SIZE);
  }, [statusFilter, queryFilter]);

  React.useEffect(() => {
    if (singleMode === "existing" && !singleExistingId && activeIdentities.length > 0) {
      setSingleExistingId(String(activeIdentities[0].identity_id));
    }
  }, [activeIdentities, singleExistingId, singleMode]);

  const loadEntries = React.useCallback(
    (rawEntries: RawBatchEntry[]) => {
      if (running) return;
      const { valid, skippedCount } = filterValidImageEntries(rawEntries);
      groupIdentityMapRef.current.clear();
      setBatchDurationMs(null);
      setFormError(null);
      setStatusFilter("all");
      setQueryFilter("");
      setVisibleLimit(PAGE_SIZE);
      setSkippedFilesCount(skippedCount);

      if (valid.length === 0) {
        setItems([]);
        setFormError(
          skippedCount > 0
            ? "No se encontraron imágenes compatibles (.jpg, .jpeg, .png, .webp) en la selección."
            : "La carpeta seleccionada está vacía.",
        );
        return;
      }

      const inferred = inferGroupingStrategy(valid);
      const nextStrategy = inferred.strategy;
      const nextSingleLabel = inferred.suggestedSingleLabel;
      setStrategy(nextStrategy);
      if (nextSingleLabel) {
        setSingleLabel(nextSingleLabel);
      }

      const nextItems: BatchEnrollItem[] = valid.map((entry, idx) => {
        const grouping = resolveItemGrouping(
          entry.relativePath,
          nextStrategy,
          nextSingleLabel,
          singleMode,
          singleExistingId,
          activeIdentities,
        );
        return {
          id: `${idx}-${entry.relativePath}-${entry.file.size}`,
          file: entry.file,
          relativePath: entry.relativePath,
          groupKey: grouping.groupKey,
          targetLabel: grouping.targetLabel,
          status: "pending",
        };
      });

      setItems(nextItems);
    },
    [activeIdentities, running, singleExistingId, singleMode],
  );

  React.useEffect(() => {
    if (items.length === 0 || running) return;
    setItems((prev) => {
      let changed = false;
      const next = prev.map((item) => {
        if (item.status !== "pending") return item;
        const grouping = resolveItemGrouping(
          item.relativePath,
          strategy,
          singleLabel,
          singleMode,
          singleExistingId,
          activeIdentities,
        );
        if (grouping.groupKey === item.groupKey && grouping.targetLabel === item.targetLabel) {
          return item;
        }
        changed = true;
        return {
          ...item,
          groupKey: grouping.groupKey,
          targetLabel: grouping.targetLabel,
        };
      });
      return changed ? next : prev;
    });
  }, [activeIdentities, items.length, running, singleExistingId, singleLabel, singleMode, strategy]);

  const handleFolderPick = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    event.target.value = "";
    if (!files || files.length === 0) return;
    loadEntries(entriesFromFileList(files));
  };

  const handleOpenFolderDialog = async () => {
    if (running || scanningFolder) return;
    const picker = (
      window as unknown as {
        showDirectoryPicker?: (opts?: { mode?: "read" }) => Promise<Parameters<typeof extractEntriesFromDirectoryHandle>[0]>;
      }
    ).showDirectoryPicker;

    if (typeof picker === "function") {
      try {
        const dirHandle = await picker({ mode: "read" });
        setScanningFolder(true);
        const { entries, skippedByName } = await extractEntriesFromDirectoryHandle(dirHandle, dirHandle.name);
        loadEntries(entries);
        if (skippedByName > 0) {
          setSkippedFilesCount((prev) => prev + skippedByName);
        }
        return;
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") {
          return;
        }
        // Si el entorno embebido bloquea showDirectoryPicker, usa fallback webkitdirectory.
      } finally {
        setScanningFolder(false);
      }
    }

    folderInputRef.current?.click();
  };

  const handleDrop = async (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragActive(false);
    if (running || scanningFolder) return;
    setScanningFolder(true);
    try {
      const raw = await extractEntriesFromDataTransfer(event.dataTransfer);
      loadEntries(raw);
    } finally {
      setScanningFolder(false);
    }
  };

  const handleClearBatch = () => {
    if (running) return;
    abortRef.current?.abort();
    groupIdentityMapRef.current.clear();
    setItems([]);
    setSkippedFilesCount(0);
    setBatchDurationMs(null);
    setFormError(null);
    setStatusFilter("all");
    setQueryFilter("");
    setVisibleLimit(PAGE_SIZE);
  };

  const groups = React.useMemo(
    () => buildBatchGroups(items, activeIdentities),
    [activeIdentities, items],
  );

  const stats = React.useMemo(() => {
    let registered = 0;
    let similar = 0;
    let failed = 0;
    let activeProcessing = 0;
    for (const item of items) {
      if (item.status === "registered") registered += 1;
      else if (item.status === "similar_found") similar += 1;
      else if (item.status === "error") failed += 1;
      else if (item.status === "checking" || item.status === "enrolling") activeProcessing += 1;
    }
    const processed = registered + similar + failed;
    return {
      total: items.length,
      processed,
      registered,
      similar,
      failed,
      activeProcessing,
    };
  }, [items]);

  const hasMatchingExistingGroup = React.useMemo(
    () => groups.some((g) => g.existingIdentityId !== null),
    [groups],
  );

  const idToIndexMap = React.useMemo(() => {
    const map = new Map<string, number>();
    items.forEach((it, idx) => map.set(it.id, idx));
    return map;
  }, [items.length]);

  const updateItemById = React.useCallback(
    (id: string, patch: Partial<BatchEnrollItem>) => {
      setItems((prev) => {
        const idx = idToIndexMap.get(id);
        if (idx === undefined || idx >= prev.length || prev[idx].id !== id) {
          const fallbackIdx = prev.findIndex((it) => it.id === id);
          if (fallbackIdx < 0) return prev;
          const next = prev.slice();
          next[fallbackIdx] = { ...next[fallbackIdx], ...patch };
          return next;
        }
        const next = prev.slice();
        next[idx] = { ...next[idx], ...patch };
        return next;
      });
    },
    [idToIndexMap],
  );

  const ensureGroupIdentityId = React.useCallback(
    async (groupKey: string, displayLabel: string, signal: AbortSignal): Promise<number> => {
      const cached = groupIdentityMapRef.current.get(groupKey);
      if (cached !== undefined) return cached;

      if (groupKey.startsWith("existing:")) {
        const idNum = Number(groupKey.slice("existing:".length));
        groupIdentityMapRef.current.set(groupKey, idNum);
        return idNum;
      }

      if (strategy === "subfolder" && existingPolicy === "reuse") {
        const match = activeIdentities.find(
          (p) => (p.display_label || p.source_subject_id).trim().toLowerCase() === displayLabel.trim().toLowerCase(),
        );
        if (match) {
          groupIdentityMapRef.current.set(groupKey, match.identity_id);
          return match.identity_id;
        }
      }

      const created = await createIdentity(
        {
          source: "participant",
          display_label: displayLabel,
          consent_granted: true,
          purpose: purpose.trim() || "control_acceso_demo",
          notice_version: "v1.0",
        },
        signal,
      );
      groupIdentityMapRef.current.set(groupKey, created.identity_id);
      return created.identity_id;
    },
    [activeIdentities, existingPolicy, purpose, strategy],
  );

  const processBatchItems = React.useCallback(
    async (targetIds: Set<string>, options?: { skipDuplicateCheck?: boolean }) => {
      if (items.length === 0 || running) return;
      setFormError(null);

      if (!consentGranted) {
        setFormError("Debes confirmar el consentimiento explícito antes de registrar la carpeta.");
        return;
      }

      if (strategy === "single" && singleMode === "new" && !singleLabel.trim()) {
        setFormError("Ingresa un nombre para la identidad de esta carpeta.");
        return;
      }
      if (strategy === "single" && singleMode === "existing" && !singleExistingId) {
        setFormError("Selecciona una identidad existente con consentimiento activo.");
        return;
      }

      const targets = items.filter((item) => targetIds.has(item.id));
      if (targets.length === 0) return;

      const controller = new AbortController();
      abortRef.current = controller;
      setRunning(true);
      const startedAt = performance.now();
      let anyDbMutation = false;

      try {
        // Resuelve/crea cada identidad bajo demanda (evita crear miles por adelantado) con promesa compartida por grupo.
        const pendingIdentityPromises = new Map<string, Promise<number>>();
        const resolveIdentityForGroup = (groupKey: string, label: string): Promise<number> => {
          const cached = groupIdentityMapRef.current.get(groupKey);
          if (cached !== undefined) return Promise.resolve(cached);
          const existingPromise = pendingIdentityPromises.get(groupKey);
          if (existingPromise) return existingPromise;

          const promise = ensureGroupIdentityId(groupKey, label, controller.signal).then((id) => {
            anyDbMutation = true;
            return id;
          });
          pendingIdentityPromises.set(groupKey, promise);
          return promise;
        };

        const duplicateThreshold = Math.max(recognizeParams.threshold, 0.55);

        // Ejecutar subida/verificación con cola de concurrencia controlada.
        let cursor = 0;
        const worker = async () => {
          while (!controller.signal.aborted) {
            const idx = cursor;
            cursor += 1;
            if (idx >= targets.length) break;

            const current = targets[idx];
            let targetIdentityId: number | undefined;
            try {
              targetIdentityId = await resolveIdentityForGroup(current.groupKey, current.targetLabel);
              if (checkDuplicates && !options?.skipDuplicateCheck) {
                updateItemById(current.id, {
                  status: "checking",
                  errorCode: undefined,
                  errorMessage: undefined,
                });
                const rec = await recognizeFace(
                  current.file,
                  { ...recognizeParams, threshold: duplicateThreshold },
                  current.file.name,
                  controller.signal,
                );
                if (
                  rec.registered &&
                  rec.identity_id !== null &&
                  rec.identity_id !== targetIdentityId
                ) {
                  updateItemById(current.id, {
                    status: "similar_found",
                    identityId: targetIdentityId,
                    similarIdentityId: rec.identity_id,
                    similarLabel: rec.label,
                    similarScore: rec.score,
                  });
                  continue;
                }
              }

              updateItemById(current.id, {
                status: "enrolling",
                errorCode: undefined,
                errorMessage: undefined,
              });
              const enrolled = await enrollFace(
                targetIdentityId,
                current.file,
                current.file.name,
                controller.signal,
                facePolicy,
              );
              anyDbMutation = true;
              updateItemById(current.id, {
                status: "registered",
                identityId: enrolled.identity_id,
                sampleId: enrolled.sample_id,
                detectionConfidence: enrolled.detection_confidence,
                elapsedMs: enrolled.timings_ms.extraction_ms + enrolled.timings_ms.insert_ms,
                similarIdentityId: undefined,
                similarLabel: undefined,
                similarScore: undefined,
              });
            } catch (err) {
              const mapped = mapBatchError(err);
              if (mapped.code === "aborted") {
                updateItemById(current.id, { status: "pending" });
                break;
              }
              updateItemById(current.id, {
                status: "error",
                identityId: targetIdentityId,
                errorCode: mapped.code,
                errorMessage: mapped.message,
              });
            }
          }
        };

        const workerCount = Math.min(CONCURRENCY_LIMIT, targets.length);
        await Promise.all(Array.from({ length: workerCount }, () => worker()));
      } catch (err) {
        const mapped = mapBatchError(err);
        if (mapped.code !== "aborted") {
          setFormError(mapped.message);
        }
      } finally {
        setBatchDurationMs(performance.now() - startedAt);
        setRunning(false);
        abortRef.current = null;
        if (anyDbMutation) {
          await onEnrolled();
        }
      }
    },
    [
      checkDuplicates,
      consentGranted,
      ensureGroupIdentityId,
      items,
      onEnrolled,
      recognizeParams,
      running,
      singleExistingId,
      singleLabel,
      singleMode,
      strategy,
      updateItemById,
    ],
  );

  const handleStartPending = () => {
    const ids = new Set(items.filter((it) => it.status === "pending").map((it) => it.id));
    void processBatchItems(ids);
  };

  const handleRetryFailed = () => {
    const ids = new Set(items.filter((it) => it.status === "error").map((it) => it.id));
    void processBatchItems(ids);
  };

  const handleForceEnrollSimilar = (itemId: string) => {
    void processBatchItems(new Set([itemId]), { skipDuplicateCheck: true });
  };

  const filteredItems = React.useMemo(() => {
    const q = queryFilter.trim().toLowerCase();
    return items.filter((item) => {
      if (statusFilter !== "all" && item.status !== statusFilter) return false;
      if (!q) return true;
      return (
        item.relativePath.toLowerCase().includes(q) ||
        item.targetLabel.toLowerCase().includes(q) ||
        (item.similarLabel?.toLowerCase().includes(q) ?? false)
      );
    });
  }, [items, queryFilter, statusFilter]);

  const progressPct = stats.total > 0 ? Math.round((stats.processed / stats.total) * 100) : 0;
  const pendingCount = items.filter((it) => it.status === "pending").length;

  const statusMeta: Record<
    BatchItemStatus,
    { label: string; variant: "neutral" | "verified" | "unknown" | "destructive" }
  > = {
    pending: { label: "Pendiente", variant: "neutral" },
    checking: { label: "Verificando", variant: "neutral" },
    enrolling: { label: "Guardando", variant: "neutral" },
    registered: { label: "Registrado", variant: "verified" },
    similar_found: { label: "Similar", variant: "unknown" },
    error: { label: "Fallido", variant: "destructive" },
  };

  return (
    <div className="flex flex-col gap-5">
      <div className="grid grid-cols-1 gap-5 md:grid-cols-[minmax(0,1.02fr)_minmax(0,0.98fr)] md:items-start">
        <Card>
          <CardHeader>
            <CardTitle>Carpeta de rostros</CardTitle>
            <CardDescription>
              Selecciona una carpeta completa o múltiples imágenes (.jpg, .png, .webp) para indexarlas por lote.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <input
              ref={setFolderInputRef}
              type="file"
              multiple
              onChange={handleFolderPick}
              tabIndex={-1}
              aria-hidden="true"
              className="pointer-events-none fixed -left-[9999px] h-px w-px opacity-0"
            />
            <input
              ref={filesInputRef}
              type="file"
              accept="image/*"
              multiple
              onChange={handleFolderPick}
              tabIndex={-1}
              aria-hidden="true"
              className="pointer-events-none fixed -left-[9999px] h-px w-px opacity-0"
            />

            <div
              onDragOver={(e) => {
                e.preventDefault();
                if (!running && !scanningFolder) setDragActive(true);
              }}
              onDragLeave={() => setDragActive(false)}
              onDrop={(e) => void handleDrop(e)}
              className={cn(
                "flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed px-4 py-8 text-center transition-colors",
                dragActive
                  ? "border-primary bg-primary/5"
                  : "border-border bg-muted/30",
              )}
            >
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary/10 text-primary">
                {scanningFolder ? <Loader2 className="h-6 w-6 animate-spin" /> : <FolderOpen className="h-6 w-6" />}
              </div>
              <div className="space-y-1">
                <p className="text-sm font-medium text-foreground">
                  {scanningFolder
                    ? "Leyendo archivos de la carpeta..."
                    : "Arrastra aquí una carpeta o usa los selectores"}
                </p>
                <p className="text-xs text-muted-foreground">
                  Subcarpetas detectan automáticamente la persona (ej. <span className="font-mono">Ana/foto1.jpg</span>).
                </p>
              </div>
              <div className="flex w-full flex-wrap justify-center gap-2 pt-1">
                <Button
                  type="button"
                  onClick={() => void handleOpenFolderDialog()}
                  disabled={running || scanningFolder}
                  className="min-w-[155px] flex-1 sm:flex-none"
                >
                  {scanningFolder ? <Loader2 className="h-4 w-4 animate-spin" /> : <FolderOpen className="h-4 w-4" />}
                  {scanningFolder ? "Leyendo carpeta..." : "Elegir carpeta"}
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => filesInputRef.current?.click()}
                  disabled={running || scanningFolder}
                  className="min-w-[155px] flex-1 sm:flex-none"
                >
                  <Images className="h-4 w-4" />
                  Elegir varias fotos
                </Button>
              </div>
            </div>

            {(items.length > 0 || skippedFilesCount > 0) && (
              <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-border bg-background px-3 py-2.5 text-xs">
                <div className="flex flex-wrap items-center gap-2 text-foreground">
                  <span className="tabular-metrics font-semibold">{items.length} imágenes válidas</span>
                  <span className="text-muted-foreground">·</span>
                  <span className="tabular-metrics text-muted-foreground">{groups.length} grupo(s)</span>
                  {skippedFilesCount > 0 && (
                    <>
                      <span className="text-muted-foreground">·</span>
                      <span className="tabular-metrics text-unknown">
                        {skippedFilesCount} archivo(s) omitidos
                      </span>
                    </>
                  )}
                </div>
                {items.length > 0 && (
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={handleClearBatch}
                    disabled={running}
                    className="h-8 px-2 text-xs text-muted-foreground hover:text-destructive"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                    Limpiar
                  </Button>
                )}
              </div>
            )}

            {items.length > LARGE_BATCH_WARNING_THRESHOLD && (
              <div
                role="status"
                className="rounded-lg border border-unknown/30 bg-unknown-soft px-3 py-2 text-xs text-unknown"
              >
                Lote grande ({items.length} fotos): el procesamiento se ejecutará en cola controlada y puede tomar unos minutos.
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Agrupación y consentimiento</CardTitle>
            <CardDescription>
              Define cómo asignar cada foto a su identidad antes de indexar en la base vectorial.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <div className="flex flex-col gap-1.5">
              <span className="text-xs font-medium text-muted-foreground">
                Modo de asignación de identidades
              </span>
              <div className="grid grid-cols-1 gap-1.5 rounded-lg bg-muted p-1 sm:grid-cols-3" role="group" aria-label="Agrupación de carpeta">
                {(
                  [
                    { id: "subfolder", label: "Por subcarpeta" },
                    { id: "single", label: "Una persona" },
                    { id: "per_file", label: "Por archivo" },
                  ] as const
                ).map((opt) => (
                  <button
                    key={opt.id}
                    type="button"
                    disabled={running}
                    onClick={() => setStrategy(opt.id)}
                    className={cn(
                      "min-h-10 whitespace-nowrap rounded-md px-2.5 text-xs font-medium transition-colors sm:text-sm",
                      strategy === opt.id
                        ? "bg-card text-foreground shadow-xs"
                        : "text-muted-foreground hover:text-foreground",
                    )}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            </div>

            {strategy === "single" && (
              <div className="flex flex-col gap-3 rounded-lg border border-border bg-muted/25 p-3">
                <div className="grid grid-cols-2 gap-1.5 rounded-md bg-muted p-1">
                  <button
                    type="button"
                    disabled={running}
                    onClick={() => setSingleMode("new")}
                    className={cn(
                      "min-h-9 rounded-sm px-2 text-xs font-medium",
                      singleMode === "new" ? "bg-card text-foreground shadow-xs" : "text-muted-foreground",
                    )}
                  >
                    Nueva persona
                  </button>
                  <button
                    type="button"
                    disabled={running || activeIdentities.length === 0}
                    onClick={() => setSingleMode("existing")}
                    className={cn(
                      "min-h-9 rounded-sm px-2 text-xs font-medium disabled:opacity-45",
                      singleMode === "existing" ? "bg-card text-foreground shadow-xs" : "text-muted-foreground",
                    )}
                  >
                    Existente ({activeIdentities.length})
                  </button>
                </div>

                {singleMode === "new" ? (
                  <div className="flex flex-col gap-1.5">
                    <label htmlFor="batch-single-name" className="text-xs font-medium text-foreground">
                      Nombre de la persona para todo el lote
                    </label>
                    <Input
                      id="batch-single-name"
                      value={singleLabel}
                      onChange={(e) => setSingleLabel(e.target.value)}
                      placeholder="Ej. Ana Pérez"
                      disabled={running}
                    />
                  </div>
                ) : (
                  <div className="flex flex-col gap-1.5">
                    <label htmlFor="batch-single-existing" className="text-xs font-medium text-foreground">
                      Identidad existente
                    </label>
                    <select
                      id="batch-single-existing"
                      value={singleExistingId}
                      onChange={(e) => setSingleExistingId(e.target.value)}
                      disabled={running}
                      className="min-h-11 rounded-lg border border-border bg-card px-3 text-sm text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    >
                      {activeIdentities.map((person) => (
                        <option key={person.identity_id} value={person.identity_id}>
                          {person.display_label || person.source_subject_id} (#{person.identity_id} · {person.sample_count} fotos)
                        </option>
                      ))}
                    </select>
                  </div>
                )}
              </div>
            )}

            {strategy === "subfolder" && hasMatchingExistingGroup && (
              <div className="flex flex-col gap-1.5 rounded-lg border border-border bg-muted/25 p-3">
                <span className="text-xs font-medium text-foreground">
                  Algunas subcarpetas coinciden con identidades activas:
                </span>
                <div className="grid grid-cols-2 gap-1.5">
                  <button
                    type="button"
                    disabled={running}
                    onClick={() => setExistingPolicy("reuse")}
                    className={cn(
                      "min-h-9 rounded-md border px-2.5 text-xs font-medium transition-colors",
                      existingPolicy === "reuse"
                        ? "border-primary bg-primary/10 text-primary"
                        : "border-border bg-card text-muted-foreground",
                    )}
                  >
                    Reutilizar existente
                  </button>
                  <button
                    type="button"
                    disabled={running}
                    onClick={() => setExistingPolicy("create_new")}
                    className={cn(
                      "min-h-9 rounded-md border px-2.5 text-xs font-medium transition-colors",
                      existingPolicy === "create_new"
                        ? "border-primary bg-primary/10 text-primary"
                        : "border-border bg-card text-muted-foreground",
                    )}
                  >
                    Crear nueva igual
                  </button>
                </div>
              </div>
            )}

            {groups.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {groups.slice(0, 6).map((group) => (
                  <Badge key={group.groupKey} variant="neutral" className="gap-1">
                    <span className="max-w-[130px] truncate">{group.displayLabel}</span>
                    <span className="tabular-metrics text-muted-foreground">({group.fileCount})</span>
                  </Badge>
                ))}
                {groups.length > 6 && (
                  <Badge variant="neutral">+{groups.length - 6} grupos más</Badge>
                )}
              </div>
            )}

            <div className="flex flex-col gap-1.5">
              <label htmlFor="batch-purpose" className="text-xs font-medium text-foreground">
                Propósito del registro
              </label>
              <Input
                id="batch-purpose"
                value={purpose}
                onChange={(e) => setPurpose(e.target.value)}
                placeholder="Ej. control_acceso_demo"
                disabled={running}
              />
            </div>

            <div className="flex flex-col gap-1.5">
              <label htmlFor="batch-face-policy" className="text-xs font-medium text-foreground">
                Política de detección por foto
              </label>
              <select
                id="batch-face-policy"
                value={facePolicy}
                onChange={(e) => setFacePolicy(e.target.value as FacePolicy)}
                disabled={running}
                className="min-h-11 rounded-lg border border-border bg-card px-3 text-sm text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                <option value="largest">Rostro principal (recomendado para carpetas / LFW)</option>
                <option value="strict_single">Estricto: rechazar si hay más de 1 rostro</option>
              </select>
            </div>

            <label
              htmlFor="batch-check-duplicates"
              className="flex cursor-pointer items-start gap-3 rounded-lg border border-border bg-background p-3 text-xs sm:text-sm"
            >
              <Checkbox
                id="batch-check-duplicates"
                checked={checkDuplicates}
                onCheckedChange={(checked) => setCheckDuplicates(checked === true)}
                disabled={running}
                className="mt-0.5"
              />
              <span className="leading-snug text-foreground">
                Verificar si el rostro ya existe con otro nombre antes de registrar (más lento; umbral estricto{" "}
                <span className="tabular-metrics font-medium">
                  {Math.max(recognizeParams.threshold, 0.55).toFixed(2)}
                </span>
                ).
              </span>
            </label>

            <label
              htmlFor="batch-consent"
              className="flex cursor-pointer items-start gap-3 rounded-lg border border-border bg-muted/40 p-3 text-xs sm:text-sm"
            >
              <Checkbox
                id="batch-consent"
                checked={consentGranted}
                onCheckedChange={(checked) => setConsentGranted(checked === true)}
                disabled={running}
                className="mt-0.5"
              />
              <span className="leading-snug text-foreground">
                Confirmo consentimiento explícito para extraer y almacenar los embeddings faciales de todas las personas incluidas en esta carpeta.
              </span>
            </label>

            {formError && (
              <div
                role="alert"
                className="flex items-start gap-2 rounded-lg border border-destructive/30 bg-destructive-soft p-3 text-sm text-destructive"
              >
                <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                <span>{formError}</span>
              </div>
            )}

            <div className="flex flex-wrap gap-2">
              {!running ? (
                <Button
                  type="button"
                  size="lg"
                  onClick={handleStartPending}
                  disabled={items.length === 0 || pendingCount === 0}
                  className="w-full"
                >
                  <Upload className="h-4 w-4" />
                  Registrar carpeta ({pendingCount})
                </Button>
              ) : (
                <Button
                  type="button"
                  variant="destructive"
                  size="lg"
                  onClick={() => abortRef.current?.abort()}
                  className="w-full"
                >
                  <Square className="h-4 w-4" />
                  Detener lote ({stats.processed}/{stats.total})
                </Button>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {items.length > 0 && (
        <Card>
          <CardHeader className="gap-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <CardTitle>Progreso y resultados del lote</CardTitle>
                <CardDescription>
                  Procesadas{" "}
                  <span className="tabular-metrics font-medium text-foreground">
                    {stats.processed}/{stats.total}
                  </span>{" "}
                  ({progressPct}%)
                  {batchDurationMs !== null && !running && (
                    <>
                      {" "}
                      en <span className="tabular-metrics font-medium text-foreground">{(batchDurationMs / 1000).toFixed(1)} s</span>
                    </>
                  )}
                </CardDescription>
              </div>

              <div className="flex flex-wrap gap-2">
                {stats.failed > 0 && !running && (
                  <Button type="button" variant="outline" size="sm" onClick={handleRetryFailed}>
                    <RotateCcw className="h-4 w-4" />
                    Reintentar fallidos ({stats.failed})
                  </Button>
                )}
                {stats.processed > 0 && (
                  <>
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={() =>
                        triggerTextDownload(
                          buildCsvReport(items),
                          `reporte_lote_${Date.now()}.csv`,
                          "text/csv;charset=utf-8",
                        )
                      }
                    >
                      <Download className="h-4 w-4" />
                      CSV
                    </Button>
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={() =>
                        triggerTextDownload(
                          buildJsonReport(items),
                          `reporte_lote_${Date.now()}.json`,
                          "application/json;charset=utf-8",
                        )
                      }
                    >
                      <Download className="h-4 w-4" />
                      JSON
                    </Button>
                  </>
                )}
              </div>
            </div>

            <div className="h-2.5 w-full overflow-hidden rounded-full bg-muted">
              <div
                className="h-full rounded-full bg-primary transition-all"
                style={{ width: `${progressPct}%` }}
              />
            </div>

            <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-4">
              <div className="rounded-lg border border-border bg-background p-3">
                <p className="text-xs text-muted-foreground">Total</p>
                <p className="tabular-metrics mt-0.5 text-lg font-semibold text-foreground">{stats.total}</p>
              </div>
              <div className="rounded-lg border border-verified/30 bg-verified-soft/45 p-3">
                <p className="text-xs text-verified">Registrados</p>
                <p className="tabular-metrics mt-0.5 text-lg font-semibold text-verified">{stats.registered}</p>
              </div>
              <div className="rounded-lg border border-unknown/30 bg-unknown-soft/55 p-3">
                <p className="text-xs text-unknown">Similares</p>
                <p className="tabular-metrics mt-0.5 text-lg font-semibold text-unknown">{stats.similar}</p>
              </div>
              <div className="rounded-lg border border-destructive/30 bg-destructive-soft/55 p-3">
                <p className="text-xs text-destructive">Fallidos</p>
                <p className="tabular-metrics mt-0.5 text-lg font-semibold text-destructive">{stats.failed}</p>
              </div>
            </div>

            <div className="flex flex-col gap-2.5 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex flex-wrap gap-1.5" role="group" aria-label="Filtrar resultados">
                {(
                  [
                    { id: "all", label: `Todos (${stats.total})` },
                    { id: "registered", label: `Registrados (${stats.registered})` },
                    { id: "similar_found", label: `Similares (${stats.similar})` },
                    { id: "error", label: `Fallidos (${stats.failed})` },
                  ] as const
                ).map((chip) => (
                  <button
                    key={chip.id}
                    type="button"
                    onClick={() => setStatusFilter(chip.id)}
                    className={cn(
                      "min-h-9 rounded-full border px-3 text-xs font-medium transition-colors",
                      statusFilter === chip.id
                        ? "border-primary bg-primary text-primary-foreground"
                        : "border-border bg-card text-muted-foreground hover:text-foreground",
                    )}
                  >
                    {chip.label}
                  </button>
                ))}
              </div>

              <div className="relative w-full sm:w-64">
                <Search className="pointer-events-none absolute top-1/2 left-3 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  value={queryFilter}
                  onChange={(e) => setQueryFilter(e.target.value)}
                  placeholder="Filtrar archivo o persona..."
                  className="pl-9"
                  aria-label="Filtrar resultados por archivo o identidad"
                />
              </div>
            </div>
          </CardHeader>

          <CardContent className="flex flex-col gap-2.5">
            {filteredItems.length === 0 ? (
              <p className="rounded-lg border border-dashed border-border py-8 text-center text-sm text-muted-foreground">
                Ningún elemento coincide con el filtro actual.
              </p>
            ) : (
              <>
                {filteredItems.slice(0, visibleLimit).map((item) => {
                  const badge = statusMeta[item.status];
                  return (
                    <div
                      key={item.id}
                      className="flex flex-col gap-3 rounded-lg border border-border bg-background p-3 sm:flex-row sm:items-center sm:justify-between"
                    >
                      <div className="flex min-w-0 items-center gap-3">
                        <LazyThumb file={item.file} alt={item.file.name} />
                        <div className="min-w-0 flex-1 space-y-0.5">
                          <div className="flex flex-wrap items-center gap-2">
                            <p className="truncate text-sm font-medium text-foreground">{item.targetLabel}</p>
                            <Badge variant={badge.variant} className="gap-1">
                              {(item.status === "checking" || item.status === "enrolling") && (
                                <Loader2 className="h-3 w-3 animate-spin" />
                              )}
                              {item.status === "registered" && <CheckCircle2 className="h-3 w-3" />}
                              {item.status === "similar_found" && <HelpCircle className="h-3 w-3" />}
                              {item.status === "error" && <AlertCircle className="h-3 w-3" />}
                              {badge.label}
                            </Badge>
                          </div>
                          <p className="truncate font-mono text-xs text-muted-foreground" title={item.relativePath}>
                            {item.relativePath}
                          </p>

                          {item.status === "registered" && (
                            <p className="tabular-metrics text-xs text-verified">
                              ID #{item.identityId} · Muestra #{item.sampleId} · conf{" "}
                              {item.detectionConfidence?.toFixed(2)} · {item.elapsedMs?.toFixed(0)} ms
                            </p>
                          )}

                          {item.status === "similar_found" && (
                            <p className="text-xs text-unknown">
                              Similar existente: <span className="font-semibold">{item.similarLabel}</span>{" "}
                              (score <span className="tabular-metrics">{item.similarScore?.toFixed(3)}</span>)
                            </p>
                          )}

                          {item.status === "error" && (
                            <p className="text-xs text-destructive">
                              [{item.errorCode}] {item.errorMessage}
                            </p>
                          )}
                        </div>
                      </div>

                      <div className="flex flex-wrap items-center gap-2 sm:shrink-0">
                        {item.status === "similar_found" && !running && (
                          <Button
                            type="button"
                            variant="outline"
                            size="sm"
                            onClick={() => handleForceEnrollSimilar(item.id)}
                          >
                            Registrar de todos modos
                          </Button>
                        )}
                        {onInspectInRecognize && (
                          <Button
                            type="button"
                            variant="outline"
                            size="sm"
                            onClick={() => onInspectInRecognize(item.file)}
                          >
                            <ScanSearch className="h-3.5 w-3.5" />
                            Ver en Buscar
                          </Button>
                        )}
                      </div>
                    </div>
                  );
                })}

                {filteredItems.length > visibleLimit && (
                  <div className="flex justify-center pt-2">
                    <Button
                      type="button"
                      variant="outline"
                      onClick={() => setVisibleLimit((prev) => prev + PAGE_SIZE)}
                    >
                      Mostrar {Math.min(PAGE_SIZE, filteredItems.length - visibleLimit)} más (mostrando {visibleLimit} de {filteredItems.length})
                    </Button>
                  </div>
                )}
              </>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
