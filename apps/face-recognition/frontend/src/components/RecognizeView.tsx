import * as React from "react";
import { AlertCircle, CheckCircle2, HelpCircle, Loader2, ScanSearch } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { CameraCapture, fileToCapturedPhoto, type CapturedPhoto } from "@/components/CameraCapture";
import { SearchSettings } from "@/components/SearchSettings";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { recognizeFace } from "@/lib/api";
import { FaceApiError, RecognizeParams, RecognizeResponse } from "@/types/face";

interface RecognizeViewProps {
  params: RecognizeParams;
  onParamsChange: (next: RecognizeParams) => void;
  initialFile?: File | null;
  onConsumedInitialFile?: () => void;
}

export function RecognizeView({
  params,
  onParamsChange,
  initialFile = null,
  onConsumedInitialFile,
}: RecognizeViewProps) {
  const [photo, setPhoto] = React.useState<CapturedPhoto | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [result, setResult] = React.useState<RecognizeResponse | null>(null);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    if (!initialFile) return;
    let cancelled = false;
    void fileToCapturedPhoto(initialFile).then((loaded) => {
      if (cancelled) {
        URL.revokeObjectURL(loaded.url);
        return;
      }
      setPhoto((prev) => {
        if (prev) URL.revokeObjectURL(prev.url);
        return loaded;
      });
      setResult(null);
      setError(null);
      onConsumedInitialFile?.();
    });
    return () => {
      cancelled = true;
    };
  }, [initialFile, onConsumedInitialFile]);

  const handleCapture = (next: CapturedPhoto | null) => {
    setPhoto(next);
    setResult(null);
    setError(null);
  };

  const handleSearch = async () => {
    if (!photo) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await recognizeFace(photo.blob, params, photo.filename);
      setResult(response);
    } catch (err) {
      if (err instanceof FaceApiError && err.code === "no_face_detected") {
        setError("No se detectó ningún rostro en la foto. Acerca la cara o prueba con otra imagen.");
      } else if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("Error inesperado al buscar el rostro.");
      }
    } finally {
      setLoading(false);
    }
  };

  const bboxTone = result ? (result.registered ? "verified" : "unknown") : "neutral";
  const scorePct = result ? Math.min(100, Math.max(0, Math.round(result.score * 100))) : 0;
  const thresholdPct = result ? Math.min(100, Math.max(0, Math.round(result.threshold * 100))) : 40;

  return (
    <div className="grid grid-cols-1 gap-5 md:grid-cols-[minmax(0,1.08fr)_minmax(0,0.92fr)] md:items-start">
      <Card>
        <CardHeader>
          <CardTitle>Foto para buscar</CardTitle>
          <CardDescription>
            Toma o sube una foto para verificar si la persona existe en la base de datos.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          <CameraCapture
            captured={photo}
            onCapture={handleCapture}
            bbox={result?.bbox ?? null}
            bboxTone={bboxTone}
            busy={loading}
          />
          <Button
            type="button"
            size="lg"
            onClick={() => void handleSearch()}
            disabled={!photo || loading}
            className="w-full"
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <ScanSearch className="h-4 w-4" />}
            {loading ? "Buscando rostro..." : "Buscar en base de datos"}
          </Button>
        </CardContent>
      </Card>

      <div className="flex flex-col gap-4">
        <AnimatePresence mode="wait">
          {error && (
            <motion.div
              key="error"
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={{ duration: 0.16 }}
              role="alert"
              className="rounded-xl border border-destructive/30 bg-destructive-soft p-4 text-sm text-destructive"
            >
              <div className="flex items-start gap-2.5">
                <AlertCircle className="mt-0.5 h-5 w-5 shrink-0" />
                <div className="space-y-1">
                  <p className="font-semibold">No se pudo completar la búsqueda</p>
                  <p>{error}</p>
                </div>
              </div>
            </motion.div>
          )}

          {result && result.registered && (
            <motion.div
              key="matched"
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={{ duration: 0.18 }}
            >
              <Card className="border-verified/35 bg-verified-soft/45">
                <CardHeader>
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <Badge variant="verified">
                      <CheckCircle2 className="h-3.5 w-3.5" />
                      Persona registrada
                    </Badge>
                    <span className="tabular-metrics text-xs text-muted-foreground">
                      ID #{result.identity_id}
                    </span>
                  </div>
                  <CardTitle className="mt-2 text-2xl">{result.label}</CardTitle>
                  <CardDescription>
                    Coincidencia confirmada sobre el índice {result.index_type.toUpperCase()} ({result.embedding_dim}-d).
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-1.5">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-medium text-muted-foreground">Similitud vs umbral</span>
                      <span className="tabular-metrics font-semibold text-verified">
                        {result.score.toFixed(4)} / {result.threshold.toFixed(2)}
                      </span>
                    </div>
                    <div className="relative h-2.5 w-full overflow-hidden rounded-full bg-muted">
                      <div
                        className="h-full rounded-full bg-verified transition-all"
                        style={{ width: `${scorePct}%` }}
                      />
                      <div
                        aria-hidden="true"
                        className="absolute top-0 bottom-0 w-0.5 bg-foreground/70"
                        style={{ left: `${thresholdPct}%` }}
                      />
                    </div>
                  </div>

                  <dl className="grid grid-cols-3 gap-2 rounded-lg border border-border bg-card p-3 text-center">
                    <div>
                      <dt className="text-[11px] text-muted-foreground">Extracción</dt>
                      <dd className="tabular-metrics text-xs font-semibold">{result.timings_ms.extraction_ms.toFixed(1)} ms</dd>
                    </div>
                    <div>
                      <dt className="text-[11px] text-muted-foreground">Búsqueda DBMS</dt>
                      <dd className="tabular-metrics text-xs font-semibold">{result.timings_ms.dbms_search_ms.toFixed(1)} ms</dd>
                    </div>
                    <div>
                      <dt className="text-[11px] text-muted-foreground">Total</dt>
                      <dd className="tabular-metrics text-xs font-semibold">{result.timings_ms.total_ms.toFixed(1)} ms</dd>
                    </div>
                  </dl>
                </CardContent>
              </Card>
            </motion.div>
          )}

          {result && !result.registered && (
            <motion.div
              key="unknown"
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={{ duration: 0.18 }}
            >
              <Card className="border-unknown/35 bg-unknown-soft/65">
                <CardHeader>
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <Badge variant="unknown">
                      <HelpCircle className="h-3.5 w-3.5" />
                      Sin coincidencia
                    </Badge>
                    <span className="tabular-metrics text-xs text-muted-foreground">
                      umbral {result.threshold.toFixed(2)}
                    </span>
                  </div>
                  <CardTitle className="mt-2 text-2xl capitalize">{result.label}</CardTitle>
                  <CardDescription>
                    Esta persona no está registrada o ninguna muestra superó el umbral de similitud configurado.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex items-center justify-between rounded-lg border border-border bg-card px-3 py-2.5 text-xs">
                    <span className="text-muted-foreground">Mejor score observado</span>
                    <span className="tabular-metrics font-semibold text-unknown">
                      {result.score.toFixed(4)}
                    </span>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Tiempo total: <span className="tabular-metrics font-medium text-foreground">{result.timings_ms.total_ms.toFixed(1)} ms</span>
                    {" · "}Rostros detectados: <span className="tabular-metrics font-medium text-foreground">{result.faces_detected}</span>
                  </p>
                </CardContent>
              </Card>
            </motion.div>
          )}

          {!result && !error && (
            <Card key="idle" className="border-dashed bg-card/70">
              <CardHeader>
                <CardTitle className="text-base">Resultado de verificación</CardTitle>
                <CardDescription>
                  Cuando tomes o subas una foto y presiones Buscar en base de datos, verás aquí si la persona está registrada o si es desconocida.
                </CardDescription>
              </CardHeader>
            </Card>
          )}
        </AnimatePresence>

        <SearchSettings params={params} onChange={onParamsChange} />
      </div>
    </div>
  );
}
