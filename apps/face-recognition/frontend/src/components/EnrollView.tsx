import * as React from "react";
import { AlertCircle, CheckCircle2, FolderOpen, Loader2, ShieldCheck, UserPlus } from "lucide-react";
import { BatchEnrollView } from "@/components/BatchEnrollView";
import { CameraCapture, type CapturedPhoto } from "@/components/CameraCapture";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { createIdentity, enrollFace } from "@/lib/api";
import { cn } from "@/lib/utils";
import { EnrollResponse, FaceApiError, IdentityRecord, RecognizeParams } from "@/types/face";

interface EnrollViewProps {
  identities: IdentityRecord[];
  recognizeParams: RecognizeParams;
  onEnrolled: () => Promise<void> | void;
  onInspectInRecognize?: (file: File) => void;
}

export function EnrollView({
  identities,
  recognizeParams,
  onEnrolled,
  onInspectInRecognize,
}: EnrollViewProps) {
  const [flow, setFlow] = React.useState<"single" | "batch">("single");
  const activeIdentities = React.useMemo(
    () => identities.filter((item) => item.source === "participant" && item.consent_status === "granted"),
    [identities],
  );

  const [mode, setMode] = React.useState<"new" | "existing">("new");
  const [displayLabel, setDisplayLabel] = React.useState("");
  const [selectedId, setSelectedId] = React.useState<string>("");
  const [purpose, setPurpose] = React.useState("control_acceso_demo");
  const [consentGranted, setConsentGranted] = React.useState(false);

  const [photo, setPhoto] = React.useState<CapturedPhoto | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [saved, setSaved] = React.useState<EnrollResponse | null>(null);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    if (mode === "existing" && !selectedId && activeIdentities.length > 0) {
      setSelectedId(String(activeIdentities[0].identity_id));
    }
  }, [activeIdentities, mode, selectedId]);

  const handleCapture = (next: CapturedPhoto | null) => {
    setPhoto(next);
    setSaved(null);
    setError(null);
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!photo) return;

    setError(null);
    setSaved(null);

    if (mode === "new") {
      if (!displayLabel.trim()) {
        setError("Ingresa un nombre o etiqueta para identificar a la persona.");
        return;
      }
      if (!consentGranted) {
        setError("El consentimiento explícito es obligatorio para registrar y guardar un rostro.");
        return;
      }
    } else if (!selectedId) {
      setError("Selecciona una identidad activa para agregar la foto.");
      return;
    }

    setLoading(true);
    try {
      let targetId = Number(selectedId);
      if (mode === "new") {
        const created = await createIdentity({
          source: "participant",
          display_label: displayLabel.trim(),
          consent_granted: consentGranted,
          purpose: purpose.trim() || "control_acceso_demo",
          notice_version: "v1.0",
        });
        targetId = created.identity_id;
      }

      const enrolled = await enrollFace(targetId, photo.blob, photo.filename);
      setSaved(enrolled);
      if (mode === "new") {
        setSelectedId(String(targetId));
      }
      await onEnrolled();
    } catch (err) {
      if (err instanceof FaceApiError) {
        if (err.code === "no_face_detected") {
          setError("No se detectó ningún rostro en la imagen. Toma una nueva foto mirando de frente.");
        } else if (err.code === "multiple_faces_detected") {
          setError("Se detectaron varios rostros. Para registrar debe aparecer exactamente una persona.");
        } else {
          setError(err.message);
        }
      } else if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("Error inesperado al registrar la muestra facial.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-foreground">Registrar rostros</h2>
          <p className="text-sm text-muted-foreground">
            Agrega una foto con cámara o procesa una carpeta completa con control de consentimiento.
          </p>
        </div>

        <div
          className="grid w-full grid-cols-2 gap-1 rounded-lg border border-border bg-muted p-1 sm:w-auto"
          role="group"
          aria-label="Modo de registro"
        >
          <button
            type="button"
            onClick={() => setFlow("single")}
            className={cn(
              "inline-flex min-h-10 items-center justify-center gap-1.5 whitespace-nowrap rounded-md px-3 text-xs font-medium transition-colors sm:text-sm",
              flow === "single"
                ? "bg-card text-foreground shadow-xs"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            <UserPlus className="h-4 w-4 shrink-0" />
            Una foto
          </button>
          <button
            type="button"
            onClick={() => setFlow("batch")}
            className={cn(
              "inline-flex min-h-10 items-center justify-center gap-1.5 whitespace-nowrap rounded-md px-3 text-xs font-medium transition-colors sm:text-sm",
              flow === "batch"
                ? "bg-card text-foreground shadow-xs"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            <FolderOpen className="h-4 w-4 shrink-0" />
            Carpeta / Lote
          </button>
        </div>
      </div>

      {flow === "batch" ? (
        <BatchEnrollView
          identities={identities}
          recognizeParams={recognizeParams}
          onEnrolled={onEnrolled}
          onInspectInRecognize={onInspectInRecognize}
        />
      ) : (
        <div className="grid grid-cols-1 gap-5 md:grid-cols-[minmax(0,1.08fr)_minmax(0,0.92fr)] md:items-start">
      <Card>
        <CardHeader>
          <CardTitle>Foto para registrar</CardTitle>
          <CardDescription>
            Captura una foto clara con un único rostro visible para guardarla en el índice.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <CameraCapture
            captured={photo}
            onCapture={handleCapture}
            bbox={saved?.bbox ?? null}
            bboxTone={saved ? "verified" : "neutral"}
            busy={loading}
          />
        </CardContent>
      </Card>

      <div className="flex flex-col gap-4">
        <Card>
          <CardHeader>
            <CardTitle>Datos y consentimiento</CardTitle>
            <CardDescription>
              Cada identidad almacena sus embeddings únicamente tras confirmar consentimiento.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="flex flex-col gap-4">
              <div className="grid grid-cols-2 gap-1.5 rounded-lg bg-muted p-1" role="group" aria-label="Tipo de registro">
                <button
                  type="button"
                  onClick={() => setMode("new")}
                  className={`min-h-10 whitespace-nowrap rounded-md px-2.5 text-xs sm:text-sm font-medium transition-colors ${
                    mode === "new" ? "bg-card text-foreground shadow-xs" : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  Nueva persona
                </button>
                <button
                  type="button"
                  onClick={() => setMode("existing")}
                  disabled={activeIdentities.length === 0}
                  className={`min-h-10 whitespace-nowrap rounded-md px-2.5 text-xs sm:text-sm font-medium transition-colors disabled:opacity-45 ${
                    mode === "existing" ? "bg-card text-foreground shadow-xs" : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  Existente ({activeIdentities.length})
                </button>
              </div>

              {mode === "new" ? (
                <>
                  <div className="flex flex-col gap-1.5">
                    <label htmlFor="enroll-name" className="text-sm font-medium text-foreground">
                      Nombre o etiqueta visible
                    </label>
                    <Input
                      id="enroll-name"
                      value={displayLabel}
                      onChange={(e) => setDisplayLabel(e.target.value)}
                      placeholder="Ej. Ana Pérez"
                      maxLength={120}
                      disabled={loading}
                    />
                  </div>

                  <div className="flex flex-col gap-1.5">
                    <label htmlFor="enroll-purpose" className="text-sm font-medium text-foreground">
                      Propósito del registro
                    </label>
                    <Input
                      id="enroll-purpose"
                      value={purpose}
                      onChange={(e) => setPurpose(e.target.value)}
                      placeholder="Ej. control_acceso_demo"
                      maxLength={120}
                      disabled={loading}
                    />
                  </div>

                  <label
                    htmlFor="enroll-consent"
                    className="flex items-start gap-3 rounded-lg border border-border bg-muted/40 p-3 text-sm cursor-pointer"
                  >
                    <Checkbox
                      id="enroll-consent"
                      checked={consentGranted}
                      onCheckedChange={(checked) => setConsentGranted(checked === true)}
                      disabled={loading}
                      className="mt-0.5"
                    />
                    <span className="leading-snug text-foreground">
                      Confirmo consentimiento explícito del participante para extraer y guardar su embedding facial en esta base de datos.
                    </span>
                  </label>
                </>
              ) : (
                <div className="flex flex-col gap-1.5">
                  <label htmlFor="enroll-existing" className="text-sm font-medium text-foreground">
                    Identidad con consentimiento activo
                  </label>
                  <select
                    id="enroll-existing"
                    value={selectedId}
                    onChange={(e) => setSelectedId(e.target.value)}
                    disabled={loading}
                    className="min-h-11 rounded-lg border border-border bg-card px-3 text-base text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  >
                    {activeIdentities.map((person) => (
                      <option key={person.identity_id} value={person.identity_id}>
                        {person.display_label || person.source_subject_id} (#{person.identity_id} · {person.sample_count} fotos)
                      </option>
                    ))}
                  </select>
                </div>
              )}

              {error && (
                <div role="alert" className="flex items-start gap-2 rounded-lg border border-destructive/30 bg-destructive-soft p-3 text-sm text-destructive">
                  <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                  <span>{error}</span>
                </div>
              )}

              <Button type="submit" size="lg" disabled={!photo || loading} className="w-full">
                {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <UserPlus className="h-4 w-4" />}
                {loading ? "Guardando muestra..." : "Guardar rostro"}
              </Button>
            </form>
          </CardContent>
        </Card>

        {saved && (
          <Card className="border-verified/35 bg-verified-soft/45">
            <CardHeader>
              <div className="flex items-center justify-between gap-2">
                <Badge variant="verified">
                  <CheckCircle2 className="h-3.5 w-3.5" />
                  Rostro guardado
                </Badge>
                <span className="tabular-metrics text-xs text-muted-foreground">
                  Muestra #{saved.sample_id}
                </span>
              </div>
              <CardTitle className="mt-1 text-lg">
                {saved.display_label || `Identidad #${saved.identity_id}`}
              </CardTitle>
              <CardDescription className="flex items-center gap-1.5">
                <ShieldCheck className="h-4 w-4 text-verified" />
                Indexado en <span className="tabular-metrics font-medium text-foreground">{saved.embedding_dim}-d</span> ({saved.model_name}) en{" "}
                <span className="tabular-metrics font-medium text-foreground">
                  {(saved.timings_ms.extraction_ms + saved.timings_ms.insert_ms).toFixed(1)} ms
                </span>.
              </CardDescription>
            </CardHeader>
          </Card>
        )}
      </div>
    </div>
      )}
    </div>
  );
}
