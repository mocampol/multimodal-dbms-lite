import * as React from "react";
import { RefreshCw, ShieldOff, Trash2, Users } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Dialog, DialogClose, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";
import { purgeAllFaces, withdrawIdentity } from "@/lib/api";
import { IdentityRecord } from "@/types/face";

interface IdentitiesViewProps {
  identities: IdentityRecord[];
  loading: boolean;
  onRefresh: () => Promise<void>;
}

const IDENTITIES_PAGE_SIZE = 60;

export function IdentitiesView({ identities, loading, onRefresh }: IdentitiesViewProps) {
  const [target, setTarget] = React.useState<IdentityRecord | null>(null);
  const [revoking, setRevoking] = React.useState(false);

  const [confirmPurgeAll, setConfirmPurgeAll] = React.useState(false);
  const [alsoDeleteIdentities, setAlsoDeleteIdentities] = React.useState(true);
  const [purgingAll, setPurgingAll] = React.useState(false);

  const [visibleLimit, setVisibleLimit] = React.useState(IDENTITIES_PAGE_SIZE);
  const [banner, setBanner] = React.useState<string | null>(null);
  const [error, setError] = React.useState<string | null>(null);

  const totalSamples = React.useMemo(
    () => identities.reduce((acc, item) => acc + item.sample_count, 0),
    [identities],
  );

  React.useEffect(() => {
    setVisibleLimit(IDENTITIES_PAGE_SIZE);
  }, [identities.length]);

  const handleConfirmWithdraw = async () => {
    if (!target) return;
    setRevoking(true);
    setError(null);
    setBanner(null);
    try {
      const res = await withdrawIdentity(target.identity_id);
      setBanner(
        `Consentimiento revocado para "${target.display_label || target.source_subject_id}" (${res.deleted_samples} muestras eliminadas del índice).`,
      );
      setTarget(null);
      await onRefresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo revocar la identidad.");
    } finally {
      setRevoking(false);
    }
  };

  const handleConfirmPurgeAll = async () => {
    setPurgingAll(true);
    setError(null);
    setBanner(null);
    try {
      const res = await purgeAllFaces(!alsoDeleteIdentities);
      setConfirmPurgeAll(false);
      if (res.samples_only) {
        setBanner(`Se eliminaron ${res.deleted_samples} fotos faciales (se conservaron las identidades).`);
      } else {
        setBanner(
          `Base limpiada: ${res.deleted_samples} fotos y ${res.deleted_identities} identidades eliminadas.`,
        );
      }
      await onRefresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudieron eliminar las fotos registradas.");
    } finally {
      setPurgingAll(false);
    }
  };

  const visibleIdentities = identities.slice(0, visibleLimit);

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-foreground">Identidades registradas</h2>
          <p className="text-sm text-muted-foreground">
            <span className="tabular-metrics font-medium text-foreground">{identities.length}</span> identidades ·{" "}
            <span className="tabular-metrics font-medium text-foreground">{totalSamples}</span> fotos indexadas.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Button type="button" variant="outline" onClick={() => void onRefresh()} disabled={loading || purgingAll}>
            <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
            Actualizar
          </Button>
          <Button
            type="button"
            variant="outline"
            onClick={() => {
              setAlsoDeleteIdentities(true);
              setConfirmPurgeAll(true);
            }}
            disabled={loading || purgingAll || (identities.length === 0 && totalSamples === 0)}
            className="border-destructive/35 text-destructive hover:bg-destructive-soft"
          >
            <Trash2 className="h-4 w-4" />
            Eliminar todas
          </Button>
        </div>
      </div>

      {banner && (
        <div role="status" className="rounded-xl border border-verified/30 bg-verified-soft px-4 py-3 text-sm text-verified">
          {banner}
        </div>
      )}

      {error && (
        <div role="alert" className="rounded-xl border border-destructive/30 bg-destructive-soft px-4 py-3 text-sm text-destructive">
          {error}
        </div>
      )}

      {identities.length === 0 ? (
        <Card className="border-dashed">
          <CardHeader className="items-center py-10 text-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-muted text-muted-foreground">
              <Users className="h-5 w-5" />
            </div>
            <h3 className="mt-2 text-base font-semibold tracking-tight sm:text-lg">
              No hay identidades registradas
            </h3>
            <CardDescription className="max-w-sm">
              Ve a la pestaña Registrar para guardar la primera persona con consentimiento y probar el reconocimiento.
            </CardDescription>
          </CardHeader>
        </Card>
      ) : (
        <>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {visibleIdentities.map((person) => {
              const isActive = person.consent_status === "granted";
              const isWithdrawn = person.consent_status === "withdrawn";
              return (
                <Card key={person.identity_id}>
                  <CardHeader className="pb-3">
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0">
                        <h3 className="truncate text-base font-semibold tracking-tight">
                          {person.display_label || person.source_subject_id}
                        </h3>
                        <p className="tabular-metrics mt-0.5 text-xs text-muted-foreground">
                          ID #{person.identity_id} · {person.source}
                        </p>
                      </div>
                      <Badge variant={isActive ? "verified" : isWithdrawn ? "destructive" : "neutral"}>
                        {person.consent_status}
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent className="flex items-center justify-between gap-2 pt-0">
                    <span className="tabular-metrics text-sm font-medium text-foreground">
                      {person.sample_count} {person.sample_count === 1 ? "foto" : "fotos"}
                    </span>
                    {!isWithdrawn ? (
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        onClick={() => setTarget(person)}
                        className="border-destructive/30 text-destructive hover:bg-destructive-soft"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                        Revocar
                      </Button>
                    ) : (
                      <span className="inline-flex items-center gap-1 text-xs text-muted-foreground">
                        <ShieldOff className="h-3.5 w-3.5" />
                        Purgada
                      </span>
                    )}
                  </CardContent>
                </Card>
              );
            })}
          </div>

          {identities.length > visibleLimit && (
            <div className="flex justify-center pt-1">
              <Button
                type="button"
                variant="outline"
                onClick={() => setVisibleLimit((prev) => prev + IDENTITIES_PAGE_SIZE)}
              >
                Mostrar {Math.min(IDENTITIES_PAGE_SIZE, identities.length - visibleLimit)} más (mostrando{" "}
                {visibleLimit} de {identities.length})
              </Button>
            </div>
          )}
        </>
      )}

      <Dialog open={target !== null} onOpenChange={(open) => !open && !revoking && setTarget(null)}>
        <DialogContent>
          <DialogTitle>Revocar consentimiento y eliminar fotos</DialogTitle>
          <DialogDescription>
            Se marcará la identidad{" "}
            <strong className="text-foreground">
              {target?.display_label || target?.source_subject_id}
            </strong>{" "}
            como revocada y se eliminarán sus {target?.sample_count ?? 0} muestras faciales de los índices vectoriales. Después de esto volverá a figurar como desconocida.
          </DialogDescription>
          <div className="mt-5 flex flex-wrap justify-end gap-2">
            <DialogClose asChild>
              <Button type="button" variant="outline" disabled={revoking}>
                Cancelar
              </Button>
            </DialogClose>
            <Button type="button" variant="destructive" onClick={() => void handleConfirmWithdraw()} disabled={revoking}>
              {revoking ? "Eliminando..." : "Sí, revocar y eliminar"}
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      <Dialog open={confirmPurgeAll} onOpenChange={(open) => !open && !purgingAll && setConfirmPurgeAll(false)}>
        <DialogContent>
          <DialogTitle>Eliminar todas las fotos registradas</DialogTitle>
          <DialogDescription>
            Se borrarán las{" "}
            <strong className="text-foreground">{totalSamples}</strong> fotos/embeddings almacenados en el índice vectorial. Esta acción no se puede deshacer.
          </DialogDescription>

          <label
            htmlFor="purge-also-identities"
            className="mt-4 flex cursor-pointer items-start gap-3 rounded-lg border border-border bg-muted/40 p-3 text-sm"
          >
            <Checkbox
              id="purge-also-identities"
              checked={alsoDeleteIdentities}
              onCheckedChange={(checked) => setAlsoDeleteIdentities(checked === true)}
              disabled={purgingAll}
              className="mt-0.5"
            />
            <span className="leading-snug text-foreground">
              Eliminar también las {identities.length} identidades y registros de consentimiento (dejar la base en 0).
            </span>
          </label>

          <div className="mt-5 flex flex-wrap justify-end gap-2">
            <DialogClose asChild>
              <Button type="button" variant="outline" disabled={purgingAll}>
                Cancelar
              </Button>
            </DialogClose>
            <Button type="button" variant="destructive" onClick={() => void handleConfirmPurgeAll()} disabled={purgingAll}>
              {purgingAll ? "Eliminando..." : "Sí, eliminar todo"}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
