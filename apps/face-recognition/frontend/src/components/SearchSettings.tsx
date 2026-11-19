import * as React from "react";
import { ChevronDown, SlidersHorizontal } from "lucide-react";
import { RecognizeParams } from "@/types/face";
import { cn } from "@/lib/utils";

interface SearchSettingsProps {
  params: RecognizeParams;
  onChange: (next: RecognizeParams) => void;
}

export function SearchSettings({ params, onChange }: SearchSettingsProps) {
  const [open, setOpen] = React.useState(false);

  return (
    <div className="rounded-xl border border-border bg-card">
      <button
        type="button"
        onClick={() => setOpen((prev) => !prev)}
        aria-expanded={open}
        className="flex min-h-11 w-full items-center justify-between gap-3 px-4 py-3 text-left text-sm font-medium text-foreground hover:bg-muted/40 rounded-xl focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        <span className="flex min-w-0 items-center gap-2.5">
          <SlidersHorizontal className="h-4 w-4 shrink-0 text-primary" />
          <span className="flex min-w-0 flex-col">
            <span className="truncate font-medium">Parámetros de búsqueda</span>
            <span className="tabular-metrics truncate text-xs font-normal text-muted-foreground">
              {params.index_type.toUpperCase()} · k={params.k} · umbral={params.threshold.toFixed(2)}
            </span>
          </span>
        </span>
        <ChevronDown className={cn("h-4 w-4 shrink-0 text-muted-foreground transition-transform", open && "rotate-180")} />
      </button>

      {open && (
        <div className="grid grid-cols-1 gap-3 border-t border-border p-4 sm:grid-cols-2">
          <div className="flex flex-col gap-1.5">
            <label htmlFor="param-index" className="text-xs font-medium text-muted-foreground">
              Índice vectorial
            </label>
            <select
              id="param-index"
              value={params.index_type}
              onChange={(e) => onChange({ ...params, index_type: e.target.value as RecognizeParams["index_type"] })}
              className="min-h-11 rounded-lg border border-border bg-card px-3 text-base text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <option value="hnsw">HNSW (rápido / grafo)</option>
              <option value="ivf">IVF (listas invertidas)</option>
            </select>
          </div>

          <div className="flex flex-col gap-1.5">
            <label htmlFor="param-metric" className="text-xs font-medium text-muted-foreground">
              Métrica de similitud
            </label>
            <select
              id="param-metric"
              value={params.metric}
              onChange={(e) => onChange({ ...params, metric: e.target.value as RecognizeParams["metric"] })}
              className="min-h-11 rounded-lg border border-border bg-card px-3 text-base text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <option value="cosine">Coseno</option>
              <option value="euclidean">Euclidiana</option>
            </select>
          </div>

          <div className="flex flex-col gap-1.5 sm:col-span-2">
            <div className="flex items-center justify-between text-xs">
              <label htmlFor="param-threshold" className="font-medium text-muted-foreground">
                Umbral mínimo para aceptar match
              </label>
              <span className="tabular-metrics font-semibold text-foreground">{params.threshold.toFixed(2)}</span>
            </div>
            <input
              id="param-threshold"
              type="range"
              min={0.1}
              max={0.95}
              step={0.01}
              value={params.threshold}
              onChange={(e) => onChange({ ...params, threshold: Number(e.target.value) })}
              className="h-9 w-full accent-primary cursor-pointer"
            />
          </div>

          <div className="flex flex-col gap-1.5">
            <label htmlFor="param-k" className="text-xs font-medium text-muted-foreground">
              Vecinos k
            </label>
            <input
              id="param-k"
              type="number"
              min={1}
              max={50}
              value={params.k}
              onChange={(e) => onChange({ ...params, k: Math.max(1, Number(e.target.value) || 1) })}
              className="tabular-metrics min-h-11 rounded-lg border border-border bg-card px-3 text-base text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            />
          </div>

          {params.index_type === "hnsw" ? (
            <div className="flex flex-col gap-1.5">
              <label htmlFor="param-ef" className="text-xs font-medium text-muted-foreground">
                HNSW ef_search
              </label>
              <input
                id="param-ef"
                type="number"
                min={10}
                max={500}
                value={params.ef_search}
                onChange={(e) => onChange({ ...params, ef_search: Math.max(10, Number(e.target.value) || 100) })}
                className="tabular-metrics min-h-11 rounded-lg border border-border bg-card px-3 text-base text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              />
            </div>
          ) : (
            <div className="flex flex-col gap-1.5">
              <label htmlFor="param-probe" className="text-xs font-medium text-muted-foreground">
                IVF n_probe
              </label>
              <input
                id="param-probe"
                type="number"
                min={1}
                max={64}
                value={params.n_probe}
                onChange={(e) => onChange({ ...params, n_probe: Math.max(1, Number(e.target.value) || 4) })}
                className="tabular-metrics min-h-11 rounded-lg border border-border bg-card px-3 text-base text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
