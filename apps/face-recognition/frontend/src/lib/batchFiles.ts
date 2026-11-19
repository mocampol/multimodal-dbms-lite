import {
  BatchEnrollItem,
  BatchGroupingStrategy,
  BatchGroupPreview,
  FaceApiError,
  IdentityRecord,
} from "@/types/face";

const SUPPORTED_EXTENSIONS = new Set([".jpg", ".jpeg", ".png", ".webp"]);
const SUPPORTED_MIME_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);

export interface RawBatchEntry {
  file: File;
  relativePath: string;
}

export interface FilterBatchResult {
  valid: RawBatchEntry[];
  skippedCount: number;
}

function getExtension(name: string): string {
  const idx = name.lastIndexOf(".");
  return idx >= 0 ? name.slice(idx).toLowerCase() : "";
}

function getBasename(path: string): string {
  const parts = path.split("/").filter(Boolean);
  return parts.length > 0 ? parts[parts.length - 1] : path;
}

export function isSupportedImageEntry(entry: RawBatchEntry): boolean {
  const basename = getBasename(entry.relativePath || entry.file.name);
  if (!basename || basename.startsWith(".") || basename === "Thumbs.db") {
    return false;
  }
  if (entry.file.size === 0) {
    return false;
  }
  const ext = getExtension(basename);
  if (SUPPORTED_EXTENSIONS.has(ext)) {
    return true;
  }
  return ext === "" && SUPPORTED_MIME_TYPES.has(entry.file.type);
}

export function filterValidImageEntries(entries: RawBatchEntry[]): FilterBatchResult {
  const valid: RawBatchEntry[] = [];
  let skippedCount = 0;

  for (const entry of entries) {
    if (isSupportedImageEntry(entry)) {
      valid.push(entry);
    } else {
      skippedCount += 1;
    }
  }

  valid.sort((a, b) => a.relativePath.localeCompare(b.relativePath, "es"));
  return { valid, skippedCount };
}

export function entriesFromFileList(fileList: FileList | File[]): RawBatchEntry[] {
  const list = Array.from(fileList);
  return list.map((file) => {
    const rel = (file as File & { webkitRelativePath?: string }).webkitRelativePath;
    return {
      file,
      relativePath: rel && rel.length > 0 ? rel : file.name,
    };
  });
}

type FileSystemEntryLike = {
  isFile: boolean;
  isDirectory: boolean;
  name: string;
  fullPath: string;
  file?: (success: (file: File) => void, error?: (err: unknown) => void) => void;
  createReader?: () => {
    readEntries: (
      success: (entries: FileSystemEntryLike[]) => void,
      error?: (err: unknown) => void,
    ) => void;
  };
};

async function readEntryRecursively(entry: FileSystemEntryLike): Promise<RawBatchEntry[]> {
  if (entry.isFile && entry.file) {
    return new Promise((resolve) => {
      entry.file!(
        (file) => {
          const cleanPath = entry.fullPath.replace(/^\/+/, "") || file.name;
          resolve([{ file, relativePath: cleanPath }]);
        },
        () => resolve([]),
      );
    });
  }

  if (entry.isDirectory && entry.createReader) {
    const reader = entry.createReader();
    const collected: FileSystemEntryLike[] = [];

    // Chromium readEntries devuelve lotes parciales; iterar hasta recibir [].
    while (true) {
      const batch = await new Promise<FileSystemEntryLike[]>((resolve) => {
        reader.readEntries(
          (items) => resolve(items),
          () => resolve([]),
        );
      });
      if (batch.length === 0) break;
      collected.push(...batch);
    }

    const nested = await Promise.all(collected.map((child) => readEntryRecursively(child)));
    return nested.flat();
  }

  return [];
}

export async function extractEntriesFromDataTransfer(dataTransfer: DataTransfer): Promise<RawBatchEntry[]> {
  const items = Array.from(dataTransfer.items || []);
  const entries = items
    .map((item) =>
      (item as unknown as { webkitGetAsEntry?: () => FileSystemEntryLike | null }).webkitGetAsEntry?.() ?? null,
    )
    .filter((entry): entry is FileSystemEntryLike => entry !== null);

  if (entries.length > 0) {
    const results = await Promise.all(entries.map((entry) => readEntryRecursively(entry)));
    return results.flat();
  }

  return entriesFromFileList(dataTransfer.files);
}

type DirectoryHandleLike = {
  kind: "directory";
  name: string;
  values: () => AsyncIterable<DirectoryHandleLike | FileHandleLike>;
};

type FileHandleLike = {
  kind: "file";
  name: string;
  getFile: () => Promise<File>;
};

export async function extractEntriesFromDirectoryHandle(
  dirHandle: DirectoryHandleLike,
  prefix = dirHandle.name,
): Promise<{ entries: RawBatchEntry[]; skippedByName: number }> {
  const entries: RawBatchEntry[] = [];
  let skippedByName = 0;

  for await (const handle of dirHandle.values()) {
    const relPath = prefix ? `${prefix}/${handle.name}` : handle.name;
    if (handle.kind === "directory") {
      if (handle.name.startsWith(".")) continue;
      const nested = await extractEntriesFromDirectoryHandle(handle, relPath);
      entries.push(...nested.entries);
      skippedByName += nested.skippedByName;
      continue;
    }

    if (handle.name.startsWith(".") || handle.name === "Thumbs.db") {
      skippedByName += 1;
      continue;
    }
    const ext = getExtension(handle.name);
    if (!SUPPORTED_EXTENSIONS.has(ext)) {
      skippedByName += 1;
      continue;
    }

    try {
      const file = await handle.getFile();
      entries.push({ file, relativePath: relPath });
    } catch {
      skippedByName += 1;
    }
  }

  return { entries, skippedByName };
}

export function cleanLabelFromFilename(filename: string): string {
  const base = getBasename(filename);
  const dotIdx = base.lastIndexOf(".");
  const stem = dotIdx > 0 ? base.slice(0, dotIdx) : base;
  const cleaned = stem.replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim();
  return cleaned || stem || "Sin nombre";
}

export function getParentFolderName(relativePath: string): string | null {
  const parts = relativePath.split("/").filter(Boolean);
  if (parts.length < 2) return null;
  return parts[parts.length - 2] || null;
}

export function inferGroupingStrategy(entries: RawBatchEntry[]): {
  strategy: BatchGroupingStrategy;
  suggestedSingleLabel: string;
} {
  const folders = new Set<string>();
  let rootFolder: string | null = null;

  for (const entry of entries) {
    const parts = entry.relativePath.split("/").filter(Boolean);
    if (parts.length >= 2) {
      rootFolder = rootFolder ?? parts[0];
      folders.add(parts[parts.length - 2]);
    }
  }

  if (folders.size >= 2) {
    return {
      strategy: "subfolder",
      suggestedSingleLabel: rootFolder || "",
    };
  }
  if (folders.size === 1) {
    const onlyFolder = Array.from(folders)[0];
    return {
      strategy: "single",
      suggestedSingleLabel: onlyFolder || "",
    };
  }
  return {
    strategy: "per_file",
    suggestedSingleLabel: "",
  };
}

export function resolveItemGrouping(
  relativePath: string,
  strategy: BatchGroupingStrategy,
  singleTargetLabel: string,
  singleMode: "new" | "existing",
  singleExistingId: string,
  activeIdentities: IdentityRecord[],
): { groupKey: string; targetLabel: string; forcedExistingId?: number } {
  if (strategy === "single") {
    if (singleMode === "existing" && singleExistingId) {
      const idNum = Number(singleExistingId);
      const found = activeIdentities.find((p) => p.identity_id === idNum);
      return {
        groupKey: `existing:${idNum}`,
        targetLabel: found?.display_label || found?.source_subject_id || `Identidad #${idNum}`,
        forcedExistingId: idNum,
      };
    }
    const label = singleTargetLabel.trim() || "Persona del lote";
    return {
      groupKey: `single:${label.toLowerCase()}`,
      targetLabel: label,
    };
  }

  if (strategy === "subfolder") {
    const folder = getParentFolderName(relativePath) || cleanLabelFromFilename(relativePath);
    return {
      groupKey: `folder:${folder.toLowerCase()}`,
      targetLabel: folder,
    };
  }

  const fileLabel = cleanLabelFromFilename(relativePath);
  return {
    groupKey: `file:${fileLabel.toLowerCase()}`,
    targetLabel: fileLabel,
  };
}

export function buildBatchGroups(
  items: BatchEnrollItem[],
  activeIdentities: IdentityRecord[],
): BatchGroupPreview[] {
  const byKey = new Map<string, BatchGroupPreview>();
  const activeByLowerLabel = new Map<string, IdentityRecord>();

  for (const person of activeIdentities) {
    if (person.consent_status !== "granted") continue;
    const key = (person.display_label || person.source_subject_id).trim().toLowerCase();
    if (key && !activeByLowerLabel.has(key)) {
      activeByLowerLabel.set(key, person);
    }
  }

  for (const item of items) {
    const existing = byKey.get(item.groupKey);
    if (existing) {
      existing.fileCount += 1;
      continue;
    }

    const explicitIdMatch = item.groupKey.startsWith("existing:")
      ? Number(item.groupKey.slice("existing:".length))
      : null;
    const matchedByName = activeByLowerLabel.get(item.targetLabel.trim().toLowerCase());

    byKey.set(item.groupKey, {
      groupKey: item.groupKey,
      displayLabel: item.targetLabel,
      fileCount: 1,
      existingIdentityId: explicitIdMatch ?? matchedByName?.identity_id ?? null,
    });
  }

  return Array.from(byKey.values());
}

export function mapBatchError(err: unknown): { code: string; message: string } {
  if (err instanceof FaceApiError) {
    if (err.code === "no_face_detected") {
      return {
        code: err.code,
        message: "No se detectó ningún rostro en la imagen.",
      };
    }
    if (err.code === "multiple_faces_detected") {
      return {
        code: err.code,
        message: "Se detectaron múltiples rostros (se requiere exactamente uno).",
      };
    }
    return { code: err.code, message: err.message };
  }
  if (err instanceof DOMException && err.name === "AbortError") {
    return { code: "aborted", message: "Cancelado por el usuario." };
  }
  if (err instanceof Error) {
    return { code: "unexpected_error", message: err.message };
  }
  return { code: "unexpected_error", message: "Error inesperado al procesar la foto." };
}

function escapeCsvCell(value: string | number | null | undefined): string {
  const str = value === null || value === undefined ? "" : String(value);
  if (/[",\n\r]/.test(str)) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

export function buildCsvReport(items: BatchEnrollItem[]): string {
  const headers = [
    "archivo",
    "estado",
    "identidad_objetivo",
    "identity_id",
    "sample_id",
    "confianza_deteccion",
    "similar_encontrado",
    "similar_score",
    "codigo_error",
    "mensaje_error",
    "tiempo_ms",
  ];

  const rows = items.map((item) =>
    [
      item.relativePath,
      item.status,
      item.targetLabel,
      item.identityId ?? "",
      item.sampleId ?? "",
      item.detectionConfidence !== undefined ? item.detectionConfidence.toFixed(4) : "",
      item.similarLabel ?? "",
      item.similarScore !== undefined ? item.similarScore.toFixed(4) : "",
      item.errorCode ?? "",
      item.errorMessage ?? "",
      item.elapsedMs !== undefined ? item.elapsedMs.toFixed(1) : "",
    ]
      .map(escapeCsvCell)
      .join(","),
  );

  return [headers.join(","), ...rows].join("\n");
}

export function buildJsonReport(items: BatchEnrollItem[]): string {
  const payload = items.map((item) => ({
    archivo: item.relativePath,
    estado: item.status,
    identidad_objetivo: item.targetLabel,
    identity_id: item.identityId ?? null,
    sample_id: item.sampleId ?? null,
    confianza_deteccion: item.detectionConfidence ?? null,
    similar_encontrado: item.similarLabel ?? null,
    similar_identity_id: item.similarIdentityId ?? null,
    similar_score: item.similarScore ?? null,
    codigo_error: item.errorCode ?? null,
    mensaje_error: item.errorMessage ?? null,
    tiempo_ms: item.elapsedMs ?? null,
  }));
  return JSON.stringify(payload, null, 2);
}

export function triggerTextDownload(content: string, filename: string, mimeType: string): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
