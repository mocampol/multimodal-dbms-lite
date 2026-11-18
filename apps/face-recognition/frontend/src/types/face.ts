export type IndexType = "hnsw" | "ivf";
export type MetricType = "cosine" | "euclidean";
export type FacePolicy = "strict_single" | "largest";
export type AggregationType = "max" | "mean_top";

export interface HealthResponse {
  status: "ok" | "degraded";
  service: string;
  model: string;
  embedding_dim: number;
  defaults: {
    index_type: IndexType;
    metric: MetricType;
    k: number;
    threshold: number;
  };
  dbms: Record<string, unknown>;
}

export interface IdentityRecord {
  identity_id: number;
  source: "participant" | "public_lfw" | "public_celeba";
  source_subject_id: string;
  display_label: string | null;
  consent_status: "granted" | "pending" | "withdrawn" | "dataset_only";
  sample_count: number;
}

export interface IdentitiesListResponse {
  identities: IdentityRecord[];
  count: number;
}

export interface CreateIdentityPayload {
  source: "participant";
  display_label: string;
  consent_granted: boolean;
  purpose: string;
  notice_version: string;
}

export interface CreateIdentityResponse {
  identity_id: number;
  source: string;
  source_subject_id: string;
  display_label: string | null;
  consent_status: string;
}

export interface WithdrawResponse {
  identity_id: number;
  consent_status: "withdrawn";
  deleted_samples: number;
  withdrawn_at: string;
}

export interface PurgeAllResponse {
  samples_only: boolean;
  deleted_samples: number;
  deleted_identities: number;
  deleted_consents: number;
}

export interface EnrollResponse {
  sample_id: number;
  identity_id: number;
  display_label: string | null;
  source: string;
  split: string;
  relative_path: string;
  faces_detected: number;
  detection_confidence: number;
  bbox: [number, number, number, number];
  model_name: string;
  embedding_dim: number;
  timings_ms: {
    extraction_ms: number;
    insert_ms: number;
  };
}

export interface RecognizeCandidate {
  identity_id: number;
  display_label: string | null;
  source_subject_id: string;
  score: number;
  best_sample_score: number;
  best_distance: number;
  matched_samples: number;
  sample_id: number;
}

export interface RecognizeResponse {
  registered: boolean;
  matched: boolean;
  status: "registered" | "unknown";
  label: string;
  identity_id: number | null;
  score: number;
  threshold: number;
  faces_detected: number;
  detection_confidence: number;
  bbox: [number, number, number, number];
  model_name: string;
  embedding_dim: number;
  index_type: IndexType;
  used_index: boolean;
  metric: string;
  k: number;
  aggregation: AggregationType;
  candidates: RecognizeCandidate[];
  timings_ms: {
    extraction_ms: number;
    dbms_search_ms: number;
    search_roundtrip_ms: number;
    total_ms: number;
  };
}

export interface RecognizeParams {
  k: number;
  threshold: number;
  index_type: IndexType;
  metric: MetricType;
  ef_search: number;
  n_probe: number;
  aggregation: AggregationType;
}

export class FaceApiError extends Error {
  readonly status: number;
  readonly code: string;

  constructor(message: string, status: number, code: string) {
    super(message);
    this.name = "FaceApiError";
    this.status = status;
    this.code = code;
  }
}

export type BatchGroupingStrategy = "subfolder" | "single" | "per_file";
export type BatchExistingMatchPolicy = "reuse" | "create_new";
export type BatchItemStatus =
  | "pending"
  | "checking"
  | "enrolling"
  | "registered"
  | "similar_found"
  | "error";

export interface BatchEnrollItem {
  id: string;
  file: File;
  relativePath: string;
  previewUrl?: string;
  groupKey: string;
  targetLabel: string;
  status: BatchItemStatus;
  identityId?: number;
  sampleId?: number;
  detectionConfidence?: number;
  elapsedMs?: number;
  similarIdentityId?: number | null;
  similarLabel?: string;
  similarScore?: number;
  errorCode?: string;
  errorMessage?: string;
}

export interface BatchGroupPreview {
  groupKey: string;
  displayLabel: string;
  fileCount: number;
  existingIdentityId: number | null;
}

export interface BatchSummaryStats {
  total: number;
  processed: number;
  registered: number;
  similar: number;
  failed: number;
}
