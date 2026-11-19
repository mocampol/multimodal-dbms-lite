import {
  CreateIdentityPayload,
  CreateIdentityResponse,
  EnrollResponse,
  FaceApiError,
  FacePolicy,
  HealthResponse,
  IdentitiesListResponse,
  PurgeAllResponse,
  RecognizeParams,
  RecognizeResponse,
  WithdrawResponse,
} from "@/types/face";

const BASE_URL = (import.meta.env.VITE_FACE_API_URL as string | undefined)?.replace(/\/$/, "") || "/face-api";

async function parseResponse<T>(res: Response): Promise<T> {
  const text = await res.text();
  let data: Record<string, unknown> | null = null;
  try {
    data = text ? (JSON.parse(text) as Record<string, unknown>) : null;
  } catch {
    data = null;
  }

  if (!res.ok) {
    const message =
      (data?.error as string | undefined) ||
      (data?.detail as string | undefined) ||
      `Error HTTP ${res.status}`;
    const code = (data?.code as string | undefined) || "request_failed";
    throw new FaceApiError(message, res.status, code);
  }
  return data as T;
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${BASE_URL}/health`);
  return parseResponse<HealthResponse>(res);
}

export async function listIdentities(): Promise<IdentitiesListResponse> {
  const res = await fetch(`${BASE_URL}/identities`);
  return parseResponse<IdentitiesListResponse>(res);
}

export async function createIdentity(
  payload: CreateIdentityPayload,
  signal?: AbortSignal,
): Promise<CreateIdentityResponse> {
  const res = await fetch(`${BASE_URL}/identities`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
  return parseResponse<CreateIdentityResponse>(res);
}

export async function withdrawIdentity(identityId: number): Promise<WithdrawResponse> {
  const res = await fetch(`${BASE_URL}/identities/${identityId}`, {
    method: "DELETE",
  });
  return parseResponse<WithdrawResponse>(res);
}

export async function purgeAllFaces(samplesOnly = false): Promise<PurgeAllResponse> {
  const res = await fetch(`${BASE_URL}/identities?samples_only=${samplesOnly ? "true" : "false"}`, {
    method: "DELETE",
  });
  return parseResponse<PurgeAllResponse>(res);
}

export async function enrollFace(
  identityId: number,
  fileBlob: Blob,
  filename = "capture.jpg",
  signal?: AbortSignal,
  facePolicy: FacePolicy = "strict_single",
): Promise<EnrollResponse> {
  const form = new FormData();
  form.append("file", fileBlob, filename);
  form.append("identity_id", String(identityId));
  form.append("split", "enroll");
  form.append("face_policy", facePolicy);

  const res = await fetch(`${BASE_URL}/enroll`, {
    method: "POST",
    body: form,
    signal,
  });
  return parseResponse<EnrollResponse>(res);
}

export async function recognizeFace(
  fileBlob: Blob,
  params: RecognizeParams,
  filename = "probe.jpg",
  signal?: AbortSignal,
): Promise<RecognizeResponse> {
  const form = new FormData();
  form.append("file", fileBlob, filename);
  form.append("k", String(params.k));
  form.append("threshold", String(params.threshold));
  form.append("index_type", params.index_type);
  form.append("metric", params.metric);
  form.append("ef_search", String(params.ef_search));
  form.append("n_probe", String(params.n_probe));
  form.append("aggregation", params.aggregation);
  form.append("face_policy", "largest");

  const res = await fetch(`${BASE_URL}/recognize`, {
    method: "POST",
    body: form,
    signal,
  });
  return parseResponse<RecognizeResponse>(res);
}
