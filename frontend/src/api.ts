export interface Capabilities {
  sessions?: { supported: boolean; resumable?: boolean; activity?: boolean; exports?: boolean; confirm_close?: boolean };
  batch?: { supported: boolean; unified?: boolean };
  generation?: {
    objectives: string[];
    durations: number[];
    clip_counts: number[];
  };
  fit?: { supported_sources: string[]; default_seed: number; preserve_fits_on_recording_change?: boolean };
  upload?: { max_bytes: number; formats: string[] };
  settings: {
    stimulus_families: string[];
    datasets: string[];
    models: string[];
    layers: number[];
    bin_width_ms: number;
    receptive_field_ms: number;
  };
  gpu: {
    visible_count: number;
    available_count: number;
    observed_at: string;
    status: "ready" | "no_gpu" | "unavailable";
  };
}

export interface Configuration {
  stimulus_family: string;
  dataset_name: string;
  model_name: string;
  layer_id: number;
  bin_width_ms: number;
  receptive_field_ms: number;
  gpu_count: number;
}

export interface Experiment {
  id: string;
  created_at: string;
  configuration: Configuration;
  recording: Recording | null;
  recording_revision: number;
  session_id: string;
}

export interface Recording {
  id: string;
  display_name: string;
  dataset_name: string;
  source: "existing" | "upload";
  stimulus_families: string[];
}

export interface RecordingCatalog {
  recordings: Recording[];
  unavailable_count: number;
}

export interface UploadResult {
  validation_status: "valid";
  experiment: Experiment;
}

export interface Fit {
  configuration: Configuration;
  id: string;
  job_id: string;
  experiment_id: string;
  recording: Recording;
  recording_revision: number;
  seed: number;
  state:
    | "queued"
    | "running"
    | "succeeded"
    | "failed"
    | "interrupted"
    | "cancelling"
    | "cancelled";
  stage: string;
  error: string | null;
  artifacts_ready: boolean;
  correlations: { channel_id: string; correlation: number | null }[];
  best_channel: { channel_id: string; correlation: number | null } | null;
}

export interface GenerationSubmission {
  request_id: string;
  channel_id: string;
  objective: "stretch" | "one-hot";
  duration_seconds: number;
  clip_count: number;
  seed: number;
}

export interface Generation extends Omit<GenerationSubmission, "request_id"> {
  id: string;
  job_id: string;
  fit_id: string;
  configuration: Configuration;
  state:
    | "queued"
    | "running"
    | "succeeded"
    | "failed"
    | "interrupted"
    | "cancelling"
    | "cancelled";
  stage: string;
  artifacts_ready: boolean;
  error: string | null;
  used_gpu_count: number;
  unused_gpu_count: number;
  clips: { filename: string; frames: number; sample_rate: number }[];
}

export interface BatchSubmission {
  request_id: string;
  gpu_count: number;
  targets: { channel_id: string; objective: string }[];
  duration_seconds: number;
  clip_count: number;
  seed: number;
}

export interface Batch {
  id: string;
  job_id: string;
  fit_id: string;
  gpu_count: number;
  used_gpu_count: number;
  unused_gpu_count: number;
  state:
    | "queued"
    | "running"
    | "succeeded"
    | "partial"
    | "failed"
    | "interrupted"
    | "cancelling"
    | "cancelled";
  error: string | null;
  artifacts_ready: boolean;
  targets: Generation[];
}

export type Job = Fit | Generation | Batch;
export interface ApplicationSession {
  id: string;
  created_at: string;
  status: "open" | "closing" | "closed";
  closed_at: string | null;
  timezone: string;
}
export interface ResumableSession {
  session: ApplicationSession;
  experiment: Experiment;
  active_work: boolean;
}
export interface SessionDetail {
  session: ApplicationSession;
  experiments: Experiment[];
  jobs: {
    id: string;
    job_id: string;
    kind: "fit" | "generation" | "batch";
    state: string;
    created_at: string;
    artifacts_ready: boolean;
    retry_of_job_id: string | null;
    retryable: boolean;
  }[];
}
export interface RecoveredWork {
  fits: Fit[];
  generations: Generation[];
  batches: Batch[];
}

interface FitSubmission {
  request_id: string;
  recording_id: string;
  recording_revision: number;
  seed: number;
}

declare global {
  interface Window {
    AUDITORY_CONFIG?: { apiBaseUrl?: string; apiBaseUrls?: string[]; connectionCommand?: string };
  }
}

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

export async function discoverApiBaseUrl(): Promise<string> {
  const configuration = window.AUDITORY_CONFIG;
  // Preserve fixed-URL/legacy sign-in behavior; an explicit check always probes.
  if (configuration?.apiBaseUrl?.trim()) return configuration.apiBaseUrl.replace(/\/+$/, "");
  if (!configuration?.apiBaseUrls?.length) return "";
  return checkBackendConnection();
}

export async function checkBackendConnection(): Promise<string> {
  const configuration = window.AUDITORY_CONFIG;
  const candidates = configuration?.apiBaseUrl?.trim()
    ? [configuration.apiBaseUrl]
    : [...new Set(configuration?.apiBaseUrls?.length ? configuration.apiBaseUrls : [""])];
  for (const candidate of candidates) {
    const base = candidate.trim().replace(/\/+$/, "");
    if (await isBackendReachable(base)) return base;
  }
  throw new Error("No configured backend is reachable. Run your connection script and allow browser local-network access if prompted.");
}

async function isBackendReachable(base: string): Promise<boolean> {
  const abort = new AbortController();
  const timeout = window.setTimeout(() => abort.abort(), 5000);
  try {
    const response = await fetch(`${base}/healthz?identify=true`, {
      signal: abort.signal, cache: "no-store", credentials: "omit", redirect: "error",
    });
    if (!response.ok) return false;
    const health = await response.json();
    return health?.status === "ok" && health.service === "auditory-cortex-control";
  } catch {
    return false;
  } finally {
    window.clearTimeout(timeout);
  }
}

export function makeApi(
  username: string,
  password: string,
  baseUrl = window.AUDITORY_CONFIG?.apiBaseUrl ?? "",
) {
  // Credentials stay in memory, never in browser storage or public config.
  const encoded = btoa(
    Array.from(new TextEncoder().encode(`${username}:${password}`), (byte) =>
      String.fromCharCode(byte),
    ).join(""),
  );
  const base = baseUrl.replace(/\/+$/, "");
  async function request<T>(
    path: string,
    body?:
      | Configuration
      | { recording_id: string }
      | File
      | FitSubmission
      | GenerationSubmission
      | BatchSubmission
      | { request_id: string }
      | Record<string, never>,
  ): Promise<T> {
    const file = body instanceof File ? body : null;
    const response = await fetch(`${base}/api/v1${path}`, {
      method: body ? "POST" : "GET",
      headers: {
        Authorization: `Basic ${encoded}`,
        ...(file
          ? {
              "Content-Type": "application/octet-stream",
              "X-Recording-Filename": encodeURIComponent(file.name),
            }
          : body
            ? { "Content-Type": "application/json" }
            : {}),
      },
      ...(body ? { body: file ?? JSON.stringify(body) } : {}),
      cache: "no-store",
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      const detail = data.detail;
      const message = typeof detail === "string" ? detail : detail?.message;
      throw new ApiError(
        message ?? `Request failed (${response.status}). Check your settings.`,
        response.status,
      );
    }
    return response.json() as Promise<T>;
  }
  async function downloadResponse(path: string): Promise<Response> {
    const response = await fetch(`${base}/api/v1${path}`, {
      headers: { Authorization: `Basic ${encoded}` },
      cache: "no-store",
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new ApiError(
        typeof data.detail === "string"
          ? data.detail
          : "Could not download samples.",
        response.status,
      );
    }
    return response;
  }
  async function download(path: string): Promise<Blob> {
    return (await downloadResponse(path)).blob();
  }
  async function exportSamples(sessionId: string, jobId?: string) {
    const scope = jobId ? `/runs/${encodeURIComponent(jobId)}` : "";
    const response = await downloadResponse(`/sessions/${encodeURIComponent(sessionId)}${scope}/download`);
    const filename = response.headers.get("Content-Disposition")?.match(/filename="?([A-Za-z0-9_-]+\.zip)"?/i)?.[1];
    return { blob: await response.blob(), filename: filename ?? "samples.zip" };
  }
  return {
    exportSamples,
    checkConnection: () => isBackendReachable(base),
    resumableSessions: () => request<ResumableSession[]>("/sessions/resumable"),
    activity: (id: string) => request<{ at: string; message: string }[]>(`/sessions/${encodeURIComponent(id)}/activity`),
    sessions: () => request<ApplicationSession[]>("/sessions"),
    session: (id: string) =>
      request<SessionDetail>(`/sessions/${encodeURIComponent(id)}`),
    closeSession: (id: string, requireIdle = false) =>
      request<ApplicationSession>(
        `/sessions/${encodeURIComponent(id)}/close${requireIdle ? "?require_idle=true" : ""}`,
        {},
      ),
    job: (id: string) => request<Job>(`/jobs/${encodeURIComponent(id)}`),
    cancelJob: (id: string) =>
      request<Job>(`/jobs/${encodeURIComponent(id)}/cancel`, {}),
    retryJob: (id: string, requestId: string) =>
      request<Job>(`/jobs/${encodeURIComponent(id)}/retry`, {
        request_id: requestId,
      }),
    recoverExperiment: (id: string) =>
      request<RecoveredWork>(`/experiments/${encodeURIComponent(id)}/work`),
    generateSamples: (fitId: string, submission: BatchSubmission) =>
      request<Batch>(`/fits/${encodeURIComponent(fitId)}/samples`, submission),
    startBatch: (fitId: string, submission: BatchSubmission) =>
      request<Batch>(`/fits/${encodeURIComponent(fitId)}/batches`, submission),
    batch: (id: string) => request<Batch>(`/batches/${encodeURIComponent(id)}`),
    downloadBatch: (id: string) =>
      download(`/batches/${encodeURIComponent(id)}/download`),
    startGeneration: (fitId: string, submission: GenerationSubmission) =>
      request<Generation>(
        `/fits/${encodeURIComponent(fitId)}/generations`,
        submission,
      ),
    generation: (id: string) =>
      request<Generation>(`/generations/${encodeURIComponent(id)}`),
    downloadGeneration: (id: string) =>
      download(`/generations/${encodeURIComponent(id)}/download`),
    capabilities: () => request<Capabilities>("/capabilities"),
    createExperiment: (configuration: Configuration, sessionId?: string | null, useVisibleCapacity = false) => {
      const params = new URLSearchParams();
      if (useVisibleCapacity) params.set("use_visible_capacity", "true");
      if (sessionId) {
        params.set("session_id", sessionId);
      } else {
        const zone = Intl.DateTimeFormat().resolvedOptions().timeZone;
        if (zone) params.set("session_timezone", zone);
      }
      return request<Experiment>(`/experiments?${params}`, configuration);
    },
    recordings: (experimentId: string) =>
      request<RecordingCatalog>(
        `/experiments/${encodeURIComponent(experimentId)}/recordings`,
      ),
    selectRecording: (experimentId: string, recordingId: string) =>
      request<Experiment>(
        `/experiments/${encodeURIComponent(experimentId)}/recording-selection?preserve_fits=true`,
        { recording_id: recordingId },
      ),
    uploadRecording: (experimentId: string, file: File) =>
      request<UploadResult>(
        `/experiments/${encodeURIComponent(experimentId)}/recording-upload?preserve_fits=true`,
        file,
      ),
    startFit: (experimentId: string, submission: FitSubmission) =>
      request<Fit>(
        `/experiments/${encodeURIComponent(experimentId)}/fits`,
        submission,
      ),
    fit: (fitId: string) => request<Fit>(`/fits/${encodeURIComponent(fitId)}`),
  };
}
