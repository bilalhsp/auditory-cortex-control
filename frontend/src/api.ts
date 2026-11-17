export interface Capabilities {
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

declare global {
  interface Window {
    AUDITORY_CONFIG?: { apiBaseUrl?: string };
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

export function makeApi(username: string, password: string) {
  // Credentials stay in memory, never in browser storage or public config.
  const encoded = btoa(
    Array.from(new TextEncoder().encode(`${username}:${password}`), (byte) =>
      String.fromCharCode(byte),
    ).join(""),
  );
  const base = (window.AUDITORY_CONFIG?.apiBaseUrl ?? "").replace(/\/$/, "");
  async function request<T>(
    path: string,
    body?: Configuration | { recording_id: string } | File,
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
  return {
    capabilities: () => request<Capabilities>("/capabilities"),
    createExperiment: (configuration: Configuration) =>
      request<Experiment>("/experiments", configuration),
    recordings: (experimentId: string) =>
      request<RecordingCatalog>(
        `/experiments/${encodeURIComponent(experimentId)}/recordings`,
      ),
    selectRecording: (experimentId: string, recordingId: string) =>
      request<Experiment>(
        `/experiments/${encodeURIComponent(experimentId)}/recording-selection`,
        { recording_id: recordingId },
      ),
    uploadRecording: (experimentId: string, file: File) =>
      request<UploadResult>(
        `/experiments/${encodeURIComponent(experimentId)}/recording-upload`,
        file,
      ),
  };
}
