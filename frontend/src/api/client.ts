import type {
  LayerData,
  LiveClearCacheResponse,
  LiveLatestResponse,
  LiveRefreshResponse,
  LiveStatusResponse,
  MetaResponse,
  ModelInfo,
  PredictResult,
  ProfileResponse,
  TimeSeriesResponse,
  ValidationResponse,
} from "../types";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function j<T>(url: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url, init);
  } catch {
    throw new ApiError(0, "cannot reach the API - is the backend running on port 8000?");
  }
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      msg = body.detail ?? body.message ?? msg;
      if (typeof msg !== "string") msg = JSON.stringify(msg);
    } catch {
      /* keep default */
    }
    throw new ApiError(res.status, msg);
  }
  return (await res.json()) as T;
}

const qs = (params: Record<string, string | number | undefined | null>) => {
  const u = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") u.set(k, String(v));
  });
  const s = u.toString();
  return s ? `?${s}` : "";
};

export const api = {
  getMeta: () => j<MetaResponse>("/api/meta"),

  getModelInfo: () => j<ModelInfo>("/api/model/info"),

  getLayer: (variable: string, date: string, region?: unknown) =>
    j<LayerData>(`/api/layers/${variable}${qs({ date, region: region ? JSON.stringify(region) : undefined })}`),

  startPredict: (body: { date: string; region_geojson: unknown; mc_passes?: number }) =>
    j<{ job_id: string; status: string }>("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),

  job: (id: string) =>
    j<{
      job_id: string;
      status: "queued" | "running" | "done" | "error";
      stage: string;
      progress: number;
      message?: string;
      result?: PredictResult;
    }>(`/api/jobs/${id}`),

  getProfile: (lat: number, lon: number, date?: string) =>
    j<ProfileResponse>(`/api/profile${qs({ lat, lon, date })}`),

  getTimeseries: (p: {
    lat?: number;
    lon?: number;
    depth: number;
    start?: string;
    end?: string;
    stride_days?: number;
    region?: unknown;
  }) =>
    j<TimeSeriesResponse>(
      `/api/timeseries${qs({
        ...p,
        region:
          p.region !== undefined && p.region !== null
            ? JSON.stringify(p.region)
            : undefined,
      })}`
    ),

  getValidation: (p: { region: unknown; date?: string; start?: string; end?: string; scatter_depth?: number }) =>
    j<ValidationResponse>(
      `/api/validation${qs({
        region: JSON.stringify(p.region),
        date: p.date,
        start: p.start,
        end: p.end,
        scatter_depth: p.scatter_depth,
      })}`
    ),

  getLiveStatus: () => j<LiveStatusResponse>("/api/live/status"),

  getLiveLatest: () => j<LiveLatestResponse>("/api/live/latest"),

  refreshLiveData: () => j<LiveRefreshResponse>("/api/live/refresh", { method: "POST" }),

  clearLiveCache: () => j<LiveClearCacheResponse>("/api/live/clear_cache", { method: "POST" }),
};

export async function pollJob(
  jobId: string,
  onProgress: (progress: number, stage: string, message?: string) => void
): Promise<PredictResult> {
  for (;;) {
    const job = await api.job(jobId);
    if (job.status === "done" && job.result) return job.result;
    if (job.status === "error") throw new ApiError(500, job.message || "prediction failed");
    onProgress(job.progress ?? 0, job.stage ?? "", job.message);
    await new Promise((r) => setTimeout(r, 650));
  }
}
