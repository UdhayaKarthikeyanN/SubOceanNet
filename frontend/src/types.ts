export interface BBox {
  latMin: number;
  latMax: number;
  lonMin: number;
  lonMax: number;
}

export interface VariableInfo {
  name: string;
  long_name: string;
  units: string;
  valid_min: number;
  valid_max: number;
}

export interface Preset {
  id: string;
  name: string;
  bbox: [number, number, number, number]; // lonMin, latMin, lonMax, latMax
}

export interface GeoJSONPolygon {
  type: "Polygon";
  coordinates: number[][][];
}

export interface MetaResponse {
  project: string;
  version: string;
  bounds: { lat_min: number; lat_max: number; lon_min: number; lon_max: number };
  grid: { resolution: number; ny: number; nx: number };
  depths: number[];
  variables: VariableInfo[];
  date_range: [string, string];
  presets: Preset[];
  mc_passes: number;
  max_region_cells: number;
  data_source: "synthetic" | "netcdf" | "live";
  demo_mode: boolean;
  land_polygons: GeoJSON.FeatureCollection;
  islands: { lat: number; lon: number; name: string }[];
  model_status: string;
}

export interface LayerData {
  variable: string;
  long_name: string;
  units: string;
  date: string;
  source: string;
  lats: number[];
  lons: number[];
  values: (number | null)[][];
  legend: { min: number; max: number };
}

export interface PredictResult {
  date: string;
  date_requested?: string;
  depths: number[];
  units: string;
  lats: number[];
  lons: number[];
  temperature: (number | null)[][][];
  uncertainty: (number | null)[][][];
  ocean_cells: number;
  total_cells: number;
  runtime_s: number;
  model_version: string;
  demo_mode: boolean;
  region_cells_selected?: number;
  qc_report?: unknown;
}

export interface ProfilePoint {
  lat: number;
  lon: number;
  snapped_lat: number;
  snapped_lon: number;
  temperature: (number | null)[];
  uncertainty: (number | null)[];
}

export interface ProfileResponse {
  lat: number;
  lon: number;
  snapped_lat: number;
  snapped_lon: number;
  date: string;
  depths: number[];
  units: string;
  temperature: (number | null)[];
  uncertainty: (number | null)[];
  reference_temperature: (number | null)[] | null;
  reference_type: string | null;
}

export interface TimeSeriesResponse {
  lat: number;
  lon: number;
  depth_m: number;
  area_mean: boolean;
  dates: string[];
  temperature: (number | null)[];
  reference_temperature: (number | null)[] | null;
  reference_type: string | null;
  all_depths_matrix: (number | null)[][];
  depths?: number[];
  units: string;
}

export interface MetricRow {
  depth: number;
  rmse: number | null;
  mae: number | null;
  bias: number | null;
  pearson_r: number | null;
  r2: number | null;
  n: number;
}

export interface ValidationResponse {
  date_start: string;
  date_end: string;
  days_sampled: number;
  region_cells: number;
  reference_type: string | null;
  demo_mode: boolean;
  metrics: MetricRow[];
  bands: Record<string, { depth_range_m: number[]; rmse: number | null; mae: number | null; bias: number | null; pearson_r: number | null; r2: number | null }>;
  scatter: { depth: number; points: [number, number][] };
  skill_note: string;
}

export interface ModelInfo {
  status: string;
  progress: number;
  message: string;
  available: boolean;
  checkpoint_path: string;
  version?: string;
  encoder_type?: string;
  architecture_summary?: string;
  parameters?: number;
  depths_m?: number[];
  input_variables?: string[];
  trained_range?: [string, string];
  val_rmse_degC?: number;
  trained_at?: string;
  demo_mode?: boolean;
  history?: {
    epoch: number[];
    train_loss: number[];
    val_loss: number[];
    val_rmse_c: number[];
    lr: number[];
  };
}

export type LiveSource = "REAL" | "CACHED REAL" | "SYNTHETIC FALLBACK";

export interface LiveVariableStatus {
  source: LiveSource;
  provider: string | null;
  observation_time: string | null;
  fetched_at: string | null;
  stale: boolean;
  error: string | null;
  data_age_hours: number | null;
  min?: number;
  max?: number;
  mean?: number;
}

export interface LiveStatusResponse {
  mode: "synthetic" | "netcdf" | "live";
  live_mode_active: boolean;
  refresh_interval_minutes: number | null;
  max_age_hours: number | null;
  variables: Record<string, LiveVariableStatus>;
}

export interface LiveLatestResponse {
  mode: "synthetic" | "netcdf" | "live";
  latest_common_date: string | null;
  note: string;
  variables: Record<string, LiveVariableStatus>;
}

/** What the shared Leaflet canvas renders. */
export interface RasterLayer {
  kind: "input" | "prediction" | "uncertainty";
  title: string;
  units: string;
  lats: number[];
  lons: number[];
  values: (number | null)[][];
  min: number;
  max: number;
}
