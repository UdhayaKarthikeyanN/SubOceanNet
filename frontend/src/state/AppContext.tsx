import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError } from "../api/client";
import type {
  GeoJSONPolygon,
  LiveStatusResponse,
  MetaResponse,
  ModelInfo,
  PredictResult,
  RasterLayer,
} from "../types";

export type Stage = "input" | "embedding" | "prediction";
export type TabId = "inputs" | "prediction" | "volume" | "profiles" | "timeseries" | "validation";

export interface SelectedRegion {
  geojson: GeoJSONPolygon;
  bbox: { latMin: number; latMax: number; lonMin: number; lonMax: number };
  cells: number; // approximate grid-cell count of the selection bbox
  presetId?: string;
}

export interface ProfileMarker {
  id: number;
  lat: number;
  lon: number;
}

export interface ToastMsg {
  id: number;
  kind: "error" | "success" | "info";
  message: string;
  retry?: () => void;
}

interface AppState {
  meta: MetaResponse | null;
  modelInfo: ModelInfo | null;
  liveStatus: LiveStatusResponse | null;
  date: string;
  setDate: (d: string) => void;
  region: SelectedRegion | null;
  setRegion: (r: SelectedRegion | null) => void;
  tab: TabId;
  setTab: (t: TabId) => void;
  stage: Stage;
  setStage: (s: Stage) => void;
  mapLayer: RasterLayer | null;
  setMapLayer: (l: RasterLayer | null) => void;
  prediction: PredictResult | null;
  setPrediction: (p: PredictResult | null) => void;
  depthIdx: number;
  setDepthIdx: (i: number) => void;
  playingDepth: boolean;
  setPlayingDepth: (b: boolean) => void;
  showUncertainty: boolean;
  setShowUncertainty: (b: boolean) => void;
  profilePoints: ProfileMarker[];
  addProfilePoint: (lat: number, lon: number) => void;
  clearProfilePoints: () => void;
  clickMode: "none" | "profile";
  setClickMode: (m: "none" | "profile") => void;
  hoverReadout: { lat: number; lon: number; value: number | null } | null;
  setHoverReadout: (v: { lat: number; lon: number; value: number | null } | null) => void;
  toasts: ToastMsg[];
  pushToast: (t: Omit<ToastMsg, "id">, ttlMs?: number) => void;
  dismissToast: (id: number) => void;
}

const Ctx = createContext<AppState | null>(null);

export function useApp(): AppState {
  const v = useContext(Ctx);
  if (!v) throw new Error("useApp outside provider");
  return v;
}

let toastSeq = 1;

export function AppProvider({ children }: { children: React.ReactNode }) {
  const [meta, setMeta] = useState<MetaResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [liveStatus, setLiveStatus] = useState<LiveStatusResponse | null>(null);
  const [date, setDate] = useState("");
  const [region, setRegionRaw] = useState<SelectedRegion | null>(null);
  const [tab, setTab] = useState<TabId>("inputs");
  const [stage, setStage] = useState<Stage>("input");
  const [mapLayer, setMapLayer] = useState<RasterLayer | null>(null);
  const [prediction, setPrediction] = useState<PredictResult | null>(null);
  const [depthIdx, setDepthIdx] = useState(6);
  const [playingDepth, setPlayingDepth] = useState(false);
  const [showUncertainty, setShowUncertainty] = useState(false);
  const [profilePoints, setProfilePoints] = useState<ProfileMarker[]>([]);
  const [clickMode, setClickMode] = useState<"none" | "profile">("none");
  const [hoverReadout, setHoverReadout] = useState<AppState["hoverReadout"]>(null);
  const [toasts, setToasts] = useState<ToastMsg[]>([]);
  const metaErrShown = useRef(false);

  const dismissToast = useCallback((id: number) => {
    setToasts((ts) => ts.filter((t) => t.id !== id));
  }, []);

  const pushToast = useCallback(
    (t: Omit<ToastMsg, "id">, ttlMs = 6500) => {
      const id = toastSeq++;
      setToasts((ts) => [...ts.slice(-3), { ...t, id }]);
      if (ttlMs > 0) window.setTimeout(() => dismissToast(id), ttlMs);
    },
    [dismissToast]
  );

  const setRegion = useCallback((r: SelectedRegion | null) => {
    setRegionRaw(r);
    setPrediction(null); // stale for a new region
    setMapLayer(null);
  }, []);

  // bootstrap metadata + model status polling
  useEffect(() => {
    let alive = true;
    api
      .getMeta()
      .then((m) => {
        if (!alive) return;
        setMeta(m);
        setDate(m.date_range[1]);
      })
      .catch((e: ApiError) => {
        if (!metaErrShown.current) {
          metaErrShown.current = true;
          pushToast({ kind: "error", message: e.message });
        }
      });
    return () => {
      alive = false;
    };
  }, [pushToast]);

  useEffect(() => {
    let stop = false;
    const tick = async () => {
      try {
        const info = await api.getModelInfo();
        if (!stop) setModelInfo(info);
        return info.status === "ready" || info.status === "error";
      } catch {
        return false;
      }
    };
    const loop = async () => {
      for (;;) {
        const done = await tick();
        if (stop || done) break;
        await new Promise((r) => setTimeout(r, 2000));
      }
    };
    loop();
    return () => {
      stop = true;
    };
  }, []);

  // live-data status polling - only active once /api/meta confirms live mode
  useEffect(() => {
    if (meta?.data_source !== "live") {
      setLiveStatus(null);
      return;
    }
    let stop = false;
    const tick = async () => {
      try {
        const status = await api.getLiveStatus();
        if (!stop) setLiveStatus(status);
      } catch {
        /* transient - keep last known status */
      }
    };
    tick();
    const id = window.setInterval(tick, 30_000);
    return () => {
      stop = true;
      window.clearInterval(id);
    };
  }, [meta?.data_source]);

  const addProfilePoint = useCallback((lat: number, lon: number) => {
    setProfilePoints((ps) =>
      ps.length >= 5 ? ps : [...ps, { id: Date.now() + ps.length, lat, lon }]
    );
  }, []);

  const clearProfilePoints = useCallback(() => setProfilePoints([]), []);

  const value = useMemo<AppState>(
    () => ({
      meta, modelInfo, liveStatus, date, setDate,
      region, setRegion, tab, setTab, stage, setStage,
      mapLayer, setMapLayer, prediction, setPrediction,
      depthIdx, setDepthIdx, playingDepth, setPlayingDepth,
      showUncertainty, setShowUncertainty,
      profilePoints, addProfilePoint, clearProfilePoints,
      clickMode, setClickMode,
      hoverReadout, setHoverReadout,
      toasts, pushToast, dismissToast,
    }),
    [
      meta, modelInfo, liveStatus, date, region, setRegion, tab, stage, mapLayer, prediction,
      depthIdx, playingDepth, showUncertainty, profilePoints, addProfilePoint,
      clearProfilePoints, clickMode, hoverReadout, toasts, pushToast, dismissToast,
    ]
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}
