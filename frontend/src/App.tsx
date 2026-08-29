import { useEffect } from "react";
import { AppProvider, useApp, type TabId } from "./state/AppContext";
import PipelineBanner from "./components/PipelineBanner";
import MapPanel from "./components/MapCanvas";
import ToastHost from "./components/ToastHost";
import Legend from "./components/Legend";
import { Badge } from "./components/ui";
import LiveStatusBadge from "./components/LiveStatusBadge";
import InputLayersView from "./views/InputLayersView";
import PredictionView from "./views/PredictionView";
import Volume3DView from "./views/Volume3DView";
import ProfilesView from "./views/ProfilesView";
import TimeSeriesView from "./views/TimeSeriesView";
import ValidationView from "./views/ValidationView";

const TABS: { id: TabId; label: string; hint: string }[] = [
  { id: "inputs", label: "Input Layers", hint: "stage 01" },
  { id: "prediction", label: "Prediction Maps", hint: "stage 02+03" },
  { id: "volume", label: "3D Volume", hint: "stage 03" },
  { id: "profiles", label: "Vertical Profiles", hint: "stage 03" },
  { id: "timeseries", label: "Time Series", hint: "stage 03" },
  { id: "validation", label: "Validation", hint: "metrics" },
];

function ModelStatusChip() {
  const { modelInfo, meta, pushToast } = useApp();
  const status = modelInfo?.status ?? meta?.model_status ?? "unknown";

  useEffect(() => {
    if (status === "error" && modelInfo?.message) {
      pushToast({ kind: "error", message: `model error: ${modelInfo.message}` }, 0);
    }
  }, [status, modelInfo?.message, pushToast]);

  if (status === "ready")
    return (
      <Badge tone={meta?.demo_mode ? "amber" : "green"}>
        {meta?.demo_mode ? "DEMO MODEL READY" : "MODEL READY"}
      </Badge>
    );
  if (status === "training")
    return (
      <span className="inline-flex items-center gap-1.5 rounded-md border border-violet-500/30 bg-violet-500/10 px-2 py-0.5 text-[11px] text-violet-300">
        <span className="oe-pulse">◍</span> warming up {modelInfo?.progress ?? 0}%
      </span>
    );
  if (status === "error") return <Badge tone="red">MODEL ERROR</Badge>;
  return <Badge tone="slate">connecting...</Badge>;
}

function Shell() {
  const {
    meta, date, setDate, tab, setTab, mapLayer,
  } = useApp();

  const dateRange = meta?.date_range;

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-[#070d1a]">
      {/* header */}
      <header className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-[#1c2b45] px-4 py-2.5">
        <div className="flex items-center gap-2.5">
          <svg width="26" height="26" viewBox="0 0 32 32">
            <rect width="32" height="32" rx="7" fill="#0b1220" stroke="#1c2b45" />
            <path d="M5 20c3-3 5 3 8 0s5 3 8 0 5 3 6 0" stroke="#22d3ee" strokeWidth="2.2" fill="none" strokeLinecap="round" />
            <path d="M5 26c3-3 5 3 8 0s5 3 8 0 5 3 6 0" stroke="#a78bfa" strokeWidth="2.2" fill="none" strokeLinecap="round" />
            <circle cx="16" cy="10" r="4.5" stroke="#f59e0b" strokeWidth="2.2" fill="none" />
          </svg>
          <div className="leading-tight">
            <h1 className="text-sm font-extrabold tracking-[0.18em] text-[#dbe4f3]">SUBOCEAN<span className="text-cyan-300">NET</span></h1>
            <p className="text-[10px] text-[#66779b]">
              satellite surfaces → latent embedding → subsurface temperature · NIO 5–30°N / 45–105°E
            </p>
          </div>
        </div>

        <div className="ml-auto flex items-center gap-3">
          <label className="flex items-center gap-2 text-xs text-[#7d8db0]">
            date
            <input
              type="date"
              value={date}
              min={dateRange?.[0]}
              max={dateRange?.[1]}
              onChange={(e) => setDate(e.target.value)}
              className="rounded-md border border-[#24365a] bg-[#101b30] px-2 py-1 font-mono text-xs text-[#dbe4f3]"
            />
          </label>
          <LiveStatusBadge />
          <ModelStatusChip />
        </div>
      </header>

      {/* always-visible pipeline diagram */}
      <PipelineBanner />

      {/* main */}
      <main className="grid min-h-0 flex-1 grid-cols-1 gap-3 p-3 lg:grid-cols-[minmax(360px,42%)_1fr]">
        {/* left column: shared map */}
        <section className="flex min-h-0 min-w-0 flex-col">
          <MapPanel />
        </section>

        {/* right column: tabs + view */}
        <section className="flex min-h-0 min-w-0 flex-col rounded-xl border border-[#1c2b45] bg-[#0a1120]">
          <nav className="flex overflow-x-auto border-b border-[#1c2b45]">
            {TABS.map((t) => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={`shrink-0 border-b-2 px-4 py-2.5 text-xs font-semibold transition ${
                  tab === t.id
                    ? "border-cyan-400 bg-cyan-500/5 text-cyan-200"
                    : "border-transparent text-[#66779b] hover:text-[#9fb0d0]"
                }`}
              >
                {t.label}
                <span className="ml-1.5 hidden text-[9px] uppercase tracking-wider opacity-60 md:inline">{t.hint}</span>
              </button>
            ))}
          </nav>
          <div className="min-h-0 flex-1 overflow-y-auto p-4">
            {tab === "inputs" && <InputLayersView />}
            {tab === "prediction" && <PredictionView />}
            {tab === "volume" && <Volume3DView />}
            {tab === "profiles" && <ProfilesView />}
            {tab === "timeseries" && <TimeSeriesView />}
            {tab === "validation" && <ValidationView />}
          </div>
        </section>
      </main>

      {/* floating legend for the active raster */}
      {mapLayer && (
        <div className="pointer-events-none fixed bottom-16 left-4 z-[600] hidden lg:block">
          <Legend
            kind={mapLayer.kind}
            min={mapLayer.min}
            max={mapLayer.max}
            units={mapLayer.units}
            title={mapLayer.title}
          />
        </div>
      )}

      <ToastHost />
    </div>
  );
}

export default function App() {
  return (
    <AppProvider>
      <Shell />
    </AppProvider>
  );
}
