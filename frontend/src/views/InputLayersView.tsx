import { useEffect, useState } from "react";
import { api } from "../api/client";
import { useApp } from "../state/AppContext";
import type { LayerData, VariableInfo } from "../types";
import Legend from "../components/Legend";
import { Badge, Card, EmptyState, Skeleton } from "../components/ui";

export default function InputLayersView() {
  const { meta, date, region, mapLayer, setMapLayer, pushToast, setStage } = useApp();
  const [activeVar, setActiveVar] = useState<string>("sst");
  const [loading, setLoading] = useState(false);
  const [cropToRegion, setCropToRegion] = useState(false);

  useEffect(() => setStage("input"), [setStage]);

  useEffect(() => {
    if (!meta || !date) return;
    let alive = true;
    setLoading(true);
    api
      .getLayer(activeVar, date, cropToRegion && region ? region.geojson : undefined)
      .then((l: LayerData) => {
        if (!alive) return;
        setMapLayer({
          kind: "input",
          title: `${l.long_name} (${l.date})`,
          units: l.units,
          lats: l.lats,
          lons: l.lons,
          values: l.values,
          min: l.legend.min,
          max: l.legend.max,
        });
      })
      .catch((e) => {
        if (alive) {
          setMapLayer(null);
          pushToast({ kind: "error", message: `layer load failed: ${e.message}`,
            retry: () => setActiveVar((v) => v) });
        }
      })
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [meta, activeVar, date, region, cropToRegion, setMapLayer, pushToast]);

  const vars: VariableInfo[] = meta?.variables ?? [];
  const active = vars.find((v) => v.name === activeVar);
  const values = mapLayer?.values ?? [];
  const finite = values.flat().filter((v): v is number => typeof v === "number");

  return (
    <div className="flex flex-col gap-4">
      <Card
        title="Stage 01 - Surface observations fed to the encoder"
        subtitle="These 7 satellite-derived fields are the ONLY inputs; the network never outputs them."
        right={<Badge tone="cyan">INPUT</Badge>}
      >
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 xl:grid-cols-4">
          {vars.length === 0 &&
            Array.from({ length: 7 }).map((_, i) => <Skeleton key={i} className="h-14" />)}
          {vars.map((v) => (
            <button
              key={v.name}
              onClick={() => setActiveVar(v.name)}
              className={`rounded-lg border px-3 py-2 text-left transition ${
                activeVar === v.name
                  ? "border-cyan-400/60 bg-cyan-500/10 shadow-[0_0_14px_-6px_rgba(34,211,238,0.8)]"
                  : "border-[#1c2b45] bg-[#0a1220] hover:border-[#33497a]"
              }`}
            >
              <div className={`font-mono text-sm font-bold ${activeVar === v.name ? "text-cyan-300" : "text-[#b9c7e2]"}`}>
                {v.name}
              </div>
              <div className="mt-0.5 text-[10px] leading-tight text-[#66779b]">{v.long_name}</div>
            </button>
          ))}
        </div>
      </Card>

      <Card
        title={active ? `${active.long_name} - ${mapLayer?.title?.includes("(") ? mapLayer.title.split("(")[1]?.replace(")", "") ?? date : date}` : "Layer"}
        subtitle={`${active?.units ?? ""} · rendered on the map · hover the map for exact cell values`}
        right={
          <label className="flex items-center gap-2 text-xs text-[#8fa2c7]">
            <input
              type="checkbox"
              className="accent-cyan-400"
              checked={cropToRegion}
              disabled={!region}
              onChange={(e) => setCropToRegion(e.target.checked)}
            />
            crop to selection
          </label>
        }
      >
        {loading ? (
          <Skeleton className="h-24" />
        ) : !mapLayer ? (
          <EmptyState title="No layer loaded" hint="Pick a variable above and make sure the backend is running." />
        ) : (
          <>
            <Legend kind="input" min={mapLayer.min} max={mapLayer.max} units={mapLayer.units} />
            <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
              <MiniStat label="cells" value={finite.length} />
              <MiniStat label="min" value={Math.min(...finite).toFixed(2)} unit={active?.units} />
              <MiniStat label="max" value={Math.max(...finite).toFixed(2)} unit={active?.units} />
              <MiniStat
                label="mean"
                value={(finite.reduce((a, b) => a + b, 0) / Math.max(finite.length, 1)).toFixed(2)}
                unit={active?.units}
              />
            </div>
          </>
        )}
      </Card>
    </div>
  );
}

function MiniStat({ label, value, unit }: { label: string; value: React.ReactNode; unit?: string }) {
  return (
    <div className="rounded-lg border border-[#1c2b45] bg-[#0a1220] px-3 py-2">
      <div className="text-[10px] uppercase tracking-wider text-[#5f7096]">{label}</div>
      <div className="font-mono text-sm text-[#dbe4f3]">
        {value} <span className="text-[10px] text-[#7d8db0]">{unit}</span>
      </div>
    </div>
  );
}
