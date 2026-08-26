import { useCallback, useEffect, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, Legend, ReferenceLine, ResponsiveContainer,
  Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis,
} from "recharts";
import { api } from "../api/client";
import { useApp } from "../state/AppContext";
import { hexStops } from "../utils/colormap";
import { Badge, Button, Card, EmptyState, Skeleton } from "../components/ui";
import type { MetricRow, ValidationResponse } from "../types";

const BAND_COLORS: Record<string, string> = {
  mixed_layer: "#22d3ee",
  thermocline: "#a78bfa",
  deep: "#f59e0b",
};

export default function ValidationView() {
  const { meta, date, region, pushToast, setTab, setStage } = useApp();
  const [res, setRes] = useState<ValidationResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [scatterDepth, setScatterDepth] = useState<number | null>(null);

  useEffect(() => setStage("prediction"), [setStage]);

  const fetchVal = useCallback(() => {
    if (!region || !date) return;
    setLoading(true);
    api
      .getValidation({
        region: region.geojson,
        start: date,
        end: date,
        scatter_depth: scatterDepth ?? undefined,
      })
      .then(setRes)
      .catch((e) => {
        setRes(null);
        pushToast({ kind: "error", message: `validation failed: ${e.message}`, retry: fetchVal });
      })
      .finally(() => setLoading(false));
  }, [region, date, scatterDepth, pushToast]);

  useEffect(() => {
    fetchVal();
  }, [fetchVal]);

  if (!region)
    return (
      <Card title="Validation dashboard">
        <EmptyState
          icon={<span>🎯</span>}
          title="No region selected"
          hint="Draw or preset a region on the map, then validation compares predictions against reference truth per depth."
          action={<Button onClick={() => setTab("inputs")}>Select a region</Button>}
        />
      </Card>
    );

  const metrics: MetricRow[] = res?.metrics ?? [];

  return (
    <div className="flex flex-col gap-4">
      <Card
        title="Validation dashboard"
        subtitle={`${res ? `${res.date_start} → ${res.date_end} (${res.days_sampled} day(s) sampled) · ${res.region_cells} cells` : "computing..."}`}
        right={
          res?.demo_mode ? (
            <Badge tone="amber">SYNTHETIC REFERENCE — DEMO MODE</Badge>
          ) : (
            <Badge tone="green">GLORYS / ARGO reference</Badge>
          )
        }
      >
        {loading && !res ? (
          <>
            <Skeleton className="h-64" />
            <p className="mt-2 text-center text-xs text-[#66779b]">
              running predictions across the region and comparing against reference truth...
            </p>
          </>
        ) : !res ? (
          <EmptyState title="No validation results" hint="Try again once the backend is reachable." />
        ) : (
          <>
            {/* metrics table */}
            <div className="overflow-x-auto rounded-lg border border-[#1c2b45]">
              <table className="w-full min-w-[520px] text-right font-mono text-xs">
                <thead>
                  <tr className="bg-[#101b30] text-[10px] uppercase tracking-wider text-[#7d8db0]">
                    <th className="px-3 py-2 text-left">depth (m)</th>
                    <th className="px-3 py-2">RMSE °C</th>
                    <th className="px-3 py-2">MAE °C</th>
                    <th className="px-3 py-2">Bias °C</th>
                    <th className="px-3 py-2">Pearson r</th>
                    <th className="px-3 py-2">R²</th>
                    <th className="px-3 py-2">n</th>
                  </tr>
                </thead>
                <tbody>
                  {metrics.map((m) => (
                    <tr key={m.depth} className="border-t border-[#14213a] hover:bg-[#101b30]/60">
                      <td className="px-3 py-1.5 text-left font-bold text-cyan-300">{m.depth}</td>
                      <td className="px-3 py-1.5">{m.rmse?.toFixed(3) ?? "-"}</td>
                      <td className="px-3 py-1.5">{m.mae?.toFixed(3) ?? "-"}</td>
                      <td className={`px-3 py-1.5 ${(m.bias ?? 0) >= 0 ? "text-emerald-300/80" : "text-orange-300/80"}`}>
                        {m.bias?.toFixed(3) ?? "-"}
                      </td>
                      <td className="px-3 py-1.5">{m.pearson_r?.toFixed(3) ?? "-"}</td>
                      <td className="px-3 py-1.5">{m.r2?.toFixed(3) ?? "-"}</td>
                      <td className="px-3 py-1.5 text-[#5f7096]">{m.n}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* band skill cards */}
            <div className="mt-4 grid grid-cols-1 gap-2 sm:grid-cols-3">
              {Object.entries(res.bands).map(([band, v]) => (
                <div key={band} className="rounded-lg border p-3"
                     style={{ borderColor: `${BAND_COLORS[band] ?? "#24365a"}44`, background: `${BAND_COLORS[band] ?? "#24365a"}0a` }}>
                  <div className="text-[11px] font-bold uppercase tracking-wider" style={{ color: BAND_COLORS[band] ?? "#8fa2c7" }}>
                    {band.replace("_", " ")} <span className="font-mono text-[10px] opacity-70">{v.depth_range_m[0]}–{v.depth_range_m[1]} m</span>
                  </div>
                  <div className="mt-1 grid grid-cols-2 gap-x-3 font-mono text-xs text-[#b9c7e2]">
                    <span>RMSE <b>{v.rmse?.toFixed(3) ?? "-"}</b></span>
                    <span>Bias <b>{v.bias?.toFixed(3) ?? "-"}</b></span>
                    <span>r <b>{v.pearson_r?.toFixed(3) ?? "-"}</b></span>
                    <span>R² <b>{v.r2?.toFixed(3) ?? "-"}</b></span>
                  </div>
                </div>
              ))}
            </div>

            {/* RMSE bar chart */}
            <div className="mt-4 rounded-lg border border-[#1c2b45] bg-[#0a1220] p-3">
              <h4 className="mb-2 text-xs font-semibold text-[#8fa2c7]">RMSE by depth</h4>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={metrics} margin={{ top: 4, right: 12, bottom: 0, left: -14 }}>
                  <CartesianGrid stroke="#16233c" vertical={false} strokeDasharray="3 3" />
                  <XAxis dataKey="depth" tick={{ fill: "#66779b", fontSize: 10 }} stroke="#24365a" label={{ value: "m", position: "insideBottomRight", offset: -4, fill: "#4a5876", fontSize: 10 }} />
                  <YAxis tick={{ fill: "#9fb0d0", fontSize: 10 }} stroke="#24365a" />
                  <Tooltip contentStyle={{ background: "#0d1526", border: "1px solid #1c2b45", borderRadius: 8 }} formatter={(v) => `${Number(v).toFixed(3)} °C`} />
                  <Bar dataKey="rmse" radius={[3, 3, 0, 0]}>
                    {metrics.map((m, i) => (
                      <Cell key={i} fill={hexStops("prediction")[i % 10]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* density scatter */}
            <div className="mt-4 flex flex-col gap-2 rounded-lg border border-[#1c2b45] bg-[#0a1220] p-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h4 className="text-xs font-semibold text-[#8fa2c7]">
                  Prediction vs reference @ {res.scatter.depth} m
                </h4>
                <select
                  className="rounded-md border border-[#24365a] bg-[#101b30] px-2 py-1 font-mono text-xs"
                  value={scatterDepth ?? res.scatter.depth}
                  onChange={(e) => setScatterDepth(Number(e.target.value))}
                >
                  {(meta?.depths ?? []).map((z) => (
                    <option key={z} value={z}>{z} m</option>
                  ))}
                </select>
              </div>
              <ResponsiveContainer width="100%" height={280}>
                <ScatterChart margin={{ top: 8, right: 12, bottom: 8, left: -6 }}>
                  <CartesianGrid stroke="#16233c" strokeDasharray="3 3" />
                  <XAxis
                    type="number" dataKey="x" name="predicted °C" domain={["auto", "auto"]}
                    tick={{ fill: "#66779b", fontSize: 10 }} stroke="#24365a"
                    label={{ value: "predicted °C", position: "insideBottom", offset: -6, fill: "#66779b", fontSize: 10 }}
                  />
                  <YAxis type="number" dataKey="y" name="reference °C" domain={["auto", "auto"]} tick={{ fill: "#9fb0d0", fontSize: 10 }} stroke="#24365a" />
                  <ZAxis range={[14, 14]} />
                  <Tooltip contentStyle={{ background: "#0d1526", border: "1px solid #1c2b45", borderRadius: 8 }} cursor={{ strokeDasharray: "4 4", stroke: "#24365a" }} />
                  <Scatter data={res.scatter.points.map(([x, y]) => ({ x, y }))} fill="#22d3ee" fillOpacity={0.35} />
                  <ReferenceLine
                    segment={[{ x: res.scatter.points[0]?.[1] ?? 0, y: res.scatter.points[0]?.[1] ?? 0 },
                              { x: res.scatter.points[res.scatter.points.length - 1]?.[1] ?? 40,
                                y: res.scatter.points[res.scatter.points.length - 1]?.[1] ?? 40 }]}
                    stroke="#f59e0b"
                    strokeDasharray="6 4"
                    label={{ value: "1:1", fill: "#f59e0b", fontSize: 10, position: "insideTopRight" }}
                  />
                </ScatterChart>
              </ResponsiveContainer>
            </div>

            <p className="mt-3 text-[11px] leading-relaxed text-[#66779b]">{res.skill_note}</p>
          </>
        )}
      </Card>
    </div>
  );
}
