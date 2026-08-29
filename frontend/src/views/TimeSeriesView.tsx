import { useCallback, useEffect, useMemo, useState } from "react";
import {
  CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api } from "../api/client";
import { useApp } from "../state/AppContext";
import { CHART_SERIES } from "../utils/colormap";
import { Badge, Button, Card, EmptyState, Skeleton } from "../components/ui";
import type { TimeSeriesResponse } from "../types";

const DEPTH_CHOICES = [0, 50, 100, 200, 500, 1000];

export default function TimeSeriesView() {
  const {
    meta, region, profilePoints, clickMode, setClickMode, pushToast,
    addProfilePoint, setStage,
  } = useApp();
  const [selDepthIdxs, setSelDepthIdxs] = useState<number[]>([]);
  const [areaMean, setAreaMean] = useState(false);
  const [pointIdx, setPointIdx] = useState(0);
  const [data, setData] = useState<TimeSeriesResponse | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => setStage("prediction"), [setStage]);

  const depths = meta?.depths ?? [];
  useEffect(() => {
    if (depths.length && selDepthIdxs.length === 0) {
      // default: three representative levels
      setSelDepthIdxs([0, depths.indexOf(100) >= 0 ? depths.indexOf(100) : 7, depths.length - 1]);
    }
  }, [depths, selDepthIdxs.length]);

  const point = profilePoints[pointIdx];

  const fetchSeries = useCallback(() => {
    if (!meta || !point) return;
    setLoading(true);
    api
      .getTimeseries({
        lat: point.lat,
        lon: point.lon,
        depth: depths[0],
        stride_days: Math.max(5, Math.round(730 / 90)),
        region: areaMean && region ? region.geojson : undefined,
      })
      .then(setData)
      .catch((e) => {
        setData(null);
        pushToast({ kind: "error", message: `timeseries failed: ${e.message}`, retry: fetchSeries });
      })
      .finally(() => setLoading(false));
  }, [meta, point, areaMean, region, depths, pushToast]);

  useEffect(fetchSeries, [fetchSeries]);

  const chartData = useMemo(() => {
    if (!data) return [];
    return data.dates.map((d, i) => {
      const row: Record<string, string | number | null> = { date: d };
      selDepthIdxs.forEach((k) => {
        row[`d${k}`] = data.all_depths_matrix[i]?.[k] ?? null;
        row[`r${k}`] = data.reference_temperature?.[i] !== undefined &&
          k === (selDepthIdxs.length ? selDepthIdxs[0] : -1)
          ? data.reference_temperature![i]
          : null;
      });
      return row;
    });
  }, [data, selDepthIdxs]);

  const hasRef = !!data?.reference_temperature?.some((v) => v !== null);

  if (!profilePoints.length)
    return (
      <Card
        title="Temperature through time"
        right={<Badge tone="amber">click map to add a point</Badge>}
      >
        <EmptyState
          icon={<span>📈</span>}
          title="Pick an ocean point first"
          hint="Enable map picking and click the ocean; or draw a region to compute polygon-mean series."
          action={<Button onClick={() => setClickMode("profile")}>Enable map picking</Button>}
        />
      </Card>
    );

  return (
    <div className="flex flex-col gap-4">
      <Card
        title="Predicted temperature through time"
        subtitle={
          areaMean && region
            ? `Polygon mean over the selected region · ${data?.dates[0] ?? ""} .. ${data?.dates[data.dates.length - 1] ?? ""}`
            : `Point ${point.lat.toFixed(2)}°N ${point.lon.toFixed(2)}°E`
        }
        right={<Badge tone="violet">stage 03 output</Badge>}
      >
        <div className="mb-3 flex flex-wrap items-center gap-3">
          <label className="flex items-center gap-2 text-xs text-[#8fa2c7]">
            point
            <select
              className="rounded-md border border-[#24365a] bg-[#101b30] px-2 py-1 font-mono text-xs"
              value={Math.min(pointIdx, profilePoints.length - 1)}
              onChange={(e) => setPointIdx(Number(e.target.value))}
            >
              {profilePoints.map((p, i) => (
                <option key={p.id} value={i}>
                  P{i + 1} ({p.lat.toFixed(1)}, {p.lon.toFixed(1)})
                </option>
              ))}
            </select>
          </label>
          <button
            onClick={() => setClickMode(clickMode === "profile" ? "none" : "profile")}
            className="rounded-md border border-[#24365a] bg-[#101b30] px-2 py-1 text-xs text-[#8fa2c7] hover:border-[#33497a]"
          >
            + add point on map
          </button>
          <label className={`flex items-center gap-2 text-xs ${region ? "text-[#8fa2c7]" : "text-[#4a5876]"}`}>
            <input
              type="checkbox"
              className="accent-cyan-400"
              disabled={!region}
              checked={areaMean}
              onChange={(e) => setAreaMean(e.target.checked)}
            />
            area-mean over selection
          </label>
          {clickMode === "profile" && (
            <span className="text-[11px] text-amber-300">click the ocean to drop another point</span>
          )}
        </div>

        <div className="mb-3 flex flex-wrap gap-1.5">
          <span className="text-[10px] uppercase tracking-widest text-[#5f7096] mr-1 self-center">depths</span>
          {depths.map((z, k) =>
            DEPTH_CHOICES.includes(z) || k === depths.length - 1 ? (
              <button
                key={z}
                onClick={() =>
                  setSelDepthIdxs((s) => (s.includes(k) ? s.filter((x) => x !== k) : [...s, k]))
                }
                className={`rounded border px-2 py-0.5 font-mono text-[11px] transition ${
                  selDepthIdxs.includes(k)
                    ? "border-cyan-400/60 bg-cyan-500/15 text-cyan-200"
                    : "border-[#24365a] bg-[#101b30] text-[#66779b]"
                }`}
              >
                {z}m
              </button>
            ) : null
          )}
        </div>

        {loading ? (
          <Skeleton className="h-72" />
        ) : !data ? (
          <EmptyState title="No series loaded" />
        ) : (
          <ResponsiveContainer width="100%" height={340}>
            <LineChart data={chartData} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
              <CartesianGrid stroke="#16233c" strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fill: "#66779b", fontSize: 9 }} stroke="#24365a" minTickGap={42} />
              <YAxis
                tick={{ fill: "#9fb0d0", fontSize: 10 }}
                stroke="#24365a"
                label={{ value: "°C", angle: -90, position: "insideLeft", fill: "#66779b", fontSize: 11 }}
                domain={["auto", "auto"]}
              />
              <Tooltip contentStyle={{ background: "#0d1526", border: "1px solid #1c2b45", borderRadius: 8, color: "#dbe4f3" }} itemStyle={{ color: "#dbe4f3" }} labelStyle={{ color: "#9fb0d0" }} />
              <Legend wrapperStyle={{ fontSize: 11 }} formatter={(value) => <span style={{ color: "#dbe4f3", fontSize: 11 }}>{value}</span>} />
              {selDepthIdxs.map((k, i) => (
                <Line
                  key={k}
                  type="monotone"
                  dataKey={`d${k}`}
                  name={`${data.depths?.[k] ?? k} m`}
                  stroke={CHART_SERIES[i % CHART_SERIES.length]}
                  strokeWidth={1.8}
                  dot={false}
                  connectNulls
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        )}
        <p className="mt-2 text-[11px] text-[#66779b]">
          Each line is the AI reconstruction at one depth level{hasRef ? "; reference truth available in the Validation tab." : "."}
        </p>
      </Card>
    </div>
  );
}
