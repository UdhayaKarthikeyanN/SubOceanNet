import { useCallback, useEffect, useState } from "react";
import {
  CartesianGrid, ErrorBar, Legend, ResponsiveContainer, Scatter, ScatterChart,
  Tooltip, XAxis, YAxis, ZAxis,
} from "recharts";
import { api } from "../api/client";
import { useApp } from "../state/AppContext";
import { CHART_SERIES } from "../utils/colormap";
import { Badge, Button, Card, EmptyState, Skeleton } from "../components/ui";
import type { ProfileResponse } from "../types";

const TOOLTIP_STYLE = {
  background: "#0d1526",
  border: "1px solid #1c2b45",
  borderRadius: 8,
  fontSize: 12,
  color: "#dbe4f3",
};

export default function ProfilesView() {
  const {
    meta, date, profilePoints, clickMode, setClickMode, pushToast,
    clearProfilePoints, setStage,
  } = useApp();
  const [profiles, setProfiles] = useState<ProfileResponse[] | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setClickMode("profile");
    setStage("prediction");
    return () => setClickMode("none");
  }, [setClickMode, setStage]);

  const fetchAll = useCallback(() => {
    if (!date || profilePoints.length === 0) {
      setProfiles(null);
      return;
    }
    setLoading(true);
    Promise.all(profilePoints.map((p) => api.getProfile(p.lat, p.lon, date)))
      .then(setProfiles)
      .catch((e) => {
        setProfiles(null);
        pushToast({ kind: "error", message: `profile load failed: ${e.message}`, retry: fetchAll });
      })
      .finally(() => setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [date, profilePoints, pushToast]);

  useEffect(() => {
    fetchAll();
  }, [fetchAll]);

  const depths = meta?.depths ?? [];
  const maxDepth = depths.length ? Math.max(...depths) : 1000;

  // build per-series scatter datasets: {x: temperature, y: depth, e: 2*unc}
  const predSeries =
    profiles?.map((p, i) => ({
      name: `P${i + 1} predicted`,
      color: CHART_SERIES[i % CHART_SERIES.length],
      dashed: false,
      data: depths
        .map((z, k) => ({
          x: p.temperature[k],
          y: z,
          e: typeof p.uncertainty[k] === "number" ? 2 * (p.uncertainty[k] as number) : undefined,
        }))
        .filter((d) => typeof d.x === "number"),
    })) ?? [];

  const refSeries =
    profiles?.flatMap((p, i) =>
      p.reference_temperature?.some((v) => v !== null)
        ? [{
            name: `P${i + 1} reference`,
            color: CHART_SERIES[i % CHART_SERIES.length],
            dashed: true,
            data: depths
              .map((z, k) => ({ x: p.reference_temperature![k], y: z }))
              .filter((d) => typeof d.x === "number"),
          }]
        : []
    ) ?? [];

  return (
    <div className="flex flex-col gap-4">
      <Card
        title="Vertical profiles T(z)"
        subtitle="Drop points on the map (amber markers P1..P5) to compare temperature-vs-depth curves."
        right={<Badge tone="amber">click map to add</Badge>}
      >
        {!date ? (
          <Skeleton className="h-64" />
        ) : profilePoints.length === 0 ? (
          <EmptyState
            icon={<span>📍</span>}
            title="No profile points yet"
            hint={
              clickMode === "profile"
                ? "Click anywhere in the ocean on the map panel to place up to 5 profile points."
                : "Enable map picking, then click the ocean."
            }
            action={<Button onClick={() => setClickMode("profile")}>Enable map picking</Button>}
          />
        ) : loading ? (
          <Skeleton className="h-72" />
        ) : (
          <>
            <ResponsiveContainer width="100%" height={400}>
              <ScatterChart margin={{ top: 10, right: 20, bottom: 12, left: 8 }}>
                <CartesianGrid stroke="#16233c" strokeDasharray="3 3" />
                <XAxis
                  type="number"
                  dataKey="x"
                  name="Temperature"
                  domain={[14, 33]}
                  tick={{ fill: "#9fb0d0", fontSize: 10 }}
                  stroke="#24365a"
                  label={{ value: "temperature (°C)", position: "insideBottom", offset: -8, fill: "#66779b", fontSize: 11 }}
                />
                <YAxis
                  type="number"
                  dataKey="y"
                  name="Depth"
                  domain={[0, maxDepth]}
                  reversed
                  tick={{ fill: "#9fb0d0", fontSize: 10 }}
                  stroke="#24365a"
                  label={{ value: "depth (m)", angle: -90, position: "insideLeft", fill: "#66779b", fontSize: 11 }}
                />
                <ZAxis range={[36, 36]} />
                <Tooltip
                  contentStyle={TOOLTIP_STYLE}
                  itemStyle={{ color: "#dbe4f3" }}
                  labelStyle={{ color: "#9fb0d0" }}
                  cursor={{ strokeDasharray: "4 4", stroke: "#24365a" }}
                  formatter={(v: number | string, n: string) =>
                    n === "Temperature" && typeof v === "number" ? `${v.toFixed(2)} °C` : String(v)
                  }
                  labelFormatter={() => ""}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} formatter={(value) => <span style={{ color: "#dbe4f3", fontSize: 11 }}>{value}</span>} />
                {predSeries.map((s, i) => (
                  <Scatter
                    key={`sp${i}`}
                    name={s.name}
                    data={s.data}
                    fill={s.color}
                    line={{ stroke: s.color, strokeWidth: 2 }}
                    shape="circle"
                    legendType="line"
                  >
                    <ErrorBar dataKey="e" direction="x" width={3} strokeWidth={1} stroke="#5f7096" fill="none" />
                  </Scatter>
                ))}
                {refSeries.map((s, i) => (
                  <Scatter
                    key={`sr${i}`}
                    name={s.name}
                    data={s.data}
                    fill="transparent"
                    line={{ stroke: s.color, strokeWidth: 1.4, strokeDasharray: "5 4" }}
                    shape="circle"
                    legendType={"plainline" as never}
                  />
                ))}
              </ScatterChart>
            </ResponsiveContainer>
            <div className="mt-2 flex items-center justify-between">
              <p className="text-[11px] text-[#66779b]">
                Solid + whiskers = AI reconstruction (±2σ MC-dropout) at {date}; dashed = reference truth.
              </p>
              <Button variant="ghost" onClick={clearProfilePoints}>
                Clear points
              </Button>
            </div>
          </>
        )}
      </Card>

      {profiles && profiles.length > 0 && (
        <Card title="Profile details">
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {profiles.map((p, i) => {
              const t0 = p.temperature[0];
              return (
                <div key={i} className="rounded-lg border border-[#1c2b45] bg-[#0a1220] px-3 py-2 text-xs">
                  <span className="font-mono font-bold" style={{ color: CHART_SERIES[i % CHART_SERIES.length] }}>
                    P{i + 1}
                  </span>{" "}
                  <span className="font-mono text-[#9fb0d0]">
                    {p.snapped_lat.toFixed(2)}°N {p.snapped_lon.toFixed(2)}°E
                  </span>
                  <div className="mt-1 text-[#66779b]">
                    SST≈{typeof t0 === "number" ? t0.toFixed(2) : "-"}°C · bottom{" "}
                    {typeof p.temperature[p.temperature.length - 1] === "number"
                      ? (p.temperature[p.temperature.length - 1] as number).toFixed(2)
                      : "-"}
                    °C @ {maxDepth} m
                  </div>
                </div>
              );
            })}
          </div>
        </Card>
      )}
    </div>
  );
}
