import { useEffect, useMemo, useRef, useState } from "react";
import Plotly from "plotly.js-dist-min";
import { useApp } from "../state/AppContext";
import { Button, Card, EmptyState } from "../components/ui";
import type { PredictResult } from "../types";

/** blue -> cyan -> green -> yellow -> orange -> red */
const TEMP_COLORSCALE: [number, string][] = [
  [0, "#1e3a8a"],
  [0.2, "#22d3ee"],
  [0.4, "#22c55e"],
  [0.6, "#facc15"],
  [0.8, "#fb923c"],
  [1, "#dc2626"],
];

const MAX_GRID = 36; // decimate lat/lon so the point cloud stays smooth to rotate

function decimateIdx(n: number, max: number): number[] {
  if (n <= max) return Array.from({ length: n }, (_, i) => i);
  const step = n / max;
  const out: number[] = [];
  for (let i = 0; i < max; i++) out.push(Math.min(n - 1, Math.floor(i * step)));
  return Array.from(new Set(out));
}

const DEPTH_STEP_M = 100;

/** Linearly interpolate the model's 15 irregular depth levels onto a
 * regular grid (one layer every 100 m) so the 3D view has an evenly
 * spaced depth axis instead of the native 0,5,10,20,...,1000 levels. */
function toRegularDepths(res: PredictResult, step: number): { depths: number[]; temperature: (number | null)[][][] } {
  const srcDepths = res.depths;
  const maxDepth = srcDepths[srcDepths.length - 1];
  const depths: number[] = [];
  for (let d = 0; d <= maxDepth + 1e-6; d += step) depths.push(Math.round(d));

  const ny = res.lats.length;
  const nx = res.lons.length;
  const temperature: (number | null)[][][] = depths.map(() =>
    Array.from({ length: ny }, () => new Array<number | null>(nx).fill(null))
  );

  depths.forEach((zd, zi) => {
    let k0 = 0;
    for (let k = 0; k < srcDepths.length - 1; k++) {
      if (srcDepths[k] <= zd && srcDepths[k + 1] >= zd) { k0 = k; break; }
    }
    const k1 = Math.min(k0 + 1, srcDepths.length - 1);
    const span = srcDepths[k1] - srcDepths[k0];
    const t = span > 0 ? (zd - srcDepths[k0]) / span : 0;
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        const v0 = res.temperature[k0]?.[iy]?.[ix];
        const v1 = res.temperature[k1]?.[iy]?.[ix];
        temperature[zi][iy][ix] = v0 == null || v1 == null ? null : v0 + t * (v1 - v0);
      }
    }
  });

  return { depths, temperature };
}

function buildScatterTrace(res: PredictResult) {
  const latIdx = decimateIdx(res.lats.length, MAX_GRID);
  const lonIdx = decimateIdx(res.lons.length, MAX_GRID);
  const { depths, temperature } = toRegularDepths(res, DEPTH_STEP_M);

  const x: number[] = [];
  const y: number[] = [];
  const z: number[] = [];
  const depth: number[] = [];
  const temp: number[] = [];

  for (let k = 0; k < depths.length; k++) {
    const d = depths[k];
    for (const iy of latIdx) {
      for (const ix of lonIdx) {
        const v = temperature[k]?.[iy]?.[ix];
        if (v === null || v === undefined || !Number.isFinite(v)) continue; // land / no data
        x.push(res.lons[ix]);
        y.push(res.lats[iy]);
        z.push(-d); // surface (0m) renders at the top, deeper values sink down
        depth.push(d);
        temp.push(v);
      }
    }
  }

  const tmin = temp.length ? Math.min(...temp) : 0;
  const tmax = temp.length ? Math.max(...temp) : 1;

  const trace: any = {
    type: "scatter3d",
    mode: "markers",
    x, y, z,
    customdata: depth,
    marker: {
      size: 3.2,
      color: temp,
      colorscale: TEMP_COLORSCALE,
      cmin: tmin,
      cmax: tmax,
      opacity: 0.9,
      colorbar: {
        title: { text: "Temp (°C)", side: "right" },
        thickness: 14,
        len: 0.8,
        tickfont: { color: "#9fb0d0", size: 10 },
        titlefont: { color: "#9fb0d0", size: 11 },
      },
    },
    hovertemplate:
      "Lat: %{y:.2f}°N<br>Lon: %{x:.2f}°E<br>Depth: %{customdata:.0f} m<br>Temp: %{marker.color:.2f} °C<extra></extra>",
  };

  const maxDepth = Math.max(...res.depths, 1000);
  const depthTicks = [0, 200, 400, 600, 800, 1000].filter((d) => d <= maxDepth + 1e-6);

  // Fit the axis ranges to the SELECTED region, not the full 45-105/5-30
  // domain - otherwise a small region renders as a sliver in one corner of
  // a box sized for the whole ocean. A little padding keeps points off the
  // walls.
  const lonSpan = res.lons[res.lons.length - 1] - res.lons[0] || 1;
  const latSpan = res.lats[res.lats.length - 1] - res.lats[0] || 1;
  const lonPad = Math.max(lonSpan * 0.04, 0.3);
  const latPad = Math.max(latSpan * 0.04, 0.3);
  // Box footprint (x:y) mirrors the region's real lon:lat proportions, so a
  // tall/thin selection isn't stretched into a square. Depth keeps a fixed
  // visual weight instead - lon/lat are in degrees (~5-30 span) and depth is
  // in metres (up to 1000), so sizing z proportionally to the raw values
  // (aspectmode "data") would make the depth axis absurdly tall.
  const xyMax = Math.max(lonSpan, latSpan);

  const layout: any = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#9fb0d0", size: 10 },
    margin: { l: 0, r: 0, b: 0, t: 10 },
    showlegend: false,
    scene: {
      bgcolor: "#0a1120",
      dragmode: "orbit",
      xaxis: {
        title: { text: "Longitude (°E)" },
        range: [res.lons[0] - lonPad, res.lons[res.lons.length - 1] + lonPad],
        gridcolor: "#1c2b45", zerolinecolor: "#1c2b45",
      },
      yaxis: {
        title: { text: "Latitude (°N)" },
        range: [res.lats[0] - latPad, res.lats[res.lats.length - 1] + latPad],
        gridcolor: "#1c2b45", zerolinecolor: "#1c2b45",
      },
      zaxis: {
        title: { text: "Depth (m)" },
        tickvals: depthTicks.map((d) => -d),
        ticktext: depthTicks.map((d) => String(d)),
        gridcolor: "#1c2b45", zerolinecolor: "#1c2b45",
      },
      aspectmode: "manual",
      aspectratio: { x: (lonSpan / xyMax) * 1.3, y: (latSpan / xyMax) * 1.3, z: 0.9 },
      camera: {
        eye: { x: 1.5, y: -1.5, z: 1.15 },
        up: { x: 0, y: 0, z: 1 },
        projection: { type: "orthographic" }, // isometric look
      },
    },
  };

  return { traces: [trace], layout };
}

export default function Volume3DView() {
  const { prediction, setTab } = useApp();
  const divRef = useRef<HTMLDivElement>(null);
  const [ready, setReady] = useState(false);

  const built = useMemo(() => (prediction ? buildScatterTrace(prediction) : null), [prediction]);

  useEffect(() => {
    if (!built || !divRef.current) return;
    setReady(false);
    Plotly.react(divRef.current, built.traces, built.layout, {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["toImage"],
    }).then(() => setReady(true));
    return () => {
      if (divRef.current) Plotly.purge(divRef.current);
    };
  }, [built]);

  if (!prediction)
    return (
      <Card title="3D volume">
        <EmptyState
          icon={<span>🧊</span>}
          title="No prediction to render"
          hint="Run a prediction first, then come back here."
          action={<Button onClick={() => setTab("prediction")}>Go to Prediction</Button>}
        />
      </Card>
    );

  return (
    <Card
      title={`Temperature volume - ${prediction.date}`}
      subtitle="Isometric view - drag to rotate, scroll to zoom, right-drag (or shift-drag) to pan"
    >
      <div className="relative h-[480px] w-full overflow-hidden rounded-lg border border-[#1c2b45] bg-[#0a1120]">
        <div ref={divRef} className="h-full w-full" />
        {!ready && (
          <div className="oe-pulse absolute inset-0 grid place-items-center text-xs text-[#5f7096]">
            building temperature volume...
          </div>
        )}
      </div>
    </Card>
  );
}
