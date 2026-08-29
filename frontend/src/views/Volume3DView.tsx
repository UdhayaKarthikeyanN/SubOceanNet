import { useEffect, useMemo, useRef, useState } from "react";
import Plotly from "plotly.js-dist-min";
import { useApp } from "../state/AppContext";
import { hexStops } from "../utils/colormap";
import { api } from "../api/client";
import { Button, Card, EmptyState, Skeleton } from "../components/ui";
import type { PredictResult } from "../types";

const MAX_GRID = 42; // decimate to keep plotly snappy

function decimate(a: number[], max: number): number[] {
  if (a.length <= max) return a;
  const step = a.length / max;
  return Array.from({ length: max }, (_, i) => a[Math.floor(i * step)]);
}

/** Interpolate the 15 irregular depths onto a regular depth grid for volume rendering. */
function interpolateDepths(
  res: PredictResult,
  nDepths: number = 24
): { depths: number[]; grid: (number | null)[][][] } {
  const srcDepths = res.depths;
  const minD = srcDepths[0];
  const maxD = srcDepths[srcDepths.length - 1];
  const regDepths = Array.from(
    { length: nDepths },
    (_, i) => minD + ((maxD - minD) * i) / (nDepths - 1)
  );

  const ny = res.lats.length;
  const nx = res.lons.length;
  const grid: (number | null)[][][] = [];

  for (const zd of regDepths) {
    let k0 = 0;
    for (let k = 0; k < srcDepths.length - 1; k++) {
      if (srcDepths[k] <= zd && srcDepths[k + 1] >= zd) {
        k0 = k;
        break;
      }
    }
    const k1 = Math.min(k0 + 1, srcDepths.length - 1);
    const span = srcDepths[k1] - srcDepths[k0];
    const t = span > 0 ? (zd - srcDepths[k0]) / span : 0;

    const layer: (number | null)[][] = [];
    for (let iy = 0; iy < ny; iy++) {
      const row: (number | null)[] = [];
      for (let ix = 0; ix < nx; ix++) {
        const v0 = res.temperature[k0]?.[iy]?.[ix];
        const v1 = res.temperature[k1]?.[iy]?.[ix];
        if (v0 == null || v1 == null) {
          row.push(null);
          continue;
        }
        row.push(v0 + t * (v1 - v0));
      }
      layer.push(row);
    }
    grid.push(layer);
  }

  return { depths: regDepths, grid };
}

type VolumeMode = "volume" | "slices" | "iso";

function buildTraces(res: PredictResult, mode: VolumeMode) {
  const lats = decimate(res.lats, MAX_GRID);
  const lons = decimate(res.lons, MAX_GRID);
  const latIdx = res.lats.map((_, i) => i).filter((i) => lats.includes(res.lats[i]));
  const lonIdx = res.lons.map((_, i) => i).filter((i) => lons.includes(res.lons[i]));

  const pick = (m: (number | null)[][][], k: number) =>
    latIdx.map((i) => lonIdx.map((j) => m[k]?.[i]?.[j] ?? null));

  const colorscale = hexStops("prediction").map((c, i, arr) => [i / (arr.length - 1), c]);

  // global temp scale across all depths for a consistent colorbar
  const all = res.temperature.flat(2).filter((v): v is number => typeof v === "number");
  const zmin = Math.min(...all);
  const zmax = Math.max(...all);

  const traces: any[] = [];

  if (mode === "volume") {
    // --- 3D volume via semi-transparent surface stack ---
    // Plotly volume trace has known issues with non-uniform grids (#6673),
    // so we render ALL interpolated depths as semi-transparent surfaces
    // to create a convincing volume effect using the reliable surface renderer.
    const nDepthVoxels = 24;
    const { depths: regDepths, grid } = interpolateDepths(res, nDepthVoxels);

    regDepths.forEach((d, k) => {
      const layer = grid[k];
      // decimate this layer
      const vLats = decimate(res.lats, MAX_GRID);
      const vLons = decimate(res.lons, MAX_GRID);
      const vLatIdx = res.lats
        .map(((_, i) => i))
        .filter((i) => vLats.includes(res.lats[i]));
      const vLonIdx = res.lons
        .map((_, i) => i)
        .filter((i) => vLons.includes(res.lons[i]));
      const zGrid = vLatIdx.map(() => vLonIdx.map(() => -d));
      const surfColor = vLatIdx.map((iy) =>
        vLonIdx.map((ix) => {
          const v = layer?.[iy]?.[ix];
          return v == null ? zmin : v;
        })
      );
      traces.push({
        type: "surface",
        x: vLons,
        y: vLats,
        z: zGrid,
        surfacecolor: surfColor,
        colorscale,
        coloraxis: "coloraxis",
        opacity: 0.15,
        showscale: false,
        hovertemplate: "T %{surfacecolor:.2f} C<br>depth " + d + " m<br>lat %{y:.2f} lon %{x:.2f}<extra></extra>",
      });
    });
  } else if (mode === "slices") {
    const depthStep = Math.max(1, Math.round(res.depths.length / 7));
    res.depths.forEach((d, k) => {
      if (k % depthStep !== 0 && k !== res.depths.length - 1) return;
      traces.push({
        type: "surface",
        x: lons,
        y: lats,
        z: pick(res.temperature, k).map((row) => row.map(() => -d)),
        surfacecolor: pick(res.temperature, k),
        coloraxis: "coloraxis",
        opacity: 0.94,
        showscale: false,
        hovertemplate: `depth ${d} m<br>lat %{{y:.2f}} lon %{{x:.2f}}<br>T %{{surfacecolor:.2f}} °C<extra></extra>`,
        name: `${d} m`,
      });
    });
    } else {
    // --- True isosurface: surfaces of constant temperature in 3D ---
    // All axes normalized to [0,1] so the grid is uniform in 3D space.
    // This is required because Plotly isosurface trace fails on grids
    // with wildly different axis scales (depth 0-1000 vs lat 5-30 vs lon 45-105).
    const lonMin = res.lons[0];
    const lonMax = res.lons[res.lons.length - 1];
    const latMin = res.lats[0];
    const latMax = res.lats[res.lats.length - 1];
    const depMin = 0;
    const depMax = Math.max(...res.depths);

    // Use all 15 original depth levels for maximum detail
    const xNorm = res.lons.map((v) => (v - lonMin) / (lonMax - lonMin || 1));
    const yNorm = res.lats.map((v) => (v - latMin) / (latMax - latMin || 1));
    const zNorm = res.depths.map((d) => d / (depMax || 1));

    // Build flat value array: value[iz * ny * nx + iy * nx + ix]
    const nx = xNorm.length;
    const ny = yNorm.length;
    const value: number[] = [];
    for (let iz = 0; iz < res.depths.length; iz++) {
      for (let iy = 0; iy < ny; iy++) {
        for (let ix = 0; ix < nx; ix++) {
          const v = res.temperature[iz]?.[iy]?.[ix];
          value.push(v == null ? zmin : v);
        }
      }
    }

    traces.push({
      type: "isosurface",
      x: xNorm,
      y: yNorm,
      z: zNorm,
      value,
      isomin: zmin + 0.1 * (zmax - zmin),
      isomax: zmax - 0.02 * (zmax - zmin),
      colorscale,
      showscale: true,
      colorbar: { title: { text: "T °C" }, thickness: 12 },
      surface: { count: 12, fill: 0.7, pattern: "all" },
      caps: { x: { show: true }, y: { show: true }, z: { show: true } },
      hovertemplate:
        "T %{value:.2f} °C<extra></extra>",
    });
  }

  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#9fb0d0", size: 10 },
    margin: { l: 0, r: 0, b: 0, t: 10 },
    showlegend: false,
    scene: {
      bgcolor: "#0a1120",
      xaxis: {
        title: { text: "lon °E" }, gridcolor: "#16233c",
        ...(mode === "iso" ? { tickvals: [0, 0.25, 0.5, 0.75, 1],
          ticktext: [String(res.lons[0]), String(res.lons[Math.floor(res.lons.length * 0.25)]),
            String(res.lons[Math.floor(res.lons.length * 0.5)]),
            String(res.lons[Math.floor(res.lons.length * 0.75)]),
            String(res.lons[res.lons.length - 1])] } : {}),
      },
      yaxis: {
        title: { text: "lat °N" }, gridcolor: "#16233c",
        ...(mode === "iso" ? { tickvals: [0, 0.25, 0.5, 0.75, 1],
          ticktext: [String(res.lats[0]), String(res.lats[Math.floor(res.lats.length * 0.25)]),
            String(res.lats[Math.floor(res.lats.length * 0.5)]),
            String(res.lats[Math.floor(res.lats.length * 0.75)]),
            String(res.lats[res.lats.length - 1])] } : {}),
      },
      zaxis: {
        title: { text: "depth m" }, autorange: true, gridcolor: "#16233c",
        ...(mode === "iso" ? { tickvals: [0, 0.25, 0.5, 0.75, 1],
          ticktext: ["0", "250", "500", "750", "1000"] } : {}),
      },
      camera: { eye: { x: 1.6, y: 1.35, z: 0.85 } },
    },
    ...(mode === "slices" || mode === "volume" || mode === "iso"
      ? { coloraxis: { colorscale, cmin: zmin, cmax: zmax, colorbar: { title: { text: "T °C" }, thickness: 12 } } }
      : {}),
  };
  return { traces, layout };
}


const MODE_LABELS: Record<VolumeMode, string> = {
  volume: "Volume",
  slices: "Depth slices",
  iso: "3D Surface",
};

export default function Volume3DView() {
  const { prediction, setTab, date, region, pushToast } = useApp();
  const divRef = useRef<HTMLDivElement>(null);
  const [mode, setMode] = useState<VolumeMode>("volume");
  const [ready, setReady] = useState(false);
  const [isoHtml, setIsoHtml] = useState<string | null>(null);
  const [isoLoading, setIsoLoading] = useState(false);
  const isoFrameRef = useRef<HTMLIFrameElement>(null);

  const built = useMemo(
    () => (prediction && mode !== "iso") ? buildTraces(prediction, mode) : null,
    [prediction, mode]
  );

  // Plotly rendering for volume + slices modes
  useEffect(() => {
    if (!built || !divRef.current || mode === "iso") return;
    setReady(false);
    Plotly.react(divRef.current, built.traces, built.layout, {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["toImage"],
    }).then(() => setReady(true));
    return () => {
      if (divRef.current) Plotly.purge(divRef.current);
    };
  }, [built, mode]);

  // Fetch PyVista isosurface HTML when switching to iso mode
  useEffect(() => {
    if (mode !== "iso" || !prediction || !region || !date) return;
    let alive = true;
    setIsoLoading(true);
    api.getIsosurface({ region: region.geojson, date })
      .then((res) => { if (alive) setIsoHtml(res.html); })
      .catch((e) => {
        if (alive) {
          pushToast({ kind: "error", message: "isosurface render failed: " + e.message });
          setIsoHtml(null);
        }
      })
      .finally(() => { if (alive) setIsoLoading(false); });
    return () => { alive = false; };
  }, [mode, prediction, region, date, pushToast]);

  // Inject the PyVista HTML into the iframe once it loads
  useEffect(() => {
    if (mode !== "iso" || !isoHtml || !isoFrameRef.current) return;
    const doc = isoFrameRef.current.contentDocument;
    if (doc) {
      doc.open();
      doc.write(isoHtml);
      doc.close();
    }
  }, [isoHtml, mode]);

  if (!prediction)
    return (
      <Card title="3D volume" right={undefined}>
        <EmptyState
          icon={<span>🧊</span>}
          title="No prediction to render"
          hint="Run a prediction first - the volume view renders the selected region's temperature field as depth-stacked slices."
          action={<Button onClick={() => setTab("prediction")}>Go to Prediction</Button>}
        />
      </Card>
    );

  return (
    <Card
      title={`Temperature volume - ${prediction.date}`}
      subtitle={mode === "iso" ? "3D bathymetric surface · drag to rotate · scroll to zoom" : "Drag to rotate · scroll to zoom · hover for values"}
      right={
        <div className="inline-flex overflow-hidden rounded-lg border border-[#24365a]">
          {(["volume", "slices", "iso"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`px-2.5 py-1 text-[11px] ${mode === m ? "bg-cyan-500/20 text-cyan-200" : "bg-[#101b30] text-[#8fa2c7]"}`}
            >
              {MODE_LABELS[m]}
            </button>
          ))}
        </div>
      }
    >
      {mode === "iso" ? (
        <div className="relative h-[420px] w-full overflow-hidden rounded-lg border border-[#1c2b45] bg-[#0a1120]">
          {isoLoading && (
            <div className="oe-pulse absolute inset-0 z-10 grid place-items-center text-xs text-[#5f7096]">
              PyVista isosurface rendering...
            </div>
          )}
          {!isoLoading && !isoHtml && (
            <div className="absolute inset-0 grid place-items-center text-xs text-[#5f7096]">
              3D Surface unavailable
            </div>
          )}
          <iframe
            ref={isoFrameRef}
            className="h-full w-full border-0"
            sandbox="allow-scripts allow-same-origin"
            title="PyVista isosurface"
          />
        </div>
      ) : (
        <div className="relative h-[420px] w-full overflow-hidden rounded-lg border border-[#1c2b45] bg-[#0a1120]">
          <div ref={divRef} className="h-full w-full" />
          {!ready && (
            <div className="oe-pulse absolute inset-0 grid place-items-center text-xs text-[#5f7096]">
              building volume mesh...
            </div>
          )}
        </div>
      )}
    </Card>
  );
}
