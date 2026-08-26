import { useEffect, useMemo, useRef, useState } from "react";
import Plotly from "plotly.js-dist-min";
import { useApp } from "../state/AppContext";
import { hexStops } from "../utils/colormap";
import { Button, Card, EmptyState, Skeleton } from "../components/ui";
import type { PredictResult } from "../types";

const MAX_GRID = 42; // decimate to keep plotly snappy

function decimate(a: number[], max: number): number[] {
  if (a.length <= max) return a;
  const step = a.length / max;
  return Array.from({ length: max }, (_, i) => a[Math.floor(i * step)]);
}

function buildTraces(res: PredictResult, mode: "slices" | "iso") {
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
  if (mode === "slices") {
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
    try {
      const xs: number[] = [];
      const ys: number[] = [];
      const zs: number[] = [];
      const vs: number[] = [];
      res.depths.forEach((d, k) => {
        const m = pick(res.temperature, k);
        m.forEach((row, iy) =>
          row.forEach((v, ix) => {
            if (typeof v === "number") {
              xs.push(lons[ix]); ys.push(lats[iy]); zs.push(-d); vs.push(v);
            }
          })
        );
      });
      traces.push({
        type: "isosurface",
        x: xs, y: ys, z: zs, value: vs,
        isomin: zmin + 0.25 * (zmax - zmin),
        isomax: zmax - 0.05 * (zmax - zmin),
        colorscale,
        showscale: true,
        colorbar: { title: { text: "T °C" }, thickness: 12 },
        surface: { count: 4, fill: 0.85 },
        caps: { x: { show: false }, y: { show: false }, z: { show: true } },
        hovertemplate: "T %{value:.2f} °C<extra></extra>",
      });
    } catch {
      traces.push({
        type: "scatter3d", x: [], y: [], z: [],
        mode: "markers", name: "isosurface unavailable",
      });
    }
  }

  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#9fb0d0", size: 10 },
    margin: { l: 0, r: 0, b: 0, t: 10 },
    showlegend: false,
    scene: {
      bgcolor: "#0a1120",
      xaxis: { title: { text: "lon °E" }, gridcolor: "#16233c" },
      yaxis: { title: { text: "lat °N" }, gridcolor: "#16233c" },
      zaxis: { title: { text: "depth m" }, autorange: true, gridcolor: "#16233c" },
      camera: { eye: { x: 1.6, y: 1.35, z: 0.85 } },
    },
    ...(mode === "slices"
      ? { coloraxis: { colorscale, cmin: zmin, cmax: zmax, colorbar: { title: { text: "T °C" }, thickness: 12 } } }
      : {}),
  };
  return { traces, layout };
}

export default function Volume3DView() {
  const { prediction, setTab } = useApp();
  const divRef = useRef<HTMLDivElement>(null);
  const [mode, setMode] = useState<"slices" | "iso">("slices");
  const [ready, setReady] = useState(false);

  const built = useMemo(
    () => (prediction ? buildTraces(prediction, mode) : null),
    [prediction, mode]
  );

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
      subtitle="Drag to rotate · scroll to zoom · hover for values"
      right={
        <div className="inline-flex overflow-hidden rounded-lg border border-[#24365a]">
          {(["slices", "iso"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`px-2.5 py-1 text-[11px] ${mode === m ? "bg-cyan-500/20 text-cyan-200" : "bg-[#101b30] text-[#8fa2c7]"}`}
            >
              {m === "slices" ? "Depth slices" : "Isosurface"}
            </button>
          ))}
        </div>
      }
    >
      <div className="relative h-[420px] w-full overflow-hidden rounded-lg border border-[#1c2b45] bg-[#0a1120]">
        <div ref={divRef} className="h-full w-full" />
        {!ready && (
          <div className="oe-pulse absolute inset-0 grid place-items-center text-xs text-[#5f7096]">
            building volume mesh...
          </div>
        )}
      </div>
    </Card>
  );
}
