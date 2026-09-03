import { useCallback, useEffect, useMemo, useState } from "react";
import { pollJob, api } from "../api/client";
import { useApp } from "../state/AppContext";
import DepthSlider from "../components/DepthSlider";
import Legend from "../components/Legend";
import PredictionProgress from "../components/PredictionProgress";
import { Badge, Button, Card, EmptyState, Stat } from "../components/ui";

export default function PredictionView() {
  const {
    meta, date, region, prediction, setPrediction,
    depthIdx, showUncertainty, setShowUncertainty,
    setMapLayer, setStage, playingDepth, pushToast,
  } = useApp();
  const [job, setJob] = useState<{ progress: number; stage: string; message?: string } | null>(null);

  useEffect(() => {
    setStage(prediction ? "prediction" : job ? "embedding" : "input");
  }, [prediction, job, setStage]);

  // derive the shared map raster from prediction state
  useEffect(() => {
    if (!prediction) {
      setMapLayer(null);
      return;
    }
    const d = prediction.depths[depthIdx];
    const src = showUncertainty ? prediction.uncertainty : prediction.temperature;
    const slice = src[depthIdx];
    const finite = slice.flat().filter((v): v is number => typeof v === "number");
    if (!finite.length) return;
    setMapLayer({
      kind: showUncertainty ? "uncertainty" : "prediction",
      title: `${showUncertainty ? "Uncertainty" : "Temperature"} @ ${d} m (${prediction.date})`,
      units: showUncertainty ? "± degC" : "degC",
      lats: prediction.lats,
      lons: prediction.lons,
      values: slice,
      min: Math.min(...finite),
      max: Math.max(...finite),
    });
  }, [prediction, depthIdx, showUncertainty, setMapLayer]);

  const runPredict = useCallback(async () => {
    if (!region || !date) return;
    setPrediction(null);
    setJob({ progress: 0, stage: "queued" });
    try {
      const started = await api.startPredict({ date, region_geojson: region.geojson });
      const result = await pollJob(started.job_id, (progress, stage, message) =>
        setJob({ progress, stage: stage || "", message })
      );
      setPrediction(result);
      pushToast({
        kind: "success",
        message: `Prediction complete: ${result.ocean_cells} ocean cells x ${result.depths.length} depths in ${result.runtime_s}s`,
      });
    } catch (e) {
      pushToast({
        kind: "error",
        message: `Prediction failed - ${(e as Error).message}`,
        retry: runPredict,
      });
    } finally {
      setJob(null);
    }
  }, [region, date, setPrediction, pushToast]);

  const stats = useMemo(() => {
    if (!prediction) return null;
    const slice = (showUncertainty ? prediction.uncertainty : prediction.temperature)[depthIdx];
    const finite = slice.flat().filter((v): v is number => typeof v === "number");
    if (!finite.length) return null;
    return {
      min: Math.min(...finite),
      max: Math.max(...finite),
      mean: finite.reduce((a, b) => a + b, 0) / finite.length,
    };
  }, [prediction, depthIdx, showUncertainty]);

  if (!meta) return <Card><SkeletonRow /></Card>;

  return (
    <div className="flex flex-col gap-4">
      <Card
        title="Stage 02 + 03 - Reconstruct subsurface temperature"
        subtitle="Encoder consumes 7 surface variables; decoder emits temperature at 15 depths with MC-dropout uncertainty."
        right={<Badge tone={meta.demo_mode ? "amber" : "green"}>{meta.demo_mode ? "DEMO MODEL" : "TRAINED"}</Badge>}
      >
        {!region ? (
          <EmptyState
            icon={<span>⬒</span>}
            title="No region selected"
            hint="Draw a rectangle/polygon on the map or choose a preset region first."
          />
        ) : (
          <div className="flex flex-wrap items-center gap-3">
            <Button onClick={runPredict} disabled={!!job}>
              {job ? "Predicting..." : "Run prediction"}
            </Button>
            <span className="font-mono text-xs text-[#7d8db0]">
              {date} · ~{region.cells || "?"} cells · MC passes: {meta.mc_passes}
            </span>
          </div>
        )}

        {job && (
          <PredictionProgress
            progress={job.progress}
            stage={job.stage}
            message={job.message}
            mcPasses={meta.mc_passes}
          />
        )}
      </Card>

      {!prediction && !job && (
        <EmptyState
          icon={<span>🌊</span>}
          title="No prediction yet"
          hint="Select a region and press Run prediction - results appear here and on the map with a depth slider."
        />
      )}

      {prediction && (
        <>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-6">
            <Stat label="date" value={prediction.date} />
            <Stat label="ocean cells" value={prediction.ocean_cells} />
            <Stat label="runtime" value={prediction.runtime_s} unit="s" />
            <Stat label="model" value={prediction.model_version} />
            {stats && (
              <>
                <Stat label={showUncertainty ? "σ range" : "T range"} value={`${stats.min.toFixed(2)} .. ${stats.max.toFixed(2)}`} unit="°C" />
                <Stat label={showUncertainty ? "mean σ" : "mean T"} value={stats.mean.toFixed(2)} unit="°C" />
              </>
            )}
          </div>

          <DepthSlider />

          <div className="flex items-center gap-3">
            <div className="inline-flex overflow-hidden rounded-lg border border-[#24365a]">
              <button
                onClick={() => setShowUncertainty(false)}
                className={`px-3 py-1.5 text-xs font-medium ${!showUncertainty ? "bg-cyan-500/20 text-cyan-200" : "bg-[#101b30] text-[#8fa2c7]"}`}
              >
                Temperature
              </button>
              <button
                onClick={() => setShowUncertainty(true)}
                className={`px-3 py-1.5 text-xs font-medium ${showUncertainty ? "bg-fuchsia-500/20 text-fuchsia-200" : "bg-[#101b30] text-[#8fa2c7]"}`}
              >
                Uncertainty ± σ
              </button>
            </div>
            {stats && (
              <Legend
                kind={showUncertainty ? "uncertainty" : "prediction"}
                min={stats.min}
                max={stats.max}
                units={showUncertainty ? "± degC" : "degC"}
              />
            )}
          </div>

          <p className="text-xs leading-relaxed text-[#66779b]">
            The map shows the selected depth level. Use <b>Animate sweep</b> to flip through all 15
            levels, or open the 3D tab for a volume rendering. Outputs contain ONLY temperature -
            input variables are never reconstructed.
          </p>
        </>
      )}
    </div>
  );
}

function SkeletonRow() {
  return (
    <div className="flex flex-col gap-3">
      <div className="oe-pulse h-10 rounded-lg bg-[#15223a]" />
      <div className="oe-pulse h-40 rounded-lg bg-[#15223a]" />
    </div>
  );
}
