import { useEffect, useRef, useState } from "react";

const STAGES = [
  { id: "preprocessing", n: "01", label: "Preprocess", color: "#38bdf8" },
  { id: "embedding", n: "02", label: "Forward passes", color: "#a78bfa" },
  { id: "decoding", n: "03", label: "Aggregate", color: "#f59e0b" },
] as const;

const STAGE_ORDER: Record<string, number> = {
  queued: -1,
  preprocessing: 0,
  embedding: 1,
  decoding: 2,
  done: 3,
};

const MC_PASS_RE = /MC pass (\d+)\/(\d+)/;

export default function PredictionProgress({
  progress,
  stage,
  message,
  mcPasses,
}: {
  progress: number;
  stage: string;
  message?: string;
  mcPasses: number;
}) {
  const [elapsed, setElapsed] = useState(0);
  const startRef = useRef(Date.now());
  useEffect(() => {
    startRef.current = Date.now();
    const id = setInterval(() => setElapsed((Date.now() - startRef.current) / 1000), 200);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const stageIdx = STAGE_ORDER[stage] ?? -1;
  const mcMatch = message?.match(MC_PASS_RE);
  const passDone = mcMatch ? parseInt(mcMatch[1], 10) : 0;
  const passTotal = mcMatch ? parseInt(mcMatch[2], 10) : mcPasses;
  const showPasses = stageIdx === 1 || stageIdx === 2;

  return (
    <div className="mt-4 overflow-hidden rounded-xl border border-[#24365a] bg-[#0a1120]">
      {/* header: big live percentage + elapsed time */}
      <div className="flex items-baseline justify-between gap-3 border-b border-[#1c2b45] px-4 pt-3 pb-2.5">
        <div className="flex items-baseline gap-2.5">
          <span className="font-mono text-2xl font-bold tabular-nums text-[#dbe4f3]">
            {Math.round(progress)}
            <span className="text-sm text-[#5f7096]">%</span>
          </span>
          <span className="text-xs font-medium tracking-wide text-[#8fa2c7]">
            {STAGES[Math.max(stageIdx, 0)]?.label ?? "Queued"}
          </span>
        </div>
        <span className="font-mono text-xs tabular-nums text-[#5f7096]">{elapsed.toFixed(1)}s</span>
      </div>

      <div className="px-4 pt-3.5 pb-4">
        {/* 3-stage strip mirroring the pipeline banner, scoped to this job */}
        <div className="flex items-center gap-1.5">
          {STAGES.map((s, i) => {
            const state = i < stageIdx ? "done" : i === stageIdx ? "active" : "pending";
            return (
              <div key={s.id} className="flex flex-1 items-center gap-1.5">
                <div
                  className={`grid h-6 w-6 shrink-0 place-items-center rounded-md text-[10px] font-bold transition-all duration-300 ${
                    state === "active" ? "oe-pulse" : ""
                  }`}
                  style={{
                    background: state === "pending" ? "#16233c" : s.color,
                    color: state === "pending" ? "#5f7096" : "#08101f",
                  }}
                >
                  {state === "done" ? "✓" : s.n}
                </div>
                <div
                  className="h-[2px] flex-1 rounded-full transition-all duration-500"
                  style={{ background: state === "pending" ? "#1c2b45" : s.color, opacity: state === "pending" ? 1 : 0.6 }}
                />
              </div>
            );
          })}
        </div>
        <div className="mt-1 flex text-[9.5px] uppercase tracking-wider text-[#5f7096]">
          {STAGES.map((s) => (
            <span key={s.id} className="flex-1">{s.label}</span>
          ))}
        </div>

        {/* overall progress bar */}
        <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-[#15223a]">
          <div
            className="h-full rounded-full bg-gradient-to-r from-sky-400 via-violet-400 to-amber-300 transition-all duration-500"
            style={{ width: `${Math.min(100, Math.max(2, progress))}%` }}
          />
        </div>

        {/* MC-dropout stochastic-pass tracker — the actual reason this takes a moment */}
        {showPasses && passTotal > 1 && (
          <div className="mt-4">
            <div className="flex items-baseline justify-between">
              <span className="text-[10px] uppercase tracking-wider text-[#5f7096]">
                Stochastic forward passes · MC-dropout uncertainty
              </span>
              <span className="font-mono text-[11px] tabular-nums text-[#8fa2c7]">
                {passDone}/{passTotal}
              </span>
            </div>
            <div className="mt-1.5 flex gap-1">
              {Array.from({ length: passTotal }).map((_, i) => {
                const done = i < passDone;
                const active = i === passDone;
                return (
                  <div
                    key={i}
                    className={`h-4 flex-1 rounded-sm transition-all duration-300 ${active ? "oe-pulse" : ""}`}
                    style={{ background: done || active ? "#a78bfa" : "#16233c" }}
                  />
                );
              })}
            </div>
            <p className="mt-1.5 text-[10.5px] leading-relaxed text-[#66779b]">
              Each pass re-runs the full encoder–decoder with dropout active; the spread across
              all {passTotal} is the ±σ uncertainty shown per depth.
            </p>
          </div>
        )}

        {/* live backend message */}
        {message && (
          <div className="mt-3 flex items-center gap-2 rounded-lg border border-[#1c2b45] bg-[#070d1a] px-3 py-1.5">
            <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-emerald-400 oe-pulse" />
            <span className="truncate font-mono text-[11px] text-[#7d8db0]">{message}</span>
          </div>
        )}
      </div>
    </div>
  );
}
