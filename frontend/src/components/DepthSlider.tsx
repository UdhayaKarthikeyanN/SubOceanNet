import { useEffect } from "react";
import { useApp } from "../state/AppContext";

const ICON_PLAY = (
  <svg width="12" height="12" viewBox="0 0 12 12"><path d="M2.5 1.5v9l8-4.5z" fill="currentColor" /></svg>
);
const ICON_PAUSE = (
  <svg width="12" height="12" viewBox="0 0 12 12"><path d="M2.5 1.5h2.6v9H2.5zm4.4 0h2.6v9H6.9z" fill="currentColor" /></svg>
);

export default function DepthSlider() {
  const { meta, depthIdx, setDepthIdx, playingDepth, setPlayingDepth } = useApp();
  const depths = meta?.depths ?? [];

  useEffect(() => {
    if (!playingDepth || depths.length === 0) return;
    const t = window.setInterval(() => setDepthIdx((depthIdx + 1) % depths.length), 650);
    return () => window.clearInterval(t);
  }, [playingDepth, depthIdx, depths.length, setDepthIdx]);

  if (!depths.length) return null;
  return (
    <div className="rounded-xl border border-[#1c2b45] bg-[#0d1526]/90 p-3">
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-baseline gap-2">
          <span className="text-[10px] uppercase tracking-widest text-[#5f7096]">Depth</span>
          <span className="font-mono text-lg font-bold text-cyan-300">{depths[depthIdx]} m</span>
        </div>
        <button
          onClick={() => setPlayingDepth(!playingDepth)}
          className={`inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs font-medium transition ${
            playingDepth
              ? "border-cyan-500/40 bg-cyan-500/15 text-cyan-200"
              : "border-[#24365a] bg-[#101b30] text-[#8fa2c7] hover:border-[#33497a]"
          }`}
        >
          {playingDepth ? ICON_PAUSE : ICON_PLAY}
          {playingDepth ? "Pause sweep" : "Animate sweep"}
        </button>
      </div>
      <input
        type="range"
        min={0}
        max={depths.length - 1}
        step={1}
        value={depthIdx}
        onChange={(e) => setDepthIdx(Number(e.target.value))}
        className="w-full accent-cyan-400"
      />
      <div className="mt-1 flex justify-between font-mono text-[9px] text-[#5f7096]">
        {depths.map((d, i) => (
          <span key={d} className={i === depthIdx ? "text-cyan-300" : ""}>
            {d}
          </span>
        ))}
      </div>
    </div>
  );
}
