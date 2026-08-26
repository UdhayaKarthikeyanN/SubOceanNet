import { cssGradient } from "../utils/colormap";

export default function Legend({
  kind,
  min,
  max,
  units,
  title,
}: {
  kind: string;
  min: number;
  max: number;
  units: string;
  title?: string;
}) {
  if (!isFinite(min) || !isFinite(max)) return null;
  return (
    <div className="pointer-events-auto w-56 rounded-lg border border-[#1c2b45] bg-[#0a1220]/95 p-2.5 shadow-xl backdrop-blur">
      {title && (
        <div className="mb-1 truncate text-[10px] font-semibold uppercase tracking-wider text-[#8fa2c7]">
          {title}
        </div>
      )}
      <div
        className="h-2.5 w-full rounded"
        style={{ background: cssGradient(kind) }}
      />
      <div className="mt-1 flex justify-between font-mono text-[10px] text-[#9fb0d0]">
        <span>{min.toFixed(2)}</span>
        <span className="text-[#66779b]">{units}</span>
        <span>{max.toFixed(2)}</span>
      </div>
    </div>
  );
}
