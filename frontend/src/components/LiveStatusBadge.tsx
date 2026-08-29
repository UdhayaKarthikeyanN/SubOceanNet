import { useState } from "react";
import { useApp } from "../state/AppContext";
import { Badge } from "./ui";
import type { LiveSource, LiveVariableStatus } from "../types";

const TONE: Record<LiveSource, "green" | "amber" | "red"> = {
  REAL: "green",
  "CACHED REAL": "amber",
  "SYNTHETIC FALLBACK": "red",
};

const RANK: Record<LiveSource, number> = {
  REAL: 0,
  "CACHED REAL": 1,
  "SYNTHETIC FALLBACK": 2,
};

const VAR_LABELS: Record<string, string> = {
  sst: "SST", sss: "SSS", sla: "SLA", cur_u: "Currents U", cur_v: "Currents V",
  wind_u: "Wind U", wind_v: "Wind V",
};

function fmtAge(h: number | null): string {
  if (h == null) return "—";
  if (h < 1) return `${Math.round(h * 60)}m`;
  if (h < 48) return `${h.toFixed(1)}h`;
  return `${(h / 24).toFixed(1)}d`;
}

function fmtTime(iso: string | null): string {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString(undefined, {
      month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

function VariableRow({ name, s }: { name: string; s: LiveVariableStatus }) {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-[#1c2b45] py-1.5 last:border-0">
      <span className="w-24 shrink-0 text-[11px] font-medium text-[#b9c7e2]">
        {VAR_LABELS[name] ?? name}
      </span>
      <Badge tone={TONE[s.source]}>{s.source}</Badge>
      <span className="flex-1 truncate text-right text-[10px] text-[#7d8db0]" title={s.error ?? undefined}>
        {s.provider ?? "—"} · obs {fmtTime(s.observation_time)} · age {fmtAge(s.data_age_hours)}
        {s.stale ? " · STALE" : ""}
      </span>
    </div>
  );
}

/** Compact header badge summarizing all 7 live variables; expands on click
 * to a per-variable REAL / CACHED REAL / SYNTHETIC FALLBACK breakdown with
 * provider, observation time, and data age. Renders nothing outside live
 * mode (data_source.type != "live"), so it never touches synthetic/netcdf
 * mode's existing UI. */
export default function LiveStatusBadge() {
  const { liveStatus } = useApp();
  const [open, setOpen] = useState(false);

  if (!liveStatus || !liveStatus.live_mode_active) return null;

  const entries = Object.entries(liveStatus.variables);
  const worst = entries.reduce<LiveSource>((acc, [, s]) => {
    return RANK[s.source] > RANK[acc] ? s.source : acc;
  }, "REAL");
  const refreshMin = liveStatus.refresh_interval_minutes;

  return (
    <div className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        className="inline-flex items-center gap-1.5 rounded-md border border-[#24365a] bg-[#101b30] px-2 py-0.5 text-[11px] text-[#b9c7e2] transition hover:border-[#33497a]"
        title="Live-data source status - click for detail"
      >
        <span className={`h-1.5 w-1.5 rounded-full ${
          worst === "REAL" ? "bg-emerald-400" : worst === "CACHED REAL" ? "bg-amber-400" : "bg-red-400"
        }`} />
        LIVE DATA · {worst}
      </button>
      {open && (
        <div className="absolute right-0 top-full z-[700] mt-1.5 w-[360px] rounded-lg border border-[#1c2b45] bg-[#0a1120] p-3 shadow-xl">
          <div className="mb-2 flex items-center justify-between">
            <span className="text-[11px] font-semibold uppercase tracking-wide text-[#9fb0d0]">
              Live data pipeline
            </span>
            {refreshMin != null && (
              <span className="text-[10px] text-[#5f7096]">refresh every {refreshMin}m</span>
            )}
          </div>
          {entries.map(([name, s]) => (
            <VariableRow key={name} name={name} s={s} />
          ))}
          <p className="mt-2 text-[10px] leading-relaxed text-[#5f7096]">
            REAL = fetched this cycle · CACHED REAL = reusing a previous fetch
            (stale past {liveStatus.max_age_hours ?? "?"}h) · SYNTHETIC FALLBACK = no
            live data available yet for that variable.
          </p>
        </div>
      )}
    </div>
  );
}
