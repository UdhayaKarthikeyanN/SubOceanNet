import React from "react";
import { useApp } from "../state/AppContext";

const STAGES = [
  {
    id: "input",
    n: "01",
    title: "INPUT VARIABLES",
    sub: "SST · SSS · SLA · currents · winds",
    color: "#38bdf8",
  },
  {
    id: "embedding",
    n: "02",
    title: "DEEP LEARNING FRAMEWORK",
    sub: "encoder -> latent vector -> decoder",
    color: "#a78bfa",
  },
  {
    id: "prediction",
    n: "03",
    title: "TEMPERATURE PREDICTION",
    sub: "15 depth levels + uncertainty",
    color: "#f59e0b",
  },
] as const;

function Arrow() {
  return (
    <svg width="26" height="14" viewBox="0 0 26 14" className="mx-1 hidden shrink-0 sm:block">
      <path d="M1 7h20m0 0l-5-5m5 5l-5 5" stroke="#3b4e75" strokeWidth="2" fill="none" strokeLinecap="round" />
    </svg>
  );
}

export default function PipelineBanner() {
  const { stage } = useApp();
  return (
    <div
      data-testid="pipeline-banner"
      className="flex items-center justify-center gap-0 overflow-x-auto border-b border-[#1c2b45] bg-[#0a1120] px-3 py-2"
    >
      {STAGES.map((s, i) => {
        const active = stage === s.id;
        return (
          <React.Fragment key={s.id}>
            {i > 0 && <Arrow />}
            <div
              className={`flex min-w-max items-center gap-2 rounded-lg border px-3 py-1.5 transition-all duration-300 ${
                active ? "border-transparent" : "border-[#1c2b45] opacity-50"
              }`}
              style={
                active
                  ? { boxShadow: `0 0 16px -4px ${s.color}`, background: `${s.color}14`, borderColor: `${s.color}66` }
                  : undefined
              }
            >
              <span
                className={`grid h-6 w-6 place-items-center rounded-md text-[11px] font-bold ${
                  active ? "oe-pulse" : ""
                }`}
                style={{ background: active ? s.color : "#22304e", color: active ? "#08101f" : "#5f7096" }}
              >
                {s.n}
              </span>
              <div className="leading-tight">
                <div
                  className="text-[11px] font-bold tracking-widest"
                  style={{ color: active ? s.color : "#7d8db0" }}
                >
                  {s.title}
                </div>
                <div className="text-[10px] text-[#66779b]">{s.sub}</div>
              </div>
            </div>
          </React.Fragment>
        );
      })}
    </div>
  );
}
