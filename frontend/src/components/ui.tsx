import React from "react";

export function Card({
  title,
  subtitle,
  right,
  children,
  className = "",
}: {
  title?: string;
  subtitle?: string;
  right?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={`rounded-xl border border-[#1c2b45] bg-[#0d1526]/80 p-4 ${className}`}>
      {(title || right) && (
        <div className="mb-3 flex items-start justify-between gap-3">
          <div>
            {title && <h3 className="text-sm font-semibold tracking-wide text-[#dbe4f3]">{title}</h3>}
            {subtitle && <p className="mt-0.5 text-xs text-[#7d8db0]">{subtitle}</p>}
          </div>
          {right}
        </div>
      )}
      {children}
    </div>
  );
}

export function Badge({
  tone = "slate",
  children,
}: {
  tone?: "slate" | "cyan" | "amber" | "violet" | "red" | "green";
  children: React.ReactNode;
}) {
  const tones: Record<string, string> = {
    slate: "bg-[#16233c] text-[#8fa2c7] border-[#24365a]",
    cyan: "bg-cyan-500/10 text-cyan-300 border-cyan-500/30",
    amber: "bg-amber-500/10 text-amber-300 border-amber-500/30",
    violet: "bg-violet-500/10 text-violet-300 border-violet-500/30",
    red: "bg-red-500/10 text-red-300 border-red-500/30",
    green: "bg-emerald-500/10 text-emerald-300 border-emerald-500/30",
  };
  return (
    <span className={`inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-[11px] font-medium ${tones[tone]}`}>
      {children}
    </span>
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`oe-pulse rounded-lg bg-[#15223a] ${className}`} />;
}

export function EmptyState({
  icon,
  title,
  hint,
  action,
}: {
  icon?: React.ReactNode;
  title: string;
  hint?: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-[#1c2b45] px-6 py-12 text-center">
      {icon && <div className="text-3xl opacity-60">{icon}</div>}
      <p className="text-sm font-medium text-[#9fb0d0]">{title}</p>
      {hint && <p className="max-w-sm text-xs leading-relaxed text-[#66779b]">{hint}</p>}
      {action && <div className="mt-2">{action}</div>}
    </div>
  );
}

export function ProgressBar({ value, label }: { value: number; label?: string }) {
  return (
    <div>
      <div className="h-2 overflow-hidden rounded-full bg-[#15223a]">
        <div
          className="h-full rounded-full bg-gradient-to-r from-sky-400 via-violet-400 to-amber-300 transition-all duration-500"
          style={{ width: `${Math.min(100, Math.max(2, value))}%` }}
        />
      </div>
      {label && <p className="mt-1.5 text-xs text-[#7d8db0]">{label}</p>}
    </div>
  );
}

export function Button({
  onClick,
  disabled,
  variant = "primary",
  children,
  className = "",
}: {
  onClick?: () => void;
  disabled?: boolean;
  variant?: "primary" | "ghost" | "danger";
  children: React.ReactNode;
  className?: string;
}) {
  const styles = {
    primary:
      "bg-gradient-to-r from-cyan-500 to-sky-600 text-white hover:brightness-110 shadow-[0_0_18px_-6px_rgba(34,211,238,0.6)]",
    ghost: "border border-[#24365a] bg-[#101b30] text-[#b9c7e2] hover:border-[#33497a]",
    danger: "bg-red-500/90 text-white hover:bg-red-500",
  }[variant];
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2 text-sm font-semibold transition disabled:cursor-not-allowed disabled:opacity-40 ${styles} ${className}`}
    >
      {children}
    </button>
  );
}

export function Stat({ label, value, unit }: { label: string; value: React.ReactNode; unit?: string }) {
  return (
    <div className="rounded-lg border border-[#1c2b45] bg-[#0a1220] px-3 py-2">
      <div className="text-[10px] uppercase tracking-wider text-[#5f7096]">{label}</div>
      <div className="font-mono text-sm text-[#dbe4f3]">
        {value}
        {unit && <span className="ml-1 text-[10px] text-[#7d8db0]">{unit}</span>}
      </div>
    </div>
  );
}
