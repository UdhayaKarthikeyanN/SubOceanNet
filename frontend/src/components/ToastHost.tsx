import { useApp } from "../state/AppContext";

const TONE: Record<string, string> = {
  error: "border-red-500/40 bg-red-950/90 text-red-100",
  success: "border-emerald-500/40 bg-emerald-950/90 text-emerald-100",
  info: "border-sky-500/40 bg-[#0c1930]/95 text-sky-100",
};

export default function ToastHost() {
  const { toasts, dismissToast } = useApp();
  return (
    <div className="pointer-events-none fixed bottom-4 right-4 z-[1200] flex w-80 flex-col gap-2">
      {toasts.map((t) => (
        <div
          key={t.id}
          className={`oe-toast pointer-events-auto rounded-lg border px-3.5 py-2.5 text-xs shadow-2xl backdrop-blur ${TONE[t.kind]}`}
        >
          <div className="flex items-start justify-between gap-2">
            <p className="leading-relaxed">{t.message}</p>
            <button onClick={() => dismissToast(t.id)} className="shrink-0 opacity-60 hover:opacity-100">
              ✕
            </button>
          </div>
          {t.retry && (
            <button
              onClick={() => {
                t.retry!();
                dismissToast(t.id);
              }}
              className="mt-2 rounded-md border border-current px-2 py-1 text-[11px] font-semibold hover:bg-white/10"
            >
              Retry
            </button>
          )}
        </div>
      ))}
    </div>
  );
}
