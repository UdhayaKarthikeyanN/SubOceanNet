// Viridis + Plasma palettes matching matplotlib anchors.

type RGB = [number, number, number];

const VIRIDIS: RGB[] = [
  [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142],
  [31, 158, 137], [53, 183, 121], [109, 205, 89], [180, 222, 44], [253, 231, 37],
];

const PLASMA: RGB[] = [
  [13, 8, 135], [84, 2, 163], [139, 10, 165], [185, 50, 137], [204, 71, 120],
  [225, 100, 98], [242, 132, 75], [252, 166, 54], [252, 206, 37], [240, 249, 33],
];

function lerpStops(stops: RGB[], t: number): RGB {
  const x = Math.min(1, Math.max(0, t)) * (stops.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  const a = stops[i];
  const b = stops[Math.min(i + 1, stops.length - 1)];
  return [
    Math.round(a[0] + f * (b[0] - a[0])),
    Math.round(a[1] + f * (b[1] - a[1])),
    Math.round(a[2] + f * (b[2] - a[2])),
  ];
}

export function viridisRGB(t: number): RGB {
  return lerpStops(VIRIDIS, t);
}
export function plasmaRGB(t: number): RGB {
  return lerpStops(PLASMA, t);
}
export function paletteFor(kind: string): (t: number) => RGB {
  return kind === "uncertainty" ? plasmaRGB : viridisRGB;
}

export function cssGradient(kind: string): string {
  const fn = paletteFor(kind);
  const parts: string[] = [];
  for (let i = 0; i <= 12; i++) {
    const [r, g, b] = fn(i / 12);
    parts.push(`rgb(${r},${g},${b}) ${((i / 12) * 100).toFixed(1)}%`);
  }
  return `linear-gradient(to right, ${parts.join(",")})`;
}

/** Recharts-ready color stops (hex). */
export function hexStops(kind: string): string[] {
  const fn = paletteFor(kind);
  const out: string[] = [];
  for (let i = 0; i <= 9; i++) {
    const [r, g, b] = fn(i / 9);
    out.push(`#${((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1)}`);
  }
  return out;
}

export const CHART_SERIES = ["#22d3ee", "#a78bfa", "#f59e0b", "#34d399", "#f472b6", "#60a5fa"];
