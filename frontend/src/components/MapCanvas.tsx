import { useEffect, useRef, useState } from "react";
import L from "leaflet";
import "leaflet-draw";
import { useApp } from "../state/AppContext";
import { paletteFor } from "../utils/colormap";
import type { RasterLayer } from "../types";

function rasterToURL(layer: RasterLayer): string {
  const h = layer.lats.length;
  const w = layer.lons.length;
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d")!;
  const img = ctx.createImageData(w, h);
  const fn = paletteFor(layer.kind);
  const span = layer.max - layer.min || 1;
  for (let y = 0; y < h; y++) {
    const srcY = h - 1 - y; // row 0 of image = max latitude
    for (let x = 0; x < w; x++) {
      const v = layer.values[srcY]?.[x];
      const o = (y * w + x) * 4;
      if (v === null || v === undefined || !isFinite(v)) {
        img.data[o + 3] = 0;
        continue;
      }
      const [r, g, b] = fn((v - layer.min) / span);
      img.data[o] = r;
      img.data[o + 1] = g;
      img.data[o + 2] = b;
      img.data[o + 3] = 235;
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL();
}

function nearestIndex(arr: number[], v: number): number {
  let lo = 0;
  let hi = arr.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] < v) lo = mid;
    else hi = mid;
  }
  return Math.abs(arr[lo] - v) <= Math.abs(arr[hi] - v) ? lo : hi;
}

export default function MapPanel() {
  const divRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<L.Map | null>(null);
  const overlayRef = useRef<L.ImageOverlay | null>(null);
  const drawnRef = useRef<L.FeatureGroup | null>(null);
  const baseRef = useRef<L.LayerGroup | null>(null);
  const markerRefs = useRef<Record<number, L.CircleMarker>>({});
  const {
    meta, region, setRegion, mapLayer, profilePoints, addProfilePoint,
    clearProfilePoints, clickMode, setClickMode, setMapLayer, pushToast,
  } = useApp();
  const [cursor, setCursor] = useState<{ lat: number; lon: number } | null>(null);

  // ---- init map (once) ----
  useEffect(() => {
    if (!divRef.current || mapRef.current) return;
    const map = L.map(divRef.current, {
      center: [17.5, 75],
      zoom: 5,
      minZoom: 4,
      zoomControl: true,
      attributionControl: false,
      maxBoundsViscosity: 0.9,
    });
    mapRef.current = map;

    const drawn = new L.FeatureGroup().addTo(map);
    drawnRef.current = drawn;

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const DrawCtl = (L.Control as any).Draw;
    map.addControl(
      new DrawCtl({
        position: "topright",
        draw: {
          polyline: false,
          circle: false,
          circlemarker: false,
          marker: false,
          rectangle: { shapeOptions: { color: "#22d3ee", weight: 2, fillOpacity: 0.06 } },
          polygon: {
            allowIntersection: false,
            shapeOptions: { color: "#22d3ee", weight: 2, fillOpacity: 0.06 },
          },
        },
        edit: { featureGroup: drawn, remove: false },
      })
    );

    const applyGeometry = (g: GeoJSON.Polygon) => {
      const ring = g.coordinates[0] as [number, number][];
      let latMin = 90, latMax = -90, lonMin = 180, lonMax = -180;
      ring.forEach(([lon, lat]) => {
        latMin = Math.min(latMin, lat); latMax = Math.max(latMax, lat);
        lonMin = Math.min(lonMin, lon); lonMax = Math.max(lonMax, lon);
      });
      setRegion({ geojson: { type: "Polygon", coordinates: [ring] },
        bbox: { latMin, latMax, lonMin, lonMax }, cells: 0 });
    };

    type DrawEvent = { layerType: string; layer: L.Layer };
    map.on("draw:created", (e) => {
      const ev = e as unknown as DrawEvent;
      drawn.clearLayers();
      drawn.addLayer(ev.layer);
      const gj = (ev.layer as unknown as { toGeoJSON: () => GeoJSON.Feature }).toGeoJSON();
      applyGeometry(gj.geometry as GeoJSON.Polygon);
    });

    const onMove = (e: L.LeafletMouseEvent) => setCursor({ lat: e.latlng.lat, lon: e.latlng.lng });
    map.on("mousemove", onMove);

    return () => {
      map.off("mousemove", onMove);
      map.remove();
      mapRef.current = null;
      overlayRef.current = null;
    };
  }, [setRegion]);

  // ---- static basemap: graticule + coastlines + islands ----
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !meta) return;
    if (baseRef.current) baseRef.current.remove();
    const group = L.layerGroup().addTo(map);
    baseRef.current = group;
    const b = meta.bounds;

    for (let la = Math.ceil(b.lat_min / 5) * 5; la <= b.lat_max; la += 5) {
      L.polyline([[la, b.lon_min], [la, b.lon_max]], { color: "#1b2947", weight: 1, interactive: false }).addTo(group);
    }
    for (let lo = Math.ceil(b.lon_min / 5) * 5; lo <= b.lon_max; lo += 5) {
      L.polyline([[b.lat_min, lo], [b.lat_max, lo]], { color: "#1b2947", weight: 1, interactive: false }).addTo(group);
    }
    L.geoJSON(meta.land_polygons as never, {
      style: { color: "#33497a", weight: 1.4, fillColor: "#16223c", fillOpacity: 0.85 },
      interactive: false,
    }).addTo(group);
    meta.islands.forEach((isl) => {
      L.circleMarker([isl.lat, isl.lon], {
        radius: 1.6, color: "#8fa2c7", fillOpacity: 0.9, weight: 1, interactive: false,
      })
        .bindTooltip(isl.name || "", { permanent: false, direction: "top" })
        .addTo(group);
    });
    map.fitBounds([
      [b.lat_min, b.lon_min],
      [b.lat_max, b.lon_max],
    ]);
    return () => {
      group.remove();
    };
  }, [meta]);

  // ---- raster overlay ----
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    if (overlayRef.current) {
      map.removeLayer(overlayRef.current);
      overlayRef.current = null;
    }
    if (!mapLayer || mapLayer.lats.length === 0) return;
    const url = rasterToURL(mapLayer);
    const bounds: L.LatLngBoundsExpression = [
      [Math.max(...mapLayer.lats), Math.min(...mapLayer.lons)],
      [Math.min(...mapLayer.lats), Math.max(...mapLayer.lons)],
    ];
    overlayRef.current = L.imageOverlay(url, bounds, { className: "oe-raster" }).addTo(map);
  }, [mapLayer]);

  // ---- profile point markers ----
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    Object.values(markerRefs.current).forEach((m) => m.remove());
    markerRefs.current = {};
    profilePoints.forEach((p, i) => {
      const m = L.circleMarker([p.lat, p.lon], {
        radius: 5, color: "#f59e0b", fillColor: "#f59e0b", fillOpacity: 1, weight: 2,
      })
        .bindTooltip(`P${i + 1}`, { permanent: true, direction: "right", className: "!bg-transparent !border-0 !text-amber-300 font-mono text-[10px] shadow-none" })
        .addTo(map);
      markerRefs.current[p.id] = m;
    });
  }, [profilePoints]);

  // ---- click-to-add profile points ----
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const onClick = (e: L.LeafletMouseEvent) => {
      if (clickMode !== "profile") return;
      const { lat, lng } = e.latlng;
      if (!meta || lat < meta.bounds.lat_min || lat > meta.bounds.lat_max ||
          lng < meta.bounds.lon_min || lng > meta.bounds.lon_max) return;
      addProfilePoint(+lat.toFixed(3), +lng.toFixed(3));
    };
    map.on("click", onClick);
    const el = divRef.current;
    if (el) el.style.cursor = clickMode === "profile" ? "crosshair" : "";
    return () => {
      map.off("click", onClick);
      if (el) el.style.cursor = "";
    };
  }, [clickMode, meta, addProfilePoint]);

  // ---- hover values over the active raster ----
  const [hoverVal, setHoverVal] = useState<number | null>(null);
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    let last = "";
    const onMove = (e: L.LeafletMouseEvent) => {
      const l = mapLayer;
      if (!l) return setHoverVal(null);
      const key = `${e.latlng.lat.toFixed(3)},${e.latlng.lng.toFixed(3)}`;
      if (key === last) return;
      last = key;
      const i = nearestIndex(l.lats, e.latlng.lat);
      const j = nearestIndex(l.lons, e.latlng.lng);
      const v = l.values[i]?.[j];
      setHoverVal(typeof v === "number" ? v : null);
    };
    map.on("mousemove", onMove);
    return () => {
      map.off("mousemove", onMove);
    };
  }, [mapLayer]);

  const presets = meta?.presets ?? [];
  const res = meta?.grid.resolution ?? 0.25;

  const applyPreset = (id?: string) => {
    const map = mapRef.current;
    if (!map) return;
    clearProfilePoints();
    if (!id) {
      setRegion(null);
      if (drawnRef.current) drawnRef.current.clearLayers();
      setMapLayer(null);
      return;
    }
    const p = presets.find((x) => x.id === id);
    if (!p) return;
    const [lonMin, latMin, lonMax, latMax] = p.bbox;
    const rect = L.rectangle(
      [[latMin, lonMin], [latMax, lonMax]] as L.LatLngBoundsLiteral,
      { color: "#22d3ee", weight: 2, fillOpacity: 0.06 }
    );
    if (drawnRef.current) {
      drawnRef.current.clearLayers();
      drawnRef.current.addLayer(rect);
    }
    setRegion({
      geojson: {
        type: "Polygon",
        coordinates: [[[lonMin, latMin], [lonMax, latMin], [lonMax, latMax],
                       [lonMin, latMax], [lonMin, latMin]]],
      },
      bbox: { latMin, latMax, lonMin, lonMax },
      cells: Math.round(((latMax - latMin) / res + 1) * ((lonMax - lonMin) / res + 1)),
      presetId: id,
    });
  };

  const sel = region;
  const selCells =
    sel &&
    Math.round(
      ((sel.bbox.latMax - sel.bbox.latMin) / res + 1) *
        ((sel.bbox.lonMax - sel.bbox.lonMin) / res + 1)
    );

  return (
    <div className="flex h-full flex-col gap-2">
      {/* preset toolbar */}
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="mr-1 text-[10px] uppercase tracking-widest text-[#5f7096]">Regions</span>
        {presets.map((p) => (
          <button
            key={p.id}
            onClick={() => applyPreset(p.id)}
            className={`rounded-md border px-2 py-1 text-[11px] transition ${
              sel?.presetId === p.id
                ? "border-cyan-400/60 bg-cyan-500/15 text-cyan-200"
                : "border-[#24365a] bg-[#101b30] text-[#8fa2c7] hover:border-[#33497a]"
            }`}
          >
            {p.name}
          </button>
        ))}
        <button
          onClick={() => applyPreset(undefined)}
          className="rounded-md border border-[#24365a] bg-transparent px-2 py-1 text-[11px] text-[#66779b] hover:text-red-300"
        >
          Clear
        </button>
        <span className="ml-auto hidden text-[10px] text-[#66779b] md:inline">
          Draw a rectangle or polygon on the map to select a custom region
        </span>
      </div>

      {/* map */}
      <div className="relative flex-1 overflow-hidden rounded-xl border border-[#1c2b45]" style={{ minHeight: 260 }}>
        <div ref={divRef} className="absolute inset-0 z-0" />
        {mapLayer && (
          <div className="pointer-events-none absolute bottom-3 left-3 z-[500]">
            <div className="pointer-events-auto rounded-lg border border-[#1c2b45] bg-[#0a1220]/95 px-3 py-2 font-mono text-[11px] text-[#9fb0d0] backdrop-blur">
              <span className="text-[#66779b]">cursor</span>{" "}
              {cursor ? `${cursor.lat.toFixed(2)}°N ${cursor.lon.toFixed(2)}°E` : "-"}
              {hoverVal !== null && (
                <>
                  {"  "}
                  <span style={{ color: mapLayer.kind === "uncertainty" ? "#e879f9" : "#67e8f9" }}>
                    ▸ {hoverVal.toFixed(3)} {mapLayer.units}
                  </span>
                </>
              )}
            </div>
          </div>
        )}
        {!meta && (
          <div className="oe-pulse absolute inset-0 grid place-items-center text-xs text-[#5f7096]">
            connecting to backend...
          </div>
        )}
      </div>

      {/* selection stats */}
      <div className="rounded-xl border border-[#1c2b45] bg-[#0d1526]/80 px-3 py-2 font-mono text-[11px] text-[#8fa2c7]">
        {sel ? (
          <>
            <span className="text-cyan-300">SELECTED</span>{"  "}
            {sel.presetId ? `preset:${sel.presetId}` : "custom-polygon"}{"  "}
            {sel.bbox.latMin.toFixed(2)}–{sel.bbox.latMax.toFixed(2)}°N,{" "}
            {sel.bbox.lonMin.toFixed(2)}–{sel.bbox.lonMax.toFixed(2)}°E{"  "}
            <span className="text-[#5f7096]">| ~{selCells} grid cells @ {res}°</span>
          </>
        ) : (
          <span className="text-[#66779b]">
            No selection yet - use the drawing tools (⬒ rectangle / ⬠ polygon, top-right of map)
            or pick a preset above.
          </span>
        )}
      </div>

      {(clickMode === "profile" || profilePoints.length > 0) && (
        <div className="flex items-center justify-between rounded-lg border border-amber-500/30 bg-amber-500/5 px-3 py-1.5 text-[11px] text-amber-200">
          <span>
            Profile mode: click the map to drop temperature-profile points ({profilePoints.length}/5)
          </span>
          <button
            onClick={() => { clearProfilePoints(); setClickMode("none"); }}
            className="rounded border border-current px-1.5 py-0.5 text-[10px] hover:bg-white/10"
          >
            Clear points
          </button>
        </div>
      )}
      {region && region.cells > 0 && region.cells > 10000 && (
        <div className="rounded-lg border border-red-500/40 bg-red-500/5 px-3 py-1.5 text-[11px] text-red-200">
          Selection is very large (~{selCells} cells); predictions are capped - choose a smaller area.
        </div>
      )}
    </div>
  );
}
