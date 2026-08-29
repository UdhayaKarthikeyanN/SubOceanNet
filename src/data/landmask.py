"""Coarse land mask for the North Indian Ocean demo.

A single set of simplified coastline polygons (lon, lat) is used for
  * masking land cells in the synthetic dataset,
  * drawing the offline coastline basemap (served via /api/meta).
Coordinates are approximate - good enough for a scientific demo, not for navigation.
"""
from __future__ import annotations

import numpy as np

# Each polygon: list of (lon, lat). The Eurasia/Arabia/Africa polygon traces
# coastlines and closes along the region's west/top edges so everything on the
# land side is inside the polygon.
LAND_POLYGONS: list[list[tuple[float, float]]] = [
    # Arabia + Iran/Pakistan (Makran) + India + Bangladesh + Myanmar + Malaya
    [
        (45.0, 12.8), (46.5, 13.2), (48.0, 14.0), (49.0, 14.6), (51.2, 15.3),
        (53.0, 16.5), (54.1, 17.0), (55.5, 17.9), (57.7, 18.9), (58.6, 20.0),
        (59.8, 22.0), (58.7, 23.6), (57.3, 25.0), (56.4, 26.3), (56.6, 27.0),
        (57.5, 25.6), (60.5, 25.3), (62.3, 25.1), (63.5, 25.2), (65.5, 25.1),
        (67.0, 24.9), (68.2, 23.9), (68.9, 23.0), (69.5, 22.4), (70.2, 21.1),
        (72.6, 21.1), (72.8, 19.1), (73.5, 16.5), (74.8, 12.9), (75.9, 10.5),
        (76.3, 9.3), (77.5, 8.1), (78.2, 8.9), (79.3, 9.5), (79.8, 10.3),
        (79.8, 11.9), (80.3, 13.1), (80.9, 15.2), (81.2, 16.0), (82.3, 16.9),
        (83.3, 17.7), (84.5, 19.0), (85.7, 19.8), (86.9, 20.7), (87.5, 21.4),
        (88.1, 21.7), (88.9, 21.9), (89.5, 21.7), (90.5, 22.1), (91.8, 22.3),
        (92.3, 20.9), (92.9, 20.1), (93.7, 19.4), (94.2, 17.5), (95.0, 16.2),
        (96.2, 16.0), (97.0, 15.4), (97.6, 14.5), (98.2, 13.4), (98.5, 11.9),
        (98.9, 10.0), (99.5, 8.2), (100.3, 6.5), (100.5, 5.0),
        (101.3, 5.6), (102.2, 6.5), (103.0, 8.5), (102.5, 11.5), (101.0, 12.8),
        (100.2, 13.8), (99.5, 15.5), (98.6, 17.5), (97.5, 19.5), (96.5, 22.0),
        (96.0, 25.0), (96.0, 28.0), (96.2, 30.0), (45.0, 30.0), (45.0, 12.8),
    ],
    # Horn of Africa (Somalia interior corner)
    [
        (51.4, 11.8), (50.6, 10.5), (49.2, 8.6), (47.6, 7.0), (46.2, 5.4),
        (45.0, 5.0), (45.0, 11.8), (51.4, 11.8),
    ],
    # Sri Lanka
    [
        (9.8, 80.2), (9.1, 80.6), (8.6, 81.2), (7.6, 81.7), (6.6, 81.5),
        (6.0, 80.6), (6.05, 79.9), (7.0, 79.7), (8.0, 79.8), (9.0, 79.9),
        (9.8, 80.2),
    ],
    # Andaman Islands
    [
        (13.4, 92.9), (12.6, 92.9), (11.5, 92.7), (10.9, 92.6), (10.3, 92.7),
        (10.8, 93.2), (11.6, 93.6), (12.5, 93.8), (13.2, 93.6), (13.6, 93.3),
        (13.4, 92.9),
    ],
    # Nicobar Islands
    [
        (9.2, 92.8), (8.5, 92.9), (7.8, 93.3), (7.1, 93.6), (7.3, 93.1),
        (8.0, 92.7), (8.7, 92.6), (9.2, 92.8),
    ],
    # Sumatra (sliver within region bounds)
    [
        (95.0, 5.0), (95.2, 5.9), (97.0, 5.9), (97.9, 5.0), (95.0, 5.0),
    ],
    # Socotra
    [
        (53.7, 12.3), (54.3, 12.5), (54.4, 12.1), (53.9, 11.9), (53.7, 12.3),
    ],
    # Lakshadweep (small footprint around the real atoll cluster - these are
    # scattered islets, not a solid landmass, so kept deliberately tight
    # rather than a full bounding box that would over-mask open ocean).
    # Margin comfortably covers the 0.25deg grid-snapped cell of each islet.
    [
        (72.55, 12.15), (73.25, 12.15), (73.25, 10.3), (72.55, 10.3), (72.55, 12.15),
    ],
    # Maldives (northern tip only - the rest of the atoll chain is south of
    # this app's 5N domain edge)
    [
        (72.8, 7.5), (73.55, 7.5), (73.55, 6.05), (72.8, 6.05), (72.8, 7.5),
    ],
]

# Small island chains rendered as dots on the basemap (too small to mask).
# Tuples are (lon, lat, name) - matches LAND_POLYGONS' (lon, lat) vertex
# order. backend/main.py unpacks accordingly before relabeling to {lat, lon}.
ISLAND_POINTS: list[tuple[float, float, str]] = [
    (72.9, 11.9, "Lakshadweep"), (73.2, 11.2, ""), (72.6, 10.5, ""),
    (73.1, 6.9, "Maldives"), (73.4, 6.3, ""), (73.0, 7.3, ""),
]


def points_in_polygon(x: np.ndarray, y: np.ndarray, poly) -> np.ndarray:
    """Vectorized ray-casting point-in-polygon test.

    x, y : broadcastable arrays (e.g. lon grid, lat grid).
    poly : iterable of (lon, lat) vertices.
    """
    rx, ry = float(poly[-1][0]), float(poly[-1][1])
    inside = np.zeros(np.broadcast(x, y).shape, dtype=bool)
    for px, py in poly:
        px, py = float(px), float(py)
        crosses = (py > y) != (ry > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = (rx - px) * (y - py) / (ry - py + 1e-300) + px
        inside ^= crosses & (x < x_cross)
        rx, ry = px, py
    return inside


def build_land_mask(lat: np.ndarray, lon: np.ndarray,
                    polygons=None) -> np.ndarray:
    """Boolean (ny, nx) array, True where the cell centre falls on land."""
    polygons = LAND_POLYGONS if polygons is None else polygons
    lon2, lat2 = np.meshgrid(np.asarray(lon, float), np.asarray(lat, float))
    land = np.zeros(lon2.shape, dtype=bool)
    for poly in polygons:
        land |= points_in_polygon(lon2, lat2, poly)
    return land


def coastlines_geojson() -> dict:
    """Land polygons as a GeoJSON FeatureCollection for the frontend basemap."""
    features = []
    for i, poly in enumerate(LAND_POLYGONS):
        ring = [[float(lo), float(la)] for lo, la in poly]
        features.append({
            "type": "Feature",
            "properties": {"name": f"land_{i}"},
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        })
    return {"type": "FeatureCollection", "features": features}
