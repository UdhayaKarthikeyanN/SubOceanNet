"""PyVista-based 3D bathymetric/temperature surface renderer.

Generates a 3D visualization where each depth-level surface undulates
based on temperature variations (like a terrain map), with connecting
side walls showing the thermal gradient.

Usage:
    from src.rendering.isosurface import render_isosurface_html
    html = render_isosurface_html(lats, lons, depths, temperature, date_str)
"""

from __future__ import annotations

import tempfile
import os

import numpy as np
import pyvista as pv
from matplotlib.colors import Normalize
import matplotlib.cm as cm


def render_isosurface_html(
    lats: list[float],
    lons: list[float],
    depths: list[int],
    temperature: list,  # list[list[list[float|None]]]  shape: (nz, ny, nx)
    date_str: str = "",
) -> str:
    """Render a 3D bathymetric/temperature surface and return as standalone HTML."""
    lats_arr = np.asarray(lats, dtype=float)
    lons_arr = np.asarray(lons, dtype=float)
    depths_arr = np.asarray(depths, dtype=float)
    temp = np.asarray(temperature, dtype=np.float64)

    # Replace None / NaN with the minimum valid temperature
    valid = np.isfinite(temp)
    fill_val = float(np.nanmin(temp[valid])) if valid.any() else 0.0
    temp = np.where(valid, temp, fill_val)

    nz_orig, ny, nx = temp.shape

    # Normalize x/y to [0, 1]
    lon_min, lon_max = float(lons_arr.min()), float(lons_arr.max())
    lat_min, lat_max = float(lats_arr.min()), float(lats_arr.max())
    dep_max = float(depths_arr.max())

    def _norm(arr, lo, hi):
        if hi - lo < 1e-12:
            return np.full_like(arr, 0.5)
        return (arr - lo) / (hi - lo)

    x_norm = _norm(lons_arr, lon_min, lon_max)
    y_norm = _norm(lats_arr, lat_min, lat_max)

    # Base z positions: surface=1.0, deep=0.0
    z_base = 1.0 - _norm(depths_arr, 0, dep_max)

    # Temperature range for colour mapping and undulation scaling
    tmin = float(np.nanmin(temp))
    tmax = float(np.nanmax(temp))
    temp_norm = Normalize(vmin=tmin, vmax=tmax)
    temp_cmap = cm.ScalarMappable(norm=temp_norm, cmap="rainbow")

    # Undulation scale: how much temperature variation deforms the surface
    # Each depth level gets a scale factor (more undulation near surface)
    undul_scale = 0.08  # max height of undulation as fraction of total z

    p = pv.Plotter(off_screen=True, window_size=[900, 700])
    p.set_background("#0a1120")

    # Build undulating surfaces
    grids = []
    for iz in range(nz_orig):
        z_base_val = z_base[iz]
        temp_slice = temp[iz]  # (ny, nx)

        # Compute temperature anomaly at this depth
        t_mean = float(np.nanmean(temp_slice))
        t_range = tmax - tmin if (tmax - tmin) > 0 else 1.0
        anomaly = (temp_slice - t_mean) / t_range  # roughly [-0.5, 0.5]

        # Create undulating z: base position + temperature-driven bumps
        ix, iy = np.meshgrid(x_norm, y_norm)  # (ny, nx)
        iz_undulating = z_base_val + anomaly * undul_scale

        # Create StructuredGrid (shape: 1, ny, nx)
        x3d = ix[np.newaxis, :, :]
        y3d = iy[np.newaxis, :, :]
        z3d = iz_undulating[np.newaxis, :, :]

        grid = pv.StructuredGrid(x3d, y3d, z3d)
        grid.point_data["temperature"] = temp_slice.ravel(order="F")
        grid.active_scalars_name = "temperature"
        grids.append((grid, temp_slice, z_base_val))

    # Render each undulating surface
    for iz, (grid, temp_slice, z_base_val) in enumerate(grids):
        p.add_mesh(
            grid,
            scalars="temperature",
            cmap="rainbow",
            show_scalar_bar=False,
            ambient=0.25,
            diffuse=0.75,
            specular=0.15,
            smooth_shading=True,
            show_edges=True,
            edge_color="#334455",
            line_width=0.5,
        )

    # Build side walls connecting adjacent surfaces
    def _make_wall(x_arr, y_fixed, z_top_arr, z_bot_arr, t_top, t_bot):
        """Front/back wall along x at fixed y."""
        n = len(x_arr)
        # Top edge points (with undulation)
        top_pts = np.column_stack([x_arr, np.full(n, y_fixed), z_top_arr])
        # Bottom edge points (with undulation)
        bot_pts = np.column_stack([x_arr, np.full(n, y_fixed), z_bot_arr])
        pts = np.vstack([top_pts, bot_pts])
        faces, scals = [], []
        for i in range(n - 1):
            faces.extend([4, i, i + 1, n + i + 1, n + i])
            scals.append(0.25 * (t_top[i] + t_top[i+1] + t_bot[i] + t_bot[i+1]))
        return pts, faces, scals

    def _make_wall_y(x_fixed, y_arr, z_top_arr, z_bot_arr, t_top, t_bot):
        """Side wall along y at fixed x."""
        n = len(y_arr)
        top_pts = np.column_stack([np.full(n, x_fixed), y_arr, z_top_arr])
        bot_pts = np.column_stack([np.full(n, x_fixed), y_arr, z_bot_arr])
        pts = np.vstack([top_pts, bot_pts])
        faces, scals = [], []
        for i in range(n - 1):
            faces.extend([4, i, i + 1, n + i + 1, n + i])
            scals.append(0.25 * (t_top[i] + t_top[i+1] + t_bot[i] + t_bot[i+1]))
        return pts, faces, scals

    for iz in range(nz_orig - 1):
        z_top = grids[iz][0].points[:, 2].reshape(ny, nx)
        z_bot = grids[iz + 1][0].points[:, 2].reshape(ny, nx)

        # Front wall (y = y_min, index 0 along x)
        pts, faces, scals = _make_wall(
            x_norm, y_norm[0], z_top[0, :], z_bot[0, :],
            temp[iz][0, :], temp[iz+1][0, :])
        if faces:
            mesh = pv.PolyData(pts, np.array(faces))
            rgb = temp_cmap.to_rgba(float(np.mean(scals)))[:3]
            p.add_mesh(mesh, color=rgb, opacity=0.95)

        # Back wall (y = y_max, index -1 along x)
        pts, faces, scals = _make_wall(
            x_norm, y_norm[-1], z_top[-1, :], z_bot[-1, :],
            temp[iz][-1, :], temp[iz+1][-1, :])
        if faces:
            mesh = pv.PolyData(pts, np.array(faces))
            rgb = temp_cmap.to_rgba(float(np.mean(scals)))[:3]
            p.add_mesh(mesh, color=rgb, opacity=0.95)

        # Right wall (x = x_max, index -1 along y)
        pts, faces, scals = _make_wall_y(
            x_norm[-1], y_norm, z_top[:, -1], z_bot[:, -1],
            temp[iz][:, -1], temp[iz+1][:, -1])
        if faces:
            mesh = pv.PolyData(pts, np.array(faces))
            rgb = temp_cmap.to_rgba(float(np.mean(scals)))[:3]
            p.add_mesh(mesh, color=rgb, opacity=0.95)

    # Colour bar
    dummy = grids[0][0].copy()
    dummy.point_data["temperature"] = np.full(dummy.n_points, (tmin + tmax) / 2)
    dummy.point_data.active_scalars_name = "temperature"
    p.add_mesh(
        dummy,
        scalars="temperature",
        cmap="rainbow",
        clim=[tmin, tmax],
        opacity=0.0,
        show_scalar_bar=True,
        scalar_bar_args={
            "title": "T (\u00b0C)",
            "color": "white",
            "title_font_size": 12,
            "position_x": 0.92,
            "position_y": 0.25,
            "width": 0.03,
            "height": 0.5,
        },
    )

    p.add_axes(color="#9fb0d0")

    axis_info = (
        f"X: {lon_min:.0f}\u2013{lon_max:.0f}\u00b0E  |  "
        f"Y: {lat_min:.0f}\u2013{lat_max:.0f}\u00b0N  |  "
        f"Z: 0\u2013{dep_max:.0f} m"
    )
    p.add_text(axis_info, position="lower_right", font_size=10, color="#8899bb")

    title = (
        f"Ocean Temperature \u2014 {date_str}" if date_str else "Ocean Temperature"
    )
    p.add_text(title, position="upper_left", font_size=14, color="#dbe4f3")

    p.camera.position = (1.6, 1.3, 0.9)
    p.camera.focal_point = (0.5, 0.5, 0.5)
    p.camera.up = (0, 0, 1)

    tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w")
    tmp.close()
    try:
        p.export_html(tmp.name)
        with open(tmp.name, encoding="utf-8") as f:
            html = f.read()
    finally:
        p.close()
        os.unlink(tmp.name)

    return html
