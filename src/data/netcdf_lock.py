"""Process-wide lock serializing NetCDF/HDF5 file access.

netCDF4/HDF5 (the C library xarray and netCDF4-python sit on top of) is not
thread-safe for concurrent access on the wheels this project installs.
FastAPI runs sync endpoints in a threadpool, so two requests that each open
a NetCDF file at the same time - e.g. dropping multiple vertical-profile
points, which the frontend fetches in parallel via Promise.all - can call
into HDF5 from separate OS threads simultaneously. That crashes the whole
process with no Python traceback (a native segfault, not a catchable
exception), not just a slow response.

Every xr.open_dataset/open_dataarray/open_mfdataset/to_netcdf call that
isn't already covered by a one-time cached load (e.g. the demo model's
warm-loaded synthetic dataset) should be wrapped in this lock.
"""
from __future__ import annotations

import threading

NETCDF_IO_LOCK = threading.Lock()
