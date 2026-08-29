"""Live (real/near-real-time) data pipeline: Copernicus Marine + ERA5 fetch,
on-disk caching, automatic background refresh, and REAL/CACHED REAL/
SYNTHETIC FALLBACK status tracking for the 7 surface input variables.

Only activated when configs/config.yaml sets data_source.type: live. See
manager.get_manager() for the orchestration entry point and
src/data/loaders.py for how it plugs into the existing preprocessing /
inference pipeline unchanged.
"""
from __future__ import annotations

from . import status
from .manager import get_manager, latest_payload, reset_manager, status_payload

__all__ = ["status", "get_manager", "reset_manager", "status_payload", "latest_payload"]
