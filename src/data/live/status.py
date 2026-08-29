"""Status labels for the live-data pipeline. Every variable is always
reported as exactly one of these - never a bare "live"/"ok" that could be
mistaken for a guarantee of freshness."""
from __future__ import annotations

REAL = "REAL"
CACHED_REAL = "CACHED REAL"
SYNTHETIC_FALLBACK = "SYNTHETIC FALLBACK"

ALL = (REAL, CACHED_REAL, SYNTHETIC_FALLBACK)
