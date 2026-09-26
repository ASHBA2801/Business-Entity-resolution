"""Shared helpers: TSV I/O, stage timing / memory reporting, fork-based parallel map."""

from __future__ import annotations

import multiprocessing as mp
import os
import resource
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
import psutil

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path: str) -> pd.DataFrame:
    """Read a challenge TSV as all-string columns (empty cells stay '')."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                       engine="pyarrow")


def rss_gb() -> float:
    return psutil.Process().memory_info().rss / 1e9


def peak_rss_gb() -> float:
    """Peak RSS of this process and of the largest finished child (Linux: KiB)."""
    self_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    child_peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6
    return max(self_peak, child_peak)


STAGE_LOG: list[dict] = []


@contextmanager
def stage(name: str):
    """Time a pipeline stage and log wall time, current RSS and peak RSS."""
    t0 = time.perf_counter()
    print(f"[stage] {name} ...", flush=True)
    yield
    dt = time.perf_counter() - t0
    rec = {"stage": name, "seconds": round(dt, 2), "rss_gb": round(rss_gb(), 2),
           "peak_rss_gb": round(peak_rss_gb(), 2)}
    STAGE_LOG.append(rec)
    print(f"[stage] {name}: {dt:.1f}s | rss {rec['rss_gb']} GB | peak {rec['peak_rss_gb']} GB",
          flush=True)


# --- fork-based parallel map ----------------------------------------------------
# Large read-only objects (sparse matrices) are published through _SHARED before the
# pool forks, so workers read them copy-on-write instead of receiving pickled copies.
_SHARED: dict = {}


def fork_map(func, items, n_jobs: int, shared: dict | None = None):
    """Map `func` over `items` in forked workers; `shared` is visible via get_shared()."""
    _SHARED.clear()
    if shared:
        _SHARED.update(shared)
    try:
        if n_jobs <= 1 or len(items) <= 1:
            return [func(x) for x in items]
        ctx = mp.get_context("fork")
        with ctx.Pool(min(n_jobs, len(items))) as pool:
            return pool.map(func, items, chunksize=1)
    finally:
        _SHARED.clear()


def get_shared(key: str):
    return _SHARED[key]


def chunk_ranges(n: int, size: int) -> list[tuple[int, int]]:
    return [(i, min(i + size, n)) for i in range(0, n, size)]


def default_n_jobs() -> int:
    return max(1, (os.cpu_count() or 2) - 2)


def stable_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return df if len(df) <= n else df.sample(n=n, random_state=seed)


def as_int_array(x) -> np.ndarray:
    return np.asarray(x, dtype=np.int64)
