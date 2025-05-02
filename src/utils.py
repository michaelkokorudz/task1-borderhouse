# utils.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

LEVELS = 10          # depth 0‑9
BUCKET_MS = 60_000   # 60 s → ms

def load_lob(csv_path: str | Path, *, utc: bool = True) -> pd.DataFrame:
    """read raw LOB, parse time, stable‑sort"""
    df = pd.read_csv(csv_path, dtype={"symbol": "category"})
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=utc)
    return df.sort_values("ts_event", kind="mergesort")

def bucketize(df: pd.DataFrame) -> pd.DataFrame:
    """add integer minute‑bucket column"""
    ns = df["ts_event"].astype("int64", copy=False)              # no .view warning
    df["bucket"] = (ns // (BUCKET_MS * 1_000_000)).astype("int64", copy=False)
    return df
