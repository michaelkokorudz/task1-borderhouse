"""
cross_asset.py
~~~~~~~~~~~~~~~~~~~~~~~
Generate a **lag‑k cross‑asset matrix** from *Integrated OFI*.

disclaimers
-----------
1. Demo CSV ships with a **single symbol (AAPL)** – the script will still
   run, but real cross‑impact needs ≥ 2 symbols.
2. Input **must** include an `ofi_integrated` column produced by
   `integrated_ofi.py`; raw best‑level OFI is not accepted here.
3. γ_ii (self‑lag) is usually excluded in Cont et al. regressions; set
   `DROP_SELF = False` if you want to keep it.

method
------
Shift each record forward by *k* minutes so the value at *t‑k* aligns
with row *t*, then pivot wide → rows = timestamps, cols = symbol_lagk.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

# ---------- CONFIG ----------------------------------------------------
SIGNAL_CSV   = "output_data/integrated_ofi.csv"   # tidy multi‑symbol input
SIGNAL_COL   = "ofi_integrated"                   # field to lag
LAGS_MIN     = [1, 2, 3, 5, 10, 20, 30]           # k values (minutes)
DROP_SELF    = True                               # drop γ_ii columns?
OUT_CSV      = "output_data/cross_asset_lags.csv" # wide output
# ----------------------------------------------------------------------

def _wide_lag(df: pd.DataFrame, col: str, k: int) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["timestamp"] += pd.Timedelta(minutes=k)                 # align t‑k → t
    return (df.pivot(index="timestamp", columns="symbol", values=col)
              .sort_index()
              .add_suffix(f"_lag{k}"))

def build_cross_asset(src: str = SIGNAL_CSV,
                      col: str = SIGNAL_COL,
                      ks      = LAGS_MIN,
                      drop_self: bool = DROP_SELF,
                      dst: str = OUT_CSV) -> None:
    df = pd.read_csv(src, usecols=["symbol", "timestamp", col])
    wide = pd.concat([_wide_lag(df, col, k) for k in ks], axis=1)

    if drop_self:
        symbols = df["symbol"].unique()
        wide = wide.drop(columns=[f"{s}_lag{k}" for s in symbols for k in ks],
                         errors="ignore")

    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(dst)
    print(f"✓ cross‑asset matrix → {dst}  ({wide.shape[0]}×{wide.shape[1]})")

if __name__ == "__main__":
    build_cross_asset()
