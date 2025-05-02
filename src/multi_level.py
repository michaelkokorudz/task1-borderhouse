"""
multi_level.py
~~~~~~~~~~~~~~
Minute‑level OFI for depths 0‑9, scaled by
Q_{M,h} = (1/2M) Σ_m mean(bid_sz+ask_sz)   – Cont et al., Eq 3
"""

import numpy as np, pandas as pd
from utils import load_lob, bucketize, BUCKET_MS, LEVELS

def _ofi_levels(df_sym: pd.DataFrame) -> pd.DataFrame:
    """minute‑bucket DataFrame [ofi_lv0…9] (double‑sum scaled)."""
    bid_px = df_sym[[f"bid_px_{i:02d}" for i in range(LEVELS)]].to_numpy()
    bid_q  = df_sym[[f"bid_sz_{i:02d}" for i in range(LEVELS)]].to_numpy()
    ask_px = df_sym[[f"ask_px_{i:02d}" for i in range(LEVELS)]].to_numpy()
    ask_q  = df_sym[[f"ask_sz_{i:02d}" for i in range(LEVELS)]].to_numpy()

    px_prev,q_prev   = np.vstack(([bid_px[0]],bid_px[:-1])),np.vstack(([bid_q[0]],bid_q[:-1]))
    ap_prev,aq_prev  = np.vstack(([ask_px[0]],ask_px[:-1])),np.vstack(([ask_q[0]],ask_q[:-1]))

    ofb = np.where(bid_px>px_prev, bid_q,
          np.where(bid_px==px_prev, bid_q-q_prev, -bid_q))
    ofa = np.where(ask_px>ap_prev,-ask_q,
          np.where(ask_px==ap_prev, ask_q-aq_prev,  ask_q))
    ofi_raw = ofb - ofa                                           # events×levels

    raw = (pd.DataFrame(ofi_raw)
             .assign(bucket=df_sym["bucket"].values)
             .groupby("bucket", sort=False, observed=True).sum())

    depth = (pd.DataFrame(bid_q+ask_q)
               .assign(bucket=df_sym["bucket"].values)
               .groupby("bucket", sort=False, observed=True).mean())
    Q = depth.mean(axis=1) / 2.0                                  # Series (bucket)

    scaled = raw.div(Q, axis=0)
    scaled.columns = [f"ofi_lv{i}" for i in range(LEVELS)]
    return scaled

def multi_level_ofi_file(src: str, dst: str = "multi_ofi.csv"):
    df = bucketize(load_lob(src))
    frames = []
    for sym, grp in df.groupby("symbol", sort=False, observed=True):
        x  = _ofi_levels(grp)
        x["timestamp"] = pd.to_datetime(x.index * BUCKET_MS, unit="ms")
        x["symbol"]    = sym
        frames.append(x.reset_index(drop=True))
    pd.concat(frames)[["symbol","timestamp"]+ [f"ofi_lv{i}" for i in range(LEVELS)]].to_csv(dst, index=False)
    print(f"✓ wrote {dst}")

if __name__ == "__main__":
    multi_level_ofi_file("input_data/first_25000_rows.csv",
                         "output_data/multi_ofi.csv")
