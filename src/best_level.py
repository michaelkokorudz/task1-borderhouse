"""
best_level.py
~~~~~~~~~~~~~
Best‑level OFI in a nutshell
---------------------------
Bid side
    ↑ price  →  +size      (aggressive buy)
    = price  →  +Δsize     (passive add/cancel)
    ↓ price  →  −size      (bid pulled)

Ask side uses the opposite sign.  
Sum these signed volumes over each 60‑s bucket to get OFI (Cont et al., Eq 1).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from utils import load_lob, bucketize, BUCKET_MS


def _ofi_level0(df_sym: pd.DataFrame) -> pd.Series:
    """
    Return *minute‑aggregated* best‑level OFI for one symbol.

    parameters
    ----------
    df_sym : DataFrame
        Must contain ['bid_px_00', 'bid_sz_00', 'ask_px_00', 'ask_sz_00', 'bucket'].

    notes
    -----
    • We slice off the first event per symbol (no previous quote to compare).
    • Computation is fully vectorised with NumPy for speed.
    """
    # Extract arrays
    bid_px = df_sym["bid_px_00"].to_numpy()
    bid_q  = df_sym["bid_sz_00"].to_numpy()
    ask_px = df_sym["ask_px_00"].to_numpy()
    ask_q  = df_sym["ask_sz_00"].to_numpy()

    # Lagged views (prepend NaN so first diff is ignored)
    px_prev  = np.concatenate(([np.nan], bid_px[:-1]))
    q_prev   = np.concatenate(([np.nan], bid_q[:-1]))
    apx_prev = np.concatenate(([np.nan], ask_px[:-1]))
    aq_prev  = np.concatenate(([np.nan], ask_q[:-1]))

    # Bid‑side contribution
    ofb = np.where(
        bid_px > px_prev,  +bid_q,         # price improved
        np.where(bid_px == px_prev, bid_q - q_prev, -bid_q)  # unchanged / deteriorated
    )
    # Ask‑side contribution (note sign flip)
    ofa = np.where(
        ask_px > apx_prev, -ask_q,
        np.where(ask_px == apx_prev, ask_q - aq_prev, +ask_q)
    )

    ofi = ofb - ofa
    # Drop NaN from first event before grouping
    valid = ~np.isnan(ofi)
    bucket_idx = df_sym["bucket"].to_numpy()[valid]

    return pd.Series(ofi[valid], index=bucket_idx, name="ofi_best").groupby(level=0).sum()


def best_level_ofi_file(lob_csv_path: str, out_path: str = 'best_ofi.csv') -> None:
    """
    Load a raw first_25000_rows CSV, compute best-level OFI per symbol and minute,
    add a timestamp column, and write results to a CSV file.

    parameters
    ----------
    lob_csv_path :
        Path to raw LOB CSV file with required columns.
    out_path : 
        Path to output CSV file (default 'best_ofi.csv').
    """
    df = bucketize(load_lob(lob_csv_path))
    frames = []
    for symbol, group in df.groupby('symbol', sort=False):
        ofi_series = _ofi_level0(group)
        # convert bucket integer into human-readable timestamp at bucket end
        timestamps = pd.to_datetime(ofi_series.index * BUCKET_MS, unit='ms')
        df_out = pd.DataFrame({
            'symbol': symbol,
            'timestamp': timestamps,
            'ofi_best': ofi_series.values
        })
        frames.append(df_out)
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(out_path, index=False)
    print(f'Wrote best-level OFI CSV (with timestamps) to {out_path}')


if __name__ == '__main__':
    # Example call (no argparse):
    best_level_ofi_file('input_data/first_25000_rows.csv', 'output_data/best_ofi.csv')