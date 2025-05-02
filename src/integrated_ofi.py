"""
integrated_ofi.py
~~~~~~~~~~~~~~~~~
10‑level OFI → single scalar per bucket.

Idea (Cont et al., Eq 4)
    x_t  : 10‑dim scaled OFI vector
    w₁   : PC‑1 of Cov(x_t)  ,  ‖w₁‖₁ = 1
    OFIᵢ(t) = w₁ᵀ·(x_t - μ)          # μ = level means
"""

import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from utils import load_lob, bucketize, BUCKET_MS
from multi_level import _ofi_levels   # builds 10‑col scaled‑OFI matrix

def _pc1(mat: pd.DataFrame):
    """return PC‑1 weight vector (L1‑normed) and its var %"""
    pca = PCA(n_components=1).fit(mat)           # PCA centres internally
    w1  = pca.components_[0]                     # length‑10
    w1 /= np.abs(w1).sum()                       # ‖w1‖₁ = 1
    return w1, pca.explained_variance_ratio_[0]

def integrated_ofi_file(src: str, dst: str = "integrated_ofi.csv"):
    df = bucketize(load_lob(src))                # add 60‑s bucket id
    out, ratios = [], []
    for sym, grp in df.groupby("symbol", sort=False, observed=True):
        x = _ofi_levels(grp)                     # rows = buckets
        w1, r = _pc1(x); ratios.append(r)
        y = (x - x.mean()).dot(w1)               # centre then project
        ts = pd.to_datetime(y.index * BUCKET_MS, unit="ms")
        out.append(pd.DataFrame({"symbol": sym,
                                 "timestamp": ts,
                                 "ofi_integrated": y.values}))
    pd.concat(out).to_csv(dst, index=False)
    print(f"✓ {dst}  | Avg PC1 var {np.mean(ratios)*100:.1f}%")

if __name__ == "__main__":
    integrated_ofi_file("input_data/first_25000_rows.csv", "output_data/integrated_ofi.csv")





