"""
Used to better visualize the LOB and what data was being worked with and understand the levels.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_depth_snapshot(row):
    ts = row["ts_event"]

    bids = [(lvl, row[f"bid_px_{lvl:02d}"], row[f"bid_sz_{lvl:02d}"]) for lvl in range(10)]
    asks = [(lvl, row[f"ask_px_{lvl:02d}"], row[f"ask_sz_{lvl:02d}"]) for lvl in range(10)]

    # best-level (level 0) prices
    best_bid = bids[0][1]
    best_ask = asks[0][1]
    width = 0.009 # width was selected for better visual clarity on bar chart

    fig, ax = plt.subplots(figsize=(10, 5))
    offset = max(s for _, _, s in bids+asks) * 0.02

    # plot bids in blue and annotate levels
    for lvl, price, size in bids:
        ax.bar(price, size, width=width, color='blue', label='Bid' if lvl == 0 else "")
        ax.text(price, size+offset, f"Lvl{lvl}", ha='center', va='bottom', color='blue')

    # plot asks in red and annotate levels
    for lvl, price, size in asks:
        ax.bar(price, size, width=width, color='red', label='Ask' if lvl == 0 else "")
        ax.text(price, size+offset, f"Lvl{lvl}", ha='center', va='bottom', color='red')

    #mid-price from level 0
    mid_price = (best_bid + best_ask) / 2
    ax.axvline(mid_price, linestyle='--', color='gray', label='Mid-price')

    ax.set_xlabel("Price")
    ax.set_ylabel("Shares resting at that price")
    ax.set_title(f"LOB Snapshot @ {ts}")
    ax.legend()
    plt.tight_layout()
    plt.show()

def main():
    df = pd.read_csv(Path("data") / "first_25000_rows.csv", parse_dates=["ts_event"])
    plot_depth_snapshot(df.iloc[0])

if __name__ == "__main__":
    main()
