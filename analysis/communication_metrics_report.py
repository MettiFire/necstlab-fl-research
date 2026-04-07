#!/usr/bin/env python3
"""Build communication/transmission report for Flower Bagging and Cyclic.

Input files are the client JSONL logs generated during training in
results/structured_metrics:
- flower_bagging_client_<id>.jsonl
- flower_cyclic_client_<id>.jsonl

The script creates:
- round-level and approach-level CSV summaries
- a compact multi-panel PNG with comparison plots
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STRUCTURED_DIR = PROJECT_ROOT / "results" / "structured_metrics"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"


def _load_approach_rows(approach: str) -> pd.DataFrame:
    paths = sorted(STRUCTURED_DIR.glob(f"{approach}_client_*.jsonl"))
    rows = []

    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    numeric_cols = [
        "round",
        "client_id",
        "bytes_received",
        "bytes_sent",
        "communication_time_proxy",
        "serialize_time",
        "deserialize_time",
        "train_time",
        "total_time",
        "num-examples",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values(["round", "client_id"]).reset_index(drop=True)


def _aggregate_round(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("round", as_index=False)
        .agg(
            clients=("client_id", "nunique"),
            bytes_sent_total=("bytes_sent", "sum"),
            bytes_received_total=("bytes_received", "sum"),
            comm_time_mean=("communication_time_proxy", "mean"),
            comm_time_sum=("communication_time_proxy", "sum"),
            serialize_time_mean=("serialize_time", "mean"),
            deserialize_time_mean=("deserialize_time", "mean"),
            train_time_mean=("train_time", "mean"),
            total_time_mean=("total_time", "mean"),
        )
    )


def _aggregate_approach(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "approach": str(df["approach"].iloc[0]),
                "rounds": int(df["round"].nunique()),
                "clients": int(df["client_id"].nunique()),
                "bytes_sent_total": float(df["bytes_sent"].sum()),
                "bytes_received_total": float(df["bytes_received"].sum()),
                "comm_time_total": float(df["communication_time_proxy"].sum()),
                "comm_time_mean": float(df["communication_time_proxy"].mean()),
                "serialize_time_mean": float(df["serialize_time"].mean()),
                "deserialize_time_mean": float(df["deserialize_time"].mean()),
                "train_time_mean": float(df["train_time"].mean()),
            }
        ]
    )


def _make_plot(round_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax1, ax2, ax3, ax4 = axes.flatten()

    for approach, g in round_df.groupby("approach"):
        ax1.plot(g["round"], g["bytes_sent_total"] / (1024 * 1024), marker="o", label=approach)
        ax2.plot(g["round"], g["bytes_received_total"] / (1024 * 1024), marker="o", label=approach)
        ax3.plot(g["round"], g["comm_time_mean"], marker="o", label=approach)

    ax1.set_title("Bytes Sent per Round (MB)")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("MB")
    ax1.grid(alpha=0.25)

    ax2.set_title("Bytes Received per Round (MB)")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("MB")
    ax2.grid(alpha=0.25)

    ax3.set_title("Communication Time Proxy Mean per Round (s)")
    ax3.set_xlabel("Round")
    ax3.set_ylabel("Seconds")
    ax3.grid(alpha=0.25)

    x = range(len(summary_df))
    ax4.bar([i - 0.18 for i in x], summary_df["bytes_sent_total"] / (1024 * 1024), width=0.36, label="sent")
    ax4.bar([i + 0.18 for i in x], summary_df["bytes_received_total"] / (1024 * 1024), width=0.36, label="received")
    ax4.set_xticks(list(x), [a.replace("flower_", "") for a in summary_df["approach"]])
    ax4.set_title("Total Transmission Volume (MB)")
    ax4.set_ylabel("MB")
    ax4.grid(axis="y", alpha=0.25)

    for ax in (ax1, ax2, ax3):
        ax.legend()
    ax4.legend()

    fig.suptitle("Flower Bagging vs Cyclic - Communication & Transmission", fontsize=14, y=0.98)
    fig.tight_layout()

    out_file = PLOTS_DIR / "communication_bagging_vs_cyclic.png"
    fig.savefig(out_file, dpi=180)
    print(f"Plot salvato in: {out_file}")


def main() -> int:
    STRUCTURED_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    bagging_df = _load_approach_rows("flower_bagging")
    cyclic_df = _load_approach_rows("flower_cyclic")

    available = [df for df in (bagging_df, cyclic_df) if not df.empty]
    if not available:
        print("Nessun file JSONL trovato in results/structured_metrics per flower_bagging o flower_cyclic.")
        return 1

    merged = pd.concat(available, ignore_index=True)

    round_frames = []
    summary_frames = []
    for approach, g in merged.groupby("approach"):
        round_df = _aggregate_round(g)
        round_df["approach"] = approach
        round_frames.append(round_df)
        summary_frames.append(_aggregate_approach(g))

    round_summary = pd.concat(round_frames, ignore_index=True).sort_values(["approach", "round"])
    approach_summary = pd.concat(summary_frames, ignore_index=True).sort_values("approach")

    round_summary_file = STRUCTURED_DIR / "communication_round_summary.csv"
    approach_summary_file = STRUCTURED_DIR / "communication_approach_summary.csv"
    snapshot_json = STRUCTURED_DIR / "latest_comm_snapshot.json"

    round_summary.to_csv(round_summary_file, index=False)
    approach_summary.to_csv(approach_summary_file, index=False)
    snapshot_json.write_text(
        json.dumps(
            {
                "round_summary_rows": len(round_summary),
                "approach_summary_rows": len(approach_summary),
                "approaches": sorted(approach_summary["approach"].tolist()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _make_plot(round_summary, approach_summary)

    print(f"CSV round-level: {round_summary_file}")
    print(f"CSV approach-level: {approach_summary_file}")
    print(f"Snapshot JSON: {snapshot_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
