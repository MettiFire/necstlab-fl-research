#!/usr/bin/env python3
"""Generate a detailed FL timeline (t0..tN) for a given run.

Inputs:
- Client JSONL directory: results/structured_metrics/runs/<run_id>
- Server profile JSONL: benchmarks/<approach>_poc/results/server_round_profile.jsonl

Outputs:
- results/structured_metrics/timeline_<approach>_<run_id>.csv
- results/structured_metrics/timeline_<approach>_<run_id>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STRUCTURED_DIR = PROJECT_ROOT / "results" / "structured_metrics"
RUNS_DIR = STRUCTURED_DIR / "runs"


def _default_server_profile(approach: str) -> Path:
    if approach == "flower_bagging":
        return PROJECT_ROOT / "benchmarks" / "flower_bagging_poc" / "results" / "server_round_profile.jsonl"
    if approach == "flower_cyclic":
        return PROJECT_ROOT / "benchmarks" / "flower_cyclic_poc" / "results" / "server_round_profile.jsonl"
    raise ValueError(f"Unsupported approach: {approach}")


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_client_rows(run_id: str, approach: str) -> pd.DataFrame:
    run_dir = RUNS_DIR / run_id
    paths = sorted(run_dir.glob(f"{approach}_client_*.jsonl"))

    rows: list[dict] = []
    for path in paths:
        rows.extend(_load_jsonl(path))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    numeric_cols = [
        "round",
        "client_id",
        "train_time",
        "total_time",
        "load_time",
        "serialize_time",
        "deserialize_time",
        "communication_time_proxy",
        "bytes_received",
        "bytes_sent",
        "profile_time_spent_parsing_fit_config_seconds",
        "profile_time_spent_creating_dataloader_seconds",
        "profile_time_spent_loading_client_data_seconds",
        "profile_time_spent_updating_booster_seconds",
        "profile_time_spent_slicing_bagging_update_seconds",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values(["round", "client_id", "timestamp"]).reset_index(drop=True)


def _load_server_rows(server_profile: Path, run_id: str) -> pd.DataFrame:
    rows = _load_jsonl(server_profile)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "run_id" in df.columns:
        filtered = df[df["run_id"].astype(str) == str(run_id)].copy()
        if not filtered.empty:
            df = filtered

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if "server_round" in df.columns:
        df["server_round"] = pd.to_numeric(df["server_round"], errors="coerce")

    return df.sort_values(["timestamp"]).reset_index(drop=True)


def _build_timeline(run_id: str, approach: str, client_df: pd.DataFrame, server_df: pd.DataFrame) -> pd.DataFrame:
    timeline_rows: list[dict] = []

    # t0: boot server (if available)
    boot_df = server_df[server_df.get("event", "") == "boot"] if not server_df.empty else pd.DataFrame()
    if not boot_df.empty:
        boot_row = boot_df.iloc[0]
        timeline_rows.append(
            {
                "event": "server_boot",
                "timestamp": boot_row["timestamp"],
                "round": None,
                "details": f"address={boot_row.get('server_address', '')} rounds={boot_row.get('num_rounds', '')}",
            }
        )

    # Server-side per-round phases
    if not server_df.empty:
        cfg_df = server_df[server_df.get("event", "") == "configure_fit"]
        agg_df = server_df[server_df.get("event", "") == "aggregate_fit"]

        for _, row in cfg_df.iterrows():
            timeline_rows.append(
                {
                    "event": "server_configure_fit",
                    "timestamp": row["timestamp"],
                    "round": int(row.get("server_round")) if pd.notna(row.get("server_round")) else None,
                    "details": (
                        f"clients={int(row.get('num_clients', 0))} "
                        f"bytes_in={int(row.get('input_bytes_total', 0))} "
                        f"bytes_out={int(row.get('npy_bytes_total', 0))} "
                        f"time_total_s={float(row.get('time_total_s', 0.0)):.6f}"
                    ),
                }
            )

        for _, row in agg_df.iterrows():
            timeline_rows.append(
                {
                    "event": "server_aggregate_fit",
                    "timestamp": row["timestamp"],
                    "round": int(row.get("server_round")) if pd.notna(row.get("server_round")) else None,
                    "details": (
                        f"results={int(row.get('num_results', 0))} "
                        f"bytes_in={int(row.get('input_bytes_total', 0))} "
                        f"bytes_json={int(row.get('json_bytes_total', 0))} "
                        f"time_total_s={float(row.get('time_total_s', 0.0)):.6f}"
                    ),
                }
            )

    # Client-side per-round windows and aggregate metrics
    if not client_df.empty:
        grouped = client_df.groupby("round", as_index=False)
        for _, g in grouped:
            round_id = int(g["round"].iloc[0])
            start_ts = g["timestamp"].min()
            end_ts = g["timestamp"].max()

            timeline_rows.append(
                {
                    "event": "client_round_window_start",
                    "timestamp": start_ts,
                    "round": round_id,
                    "details": f"clients={g['client_id'].nunique()}",
                }
            )

            timeline_rows.append(
                {
                    "event": "client_round_window_end",
                    "timestamp": end_ts,
                    "round": round_id,
                    "details": (
                        f"update_booster_sum_s={g['profile_time_spent_updating_booster_seconds'].sum():.6f} "
                        f"update_booster_mean_s={g['profile_time_spent_updating_booster_seconds'].mean():.6f} "
                        f"comm_time_sum_s={g['communication_time_proxy'].sum():.6f} "
                        f"comm_time_mean_s={g['communication_time_proxy'].mean():.6f} "
                        f"bytes_sent_total={int(g['bytes_sent'].sum())} "
                        f"bytes_recv_total={int(g['bytes_received'].sum())}"
                    ),
                }
            )

    summary_df = server_df[server_df.get("event", "") == "summary"] if not server_df.empty else pd.DataFrame()
    if not summary_df.empty:
        s = summary_df.iloc[-1]
        timeline_rows.append(
            {
                "event": "server_summary",
                "timestamp": s["timestamp"],
                "round": None,
                "details": (
                    f"total_time_s={float(s.get('total_time_s', 0.0)):.6f} "
                    f"avg_round_time_s={float(s.get('avg_round_time_s', 0.0)):.6f}"
                ),
            }
        )

    timeline_df = pd.DataFrame(timeline_rows)
    if timeline_df.empty:
        return timeline_df

    timeline_df = timeline_df.sort_values("timestamp").reset_index(drop=True)
    t0 = timeline_df["timestamp"].iloc[0]
    timeline_df["elapsed_s"] = (timeline_df["timestamp"] - t0).dt.total_seconds()
    timeline_df.insert(0, "t_label", [f"t{i}" for i in range(len(timeline_df))])

    return timeline_df


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate detailed FL timeline for a run_id")
    parser.add_argument("--run-id", required=True, help="Run ID folder name under results/structured_metrics/runs")
    parser.add_argument(
        "--approach",
        required=True,
        choices=["flower_bagging", "flower_cyclic"],
        help="Approach name used in client JSONL filenames",
    )
    parser.add_argument(
        "--server-profile",
        default=None,
        help="Optional explicit path to server_round_profile.jsonl",
    )
    args = parser.parse_args()

    server_profile = Path(args.server_profile) if args.server_profile else _default_server_profile(args.approach)

    client_df = _load_client_rows(run_id=args.run_id, approach=args.approach)
    server_df = _load_server_rows(server_profile=server_profile, run_id=args.run_id)

    timeline_df = _build_timeline(
        run_id=args.run_id,
        approach=args.approach,
        client_df=client_df,
        server_df=server_df,
    )

    if timeline_df.empty:
        print("Nessun dato timeline trovato. Verifica run_id/approach e profilo server.")
        return 1

    STRUCTURED_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = STRUCTURED_DIR / f"timeline_{args.approach}_{args.run_id}.csv"
    out_json = STRUCTURED_DIR / f"timeline_{args.approach}_{args.run_id}.json"

    timeline_df.to_csv(out_csv, index=False)
    out_json.write_text(timeline_df.to_json(orient="records", date_format="iso", indent=2), encoding="utf-8")

    # Quick summary with communication metrics
    if not client_df.empty:
        comm_sum = float(client_df["communication_time_proxy"].sum())
        comm_mean = float(client_df["communication_time_proxy"].mean())
        upd_sum = float(client_df["profile_time_spent_updating_booster_seconds"].sum())
        upd_mean = float(client_df["profile_time_spent_updating_booster_seconds"].mean())
        print(
            "Summary client metrics: "
            f"comm_sum_s={comm_sum:.6f}, comm_mean_s={comm_mean:.6f}, "
            f"update_sum_s={upd_sum:.6f}, update_mean_s={upd_mean:.6f}"
        )

    print(f"Timeline CSV: {out_csv}")
    print(f"Timeline JSON: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
