#!/usr/bin/env python3
"""NVFlare POC metrics pipeline.

This script converts NVFlare artifacts to the same round-level JSONL schema used
by Flower in results/structured_metrics/runs/<run_id>, then produces CSV
summaries comparable with the existing Flower outputs.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STRUCTURED_DIR = PROJECT_ROOT / "results" / "structured_metrics"
RUNS_DIR = STRUCTURED_DIR / "runs"
SUMMARIES_DIR = STRUCTURED_DIR / "summaries"
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "results" / "nvflare_poc_workspace"
DEFAULT_APPROACH = "nvidia_flare_poc"


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _first_value(raw: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        if key in raw and raw[key] is not None:
            return raw[key]
    return default


def _infer_client_id(path: Path, site_one_indexed: bool) -> int | None:
    text = str(path)

    site_match = re.search(r"site[-_](\d+)", text)
    if site_match:
        site_id = int(site_match.group(1))
        return site_id - 1 if site_one_indexed else site_id

    client_match = re.search(r"client[-_](\d+)", text)
    if client_match:
        return int(client_match.group(1))

    return None


def _iter_json_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)

    elif path.suffix == ".json":
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return rows

        if isinstance(obj, dict):
            for key in ("rows", "metrics", "records", "events"):
                value = obj.get(key)
                if isinstance(value, list):
                    rows.extend(x for x in value if isinstance(x, dict))
                    return rows
            rows.append(obj)
        elif isinstance(obj, list):
            rows.extend(x for x in obj if isinstance(x, dict))

    return rows


def _normalize_row(
    raw: dict[str, Any],
    approach: str,
    run_id: str,
    fallback_client_id: int,
    seq_index: int,
) -> dict[str, Any]:
    round_number = _to_int(_first_value(raw, ["round", "current_round", "global_round", "fl_round"]), seq_index + 1)
    client_id = _to_int(
        _first_value(raw, ["client_id", "site_id", "site", "client", "clientIndex"]),
        fallback_client_id,
    )

    # Normalize site IDs if they are site-1..site-N style.
    if client_id > 0 and _first_value(raw, ["site_id", "site"]) is not None and "client_id" not in raw:
        client_id -= 1

    serialize_time = _to_float(_first_value(raw, ["serialize_time", "serialization_time"]))
    deserialize_time = _to_float(_first_value(raw, ["deserialize_time", "deserialization_time"]))
    communication_time_proxy = _to_float(
        _first_value(raw, ["communication_time_proxy", "communication_time", "comm_time"]),
        serialize_time + deserialize_time,
    )
    train_time = _to_float(_first_value(raw, ["train_time", "training_time", "fit_time"]))

    payload = {
        "timestamp": str(_first_value(raw, ["timestamp", "time", "datetime"], datetime.now(timezone.utc).isoformat())),
        "run_id": str(run_id),
        "approach": approach,
        "client_id": client_id,
        "round": round_number,
        "num-examples": _to_int(_first_value(raw, ["num-examples", "num_examples", "n_examples", "samples"])),
        "train_time": train_time,
        "load_time": _to_float(_first_value(raw, ["load_time", "data_load_time"])),
        "deserialize_time": deserialize_time,
        "serialize_time": serialize_time,
        "communication_time_proxy": communication_time_proxy,
        "bytes_received": _to_int(_first_value(raw, ["bytes_received", "rx_bytes", "received_bytes"])),
        "bytes_sent": _to_int(_first_value(raw, ["bytes_sent", "tx_bytes", "sent_bytes"])),
        "total_time": _to_float(
            _first_value(raw, ["total_time", "round_time", "elapsed_time"]),
            train_time + communication_time_proxy,
        ),

        "latenza_up_ms": _to_float(_first_value(raw, ["latenza_up_ms", "latency_up_ms"])),
        "latenza_down_ms": _to_float(_first_value(raw, ["latenza_down_ms", "latency_down_ms"])),
        # Timestamp fields for additional analysis
        "server_timestamp_t1": _to_float(_first_value(raw, ["server_timestamp_t1"])),
        "client_timestamp_t2": _to_float(_first_value(raw, ["client_timestamp_t2"])),
        "client_timestamp_t9": _to_float(_first_value(raw, ["client_timestamp_t9"])),
    }

    return payload


def extract_metrics(run_id: str, source_dir: Path, approach: str, site_one_indexed: bool) -> Path:
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    candidate_files = sorted(
        p
        for p in source_dir.rglob("*")
        if p.is_file()
        and p.suffix in {".json", ".jsonl"}
        and "structured_metrics" not in str(p)
        and "summaries" not in str(p)
    )

    rows_per_client: dict[int, list[dict[str, Any]]] = {}
    scanned = 0

    for path in candidate_files:
        scanned += 1
        rows = _iter_json_rows(path)
        if not rows:
            continue

        fallback_client = _infer_client_id(path, site_one_indexed)
        if fallback_client is None:
            fallback_client = 0

        for idx, row in enumerate(rows):
            normalized = _normalize_row(
                raw=row,
                approach=approach,
                run_id=run_id,
                fallback_client_id=fallback_client,
                seq_index=idx,
            )
            rows_per_client.setdefault(int(normalized["client_id"]), []).append(normalized)

    written = 0
    for client_id, rows in rows_per_client.items():
        output_file = run_dir / f"{approach}_client_{client_id}.jsonl"
        with output_file.open("a", encoding="utf-8") as f:
            for row in sorted(rows, key=lambda x: (int(x["round"]), str(x["timestamp"]))):
                f.write(json.dumps(row) + "\n")
                written += 1

    snapshot = {
        "run_id": run_id,
        "approach": approach,
        "source_dir": str(source_dir),
        "candidate_files_scanned": scanned,
        "clients_found": sorted(rows_per_client.keys()),
        "rows_written": written,
        "run_dir": str(run_dir),
    }
    snapshot_file = run_dir / f"{approach}_extract_snapshot.json"
    snapshot_file.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    print(f"Righe scritte: {written}")
    print(f"Run directory: {run_dir}")
    print(f"Snapshot: {snapshot_file}")
    return run_dir


def _load_run_rows(run_id: str, approach: str) -> pd.DataFrame:
    run_dir = RUNS_DIR / run_id
    paths = sorted(run_dir.glob(f"{approach}_client_*.jsonl"))
    rows: list[dict[str, Any]] = []

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


def summarize_run(run_id: str, approach: str, include_flower_comparison: bool) -> tuple[Path, Path]:
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)

    df = _load_run_rows(run_id, approach)
    if df.empty:
        raise RuntimeError(f"Nessuna metrica trovata per approach={approach} run_id={run_id}")

    round_summary = (
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
            latenza_up_ms_mean=("latenza_up_ms", "mean"),
            latenza_up_ms_max=("latenza_up_ms", "max"),
            latenza_down_ms_mean=("latenza_down_ms", "mean"),
            latenza_down_ms_max=("latenza_down_ms", "max"),
        )
        .sort_values("round")
    )
    round_summary["approach"] = approach
    round_summary["run_id"] = run_id

    approach_summary = pd.DataFrame(
        [
            {
                "approach": approach,
                "run_id": run_id,
                "rounds": int(df["round"].nunique()),
                "clients": int(df["client_id"].nunique()),
                "rows": int(len(df)),
                "bytes_sent_total": float(df["bytes_sent"].sum()),
                "bytes_received_total": float(df["bytes_received"].sum()),
                "comm_time_total": float(df["communication_time_proxy"].sum()),
                "comm_time_mean": float(df["communication_time_proxy"].mean()),
                "serialize_time_mean": float(df["serialize_time"].mean()),
                "deserialize_time_mean": float(df["deserialize_time"].mean()),
                "train_time_mean": float(df["train_time"].mean()),
                "latenza_up_ms_mean": float(df["latenza_up_ms"].mean()),
                "latenza_up_ms_max": float(df["latenza_up_ms"].max()),
                "latenza_down_ms_mean": float(df["latenza_down_ms"].mean()),
                "latenza_down_ms_max": float(df["latenza_down_ms"].max()),
            }
        ]
    )

    round_out = SUMMARIES_DIR / f"communication_round_summary_{approach}_{run_id}.csv"
    approach_out = SUMMARIES_DIR / f"communication_approach_summary_{approach}_{run_id}.csv"

    round_summary.to_csv(round_out, index=False)
    approach_summary.to_csv(approach_out, index=False)

    if include_flower_comparison:
        flower_summary = SUMMARIES_DIR / "communication_approach_summary.csv"
        if flower_summary.exists():
            flower_df = pd.read_csv(flower_summary)
            if "run_id" not in flower_df.columns:
                flower_df["run_id"] = "unknown"
            merged = pd.concat([flower_df, approach_summary], ignore_index=True)
            merged_out = SUMMARIES_DIR / f"communication_approach_summary_with_nvflare_{run_id}.csv"
            merged.to_csv(merged_out, index=False)
            print(f"Confronto Flower+NVFlare: {merged_out}")

    print(f"Round summary: {round_out}")
    print(f"Approach summary: {approach_out}")
    return round_out, approach_out


def main() -> int:
    parser = argparse.ArgumentParser(description="NVFlare metrics pipeline")
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=["extract", "summarize", "all"],
        help="Fase da eseguire",
    )
    parser.add_argument("--run-id", required=True, help="Run id target")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory sorgente da cui leggere artefatti/log NVFlare",
    )
    parser.add_argument(
        "--approach",
        type=str,
        default=DEFAULT_APPROACH,
        help="Nome approach per output compatibili con Flower",
    )
    parser.add_argument(
        "--site-one-indexed",
        action="store_true",
        help="Interpreta site-1..site-N come client_id 0..N-1",
    )
    parser.add_argument(
        "--no-flower-merge",
        action="store_true",
        help="Non genera il CSV merged con summary Flower",
    )

    args = parser.parse_args()

    if args.command in {"extract", "all"}:
        extract_metrics(
            run_id=args.run_id,
            source_dir=args.source_dir,
            approach=args.approach,
            site_one_indexed=args.site_one_indexed,
        )

    if args.command in {"summarize", "all"}:
        summarize_run(
            run_id=args.run_id,
            approach=args.approach,
            include_flower_comparison=not args.no_flower_merge,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
