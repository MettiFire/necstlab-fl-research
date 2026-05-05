"""
Flower Cyclic Benchmark - Server (PoC legacy-compatible)
NECSTLab - Polimi LS2

Avvio server Flower in modalita' PoC tramite API compat (`start_server`).
Questa versione e' compatibile con Flower 1.23 (no parametro `app=`).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import tempfile
import time
from datetime import datetime, timezone
import warnings
from pathlib import Path
from typing import Any

import flwr as fl
import numpy as np
import xgboost as xgb
from flwr.common import Metrics
from flwr.server.strategy import FedXgbCyclic

warnings.filterwarnings("ignore", category=DeprecationWarning)


def _ensure_json_model_bytes(model_bytes: bytes) -> bytes:
    """Normalizza i bytes modello in JSON UTF-8 per FedXgbCyclic."""
    # Converti a bytes se necessario
    if isinstance(model_bytes, np.ndarray):
        model_bytes = model_bytes.tobytes()
    elif not isinstance(model_bytes, bytes):
        model_bytes = bytes(model_bytes)
    
    # In modalità NumPyClient i tensors arrivano serializzati come .npy
    # (prefisso tipico: 0x93NUMPY). Li decodifico prima in bytes modello.
    if model_bytes.startswith(b"\x93NUMPY"):
        with io.BytesIO(model_bytes) as bio:
            arr = np.load(bio, allow_pickle=False)
        if isinstance(arr, np.ndarray):
            model_bytes = arr.astype(np.uint8, copy=False).tobytes()

    try:
        json.loads(bytearray(model_bytes))
        return model_bytes
    except Exception:
        bst = xgb.Booster()
        try:
            bst.load_model(bytearray(model_bytes))
        except Exception:
            raise

        try:
            raw = bst.save_raw(raw_format="json")
            json.loads(bytes(raw).decode("utf-8"))
            return bytes(raw)
        except Exception:
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                    tmp_path = tmp.name
                bst.save_model(tmp_path)
                with open(tmp_path, "rb") as f:
                    raw = f.read()
                json.loads(raw.decode("utf-8"))
                return raw
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)


def _ensure_npy_tensor_bytes(model_bytes: bytes) -> bytes:
    """Converte bytes modello in tensor .npy uint8 compatibile NumPyClient."""
    if isinstance(model_bytes, np.ndarray):
        model_bytes = model_bytes.tobytes()
    elif not isinstance(model_bytes, bytes):
        model_bytes = bytes(model_bytes)

    if model_bytes.startswith(b"\x93NUMPY"):
        return model_bytes

    arr = np.frombuffer(model_bytes, dtype=np.uint8)
    with io.BytesIO() as bio:
        np.save(bio, arr, allow_pickle=False)
        return bio.getvalue()


class RobustFedXgbCyclic(FedXgbCyclic):
    """FedXgbCyclic con normalizzazione payload modello lato server."""

    def __init__(self, *args, profile_path: Path | None = None, run_id: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.profile_path = profile_path
        self.run_id = run_id

    def _append_profile(self, payload: dict[str, Any]) -> None:
        if self.profile_path is None:
            return
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        with self.profile_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def configure_fit(self, server_round, parameters, client_manager):
        start_total = time.perf_counter()
        start_super = time.perf_counter()
        fit_instructions = super().configure_fit(server_round, parameters, client_manager)
        super_time = time.perf_counter() - start_super

        selection_time = 0.0

        # Selezione deterministica round-robin: garantisce copertura uniforme
        # dei client nei round (es. 9 round -> 9 client diversi).
        if fit_instructions:
            start_select = time.perf_counter()
            try:
                num_available = int(client_manager.num_available())
                if num_available > 0:
                    sampled = client_manager.sample(
                        num_clients=num_available,
                        min_num_clients=num_available,
                    )

                    def _cid_key(proxy):
                        cid = str(getattr(proxy, "cid", ""))
                        return (0, int(cid)) if cid.isdigit() else (1, cid)

                    ordered_clients = sorted(sampled, key=_cid_key)
                    selected_client = ordered_clients[(int(server_round) - 1) % len(ordered_clients)]
                    fit_instructions = [(selected_client, fit_instructions[0][1])]
                    print(
                        f"[INFO] Cyclic deterministic selection: round={server_round}, "
                        f"selected_client={getattr(selected_client, 'cid', 'unknown')}"
                    )
            except Exception as exc:
                # Fallback sicuro: in caso di errore usa la selezione strategy base.
                print(f"[WARN] Deterministic selection fallback: {type(exc).__name__}: {exc}")
            finally:
                selection_time = time.perf_counter() - start_select

        num_tensors = 0
        input_bytes = 0
        json_bytes_total = 0
        npy_bytes = 0
        json_norm_time = 0.0
        npy_wrap_time = 0.0

        for _, fit_ins in fit_instructions:
            tensors = fit_ins.parameters.tensors
            for idx, tensor in enumerate(tensors):
                num_tensors += 1
                input_bytes += len(tensor)

                start_json = time.perf_counter()
                json_tensor = _ensure_json_model_bytes(tensor)
                json_norm_time += time.perf_counter() - start_json

                start_npy = time.perf_counter()
                npy_tensor = _ensure_npy_tensor_bytes(json_tensor)
                npy_wrap_time += time.perf_counter() - start_npy

                json_bytes_total += len(json_tensor)
                npy_bytes += len(npy_tensor)
                tensors[idx] = npy_tensor

        total_time = time.perf_counter() - start_total
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "event": "configure_fit",
            "server_round": int(server_round),
            "num_clients": int(len(fit_instructions)),
            "num_tensors": int(num_tensors),
            "input_bytes_total": int(input_bytes),
            "json_bytes_total": int(json_bytes_total),
            "npy_bytes_total": int(npy_bytes),
            "time_super_s": float(super_time),
            "time_selection_s": float(selection_time),
            "time_json_norm_s": float(json_norm_time),
            "time_npy_wrap_s": float(npy_wrap_time),
            "time_total_s": float(total_time),
        }
        self._append_profile(payload)
        print(
            "[PROFILE][cyclic][configure_fit] "
            f"round={server_round} clients={len(fit_instructions)} tensors={num_tensors} "
            f"bytes_in={input_bytes} bytes_out={npy_bytes} total_s={total_time:.4f}",
            flush=True,
        )
        return fit_instructions

    def aggregate_fit(self, server_round, results, failures):
        start_total = time.perf_counter()

        num_tensors = 0
        input_bytes = 0
        json_bytes_total = 0
        json_norm_time = 0.0

        for _, fit_res in results:
            tensors = fit_res.parameters.tensors
            for idx, tensor in enumerate(tensors):
                num_tensors += 1
                input_bytes += len(tensor)

                start_json = time.perf_counter()
                json_tensor = _ensure_json_model_bytes(tensor)
                json_norm_time += time.perf_counter() - start_json

                json_bytes_total += len(json_tensor)
                tensors[idx] = json_tensor

        start_super = time.perf_counter()
        out = super().aggregate_fit(server_round, results, failures)
        super_time = time.perf_counter() - start_super
        total_time = time.perf_counter() - start_total

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "event": "aggregate_fit",
            "server_round": int(server_round),
            "num_results": int(len(results)),
            "num_failures": int(len(failures)),
            "num_tensors": int(num_tensors),
            "input_bytes_total": int(input_bytes),
            "json_bytes_total": int(json_bytes_total),
            "time_json_norm_s": float(json_norm_time),
            "time_super_s": float(super_time),
            "time_total_s": float(total_time),
        }
        self._append_profile(payload)
        print(
            "[PROFILE][cyclic][aggregate_fit] "
            f"round={server_round} results={len(results)} tensors={num_tensors} "
            f"bytes_in={input_bytes} bytes_json={json_bytes_total} total_s={total_time:.4f}",
            flush=True,
        )
        return out


def weighted_mae(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """Aggrega la MAE pesata sul numero di esempi."""
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {"mae": 0.0}

    weighted = 0.0
    for num_examples, m in metrics:
        weighted += float(m.get("mae", 0.0)) * num_examples
    return {"mae": weighted / total_examples}


def main() -> None:
    parser = argparse.ArgumentParser(description="Avvia il server Flower Cyclic in modalita PoC")
    parser.add_argument("--server_address", type=str, required=True, help="Es: 0.0.0.0:8080")
    parser.add_argument("--num_rounds", type=int, default=10, help="Numero di round federati")
    parser.add_argument("--local_epochs", type=int, default=1, help="Round locali per client")
    parser.add_argument("--fraction_train", type=float, default=1.0, help="Frazione client per fit")
    parser.add_argument("--fraction_evaluate", type=float, default=0.0, help="Frazione client per evaluate (0=skip evaluate)")
    parser.add_argument("--min_fit_clients", type=int, default=1, help="Min client per fit")
    parser.add_argument("--min_evaluate_clients", type=int, default=0, help="Min client per evaluate (0=skip evaluate)")
    parser.add_argument("--min_available_clients", type=int, default=9, help="Client minimi connessi")
    parser.add_argument("--test_fraction", type=float, default=0.2, help="Validation split lato client")
    parser.add_argument("--objective", type=str, default="reg:squarederror", help="Objective XGBoost")
    parser.add_argument("--max_depth", type=int, default=6, help="Max depth XGBoost")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate XGBoost")
    parser.add_argument("--subsample", type=float, default=0.8, help="Subsample XGBoost")
    parser.add_argument("--colsample_bytree", type=float, default=0.8, help="Colsample bytree XGBoost")
    args = parser.parse_args()

    print("[BOOT] Avvio processo server Flower Cyclic PoC legacy...", flush=True)
    print(f"[BOOT] start_server su {args.server_address} con {args.num_rounds} round...", flush=True)

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    profile_path = output_dir / "server_round_profile.jsonl"
    if profile_path.exists():
        profile_path.unlink()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    print(f"[BOOT] run_id={run_id}", flush=True)

    def append_server_event(event: str, **extra: Any) -> None:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "event": event,
        }
        payload.update(extra)
        with profile_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    append_server_event("boot", server_address=args.server_address, num_rounds=int(args.num_rounds))

    def fit_config(server_round: int) -> dict[str, float | int | str]:
        return {
            "server-round": int(server_round),
            "run-id": run_id,
            "local-epochs": int(args.local_epochs),
            "train-method": "cyclic",
            "test-fraction": float(args.test_fraction),
            "objective": args.objective,
            "max-depth": int(args.max_depth),
            "learning-rate": float(args.learning_rate),
            "subsample": float(args.subsample),
            "colsample-bytree": float(args.colsample_bytree),
        }

    def evaluate_config(server_round: int) -> dict[str, float | int | str]:
        _ = server_round
        return {
            "test-fraction": float(args.test_fraction),
            "objective": args.objective,
            "max-depth": int(args.max_depth),
            "learning-rate": float(args.learning_rate),
        }

    evaluate_fraction = max(0.0, float(args.fraction_evaluate))

    strategy = RobustFedXgbCyclic(
        fraction_fit=float(args.fraction_train),
        fraction_evaluate=evaluate_fraction,
        min_fit_clients=int(args.min_fit_clients),
        min_evaluate_clients=int(args.min_evaluate_clients),
        min_available_clients=int(args.min_available_clients),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config if evaluate_fraction > 0.0 else None,
        evaluate_metrics_aggregation_fn=weighted_mae,
        profile_path=profile_path,
        run_id=run_id,
    )

    start_total = time.time()
    history = fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=int(args.num_rounds)),
        strategy=strategy,
    )
    total_time = time.time() - start_total

    append_server_event(
        "summary",
        total_time_s=float(total_time),
        avg_round_time_s=(float(total_time) / int(args.num_rounds)) if int(args.num_rounds) > 0 else None,
    )

    params = {
        "objective": args.objective,
        "max_depth": int(args.max_depth),
        "learning_rate": float(args.learning_rate),
    }
    global_model = strategy.global_model
    if global_model:
        bst = xgb.Booster(params=params)
        bst.load_model(bytearray(global_model))
        model_path = output_dir / "final_model.json"
        bst.save_model(str(model_path))
        print(f"Modello salvato: {model_path}")
    else:
        print("Nessun modello globale prodotto (global_model vuoto).")

    timing_metrics = {
        "approach": "cyclic",
        "num_rounds": int(args.num_rounds),
        "total_time": total_time,
        "avg_round_time": (total_time / args.num_rounds) if args.num_rounds > 0 else None,
        "history_losses_distributed": history.losses_distributed,
        "history_metrics_distributed": history.metrics_distributed,
    }

    timing_path = output_dir / "timing_metrics.json"
    with timing_path.open("w", encoding="utf-8") as f:
        json.dump(timing_metrics, f, indent=2)

    print("\n✅ Training completato!")
    print(f"   ⏱️ Tempo totale: {total_time:.2f}s")
    if timing_metrics["avg_round_time"] is not None:
        print(f"   ⏱️ Tempo medio/round: {timing_metrics['avg_round_time']:.2f}s")


if __name__ == "__main__":
    main()
