"""
Flower Bagging Benchmark - Server (PoC legacy-compatible)
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
from flwr.server.strategy import FedXgbBagging

warnings.filterwarnings("ignore", category=DeprecationWarning)


def _tensor_nbytes(tensor: bytes) -> int:
    return int(getattr(tensor, "nbytes", len(tensor)))


def _append_server_profile(profile_path: Path, payload: dict[str, Any]) -> None:
    with profile_path.open("a", encoding="utf-8") as f:
        json.dump(payload, f)
        f.write("\n")


def _ensure_json_model_bytes(bytes_modello: bytes) -> bytes:
    """Normalizza i bytes del modello in JSON UTF-8 per FedXgbBagging."""
    # Converti a bytes se necessario
    if isinstance(bytes_modello, np.ndarray):
        bytes_modello = bytes_modello.tobytes()
    elif not isinstance(bytes_modello, bytes):
        bytes_modello = bytes(bytes_modello)

    # In modalità NumPyClient i tensors arrivano serializzati come .npy
    if bytes_modello.startswith(b"\x93NUMPY"):
        with io.BytesIO(bytes_modello) as bio:
            array = np.load(bio, allow_pickle=False)
        if isinstance(array, np.ndarray):
            bytes_modello = array.astype(np.uint8, copy=False).tobytes()

    try:
        json.loads(bytearray(bytes_modello))
        return bytes_modello
    except Exception:
        bst = xgb.Booster()
        try:
            bst.load_model(bytearray(bytes_modello))
        except Exception:
            raise

        try:
            raw_bytes = bst.save_raw(raw_format="json")
            json.loads(bytes(raw_bytes).decode("utf-8"))
            return bytes(raw_bytes)
        except Exception:
            percorso_tmp = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                    percorso_tmp = tmp.name
                bst.save_model(percorso_tmp)
                with open(percorso_tmp, "rb") as f:
                    raw_bytes = f.read()
                json.loads(raw_bytes.decode("utf-8"))
                return raw_bytes
            finally:
                if percorso_tmp and os.path.exists(percorso_tmp):
                    os.remove(percorso_tmp)


def _ensure_npy_tensor_bytes(bytes_modello: bytes) -> bytes:
    """Converte bytes modello in tensor .npy uint8 compatibile NumPyClient."""
    if isinstance(bytes_modello, np.ndarray):
        bytes_modello = bytes_modello.tobytes()
    elif not isinstance(bytes_modello, bytes):
        bytes_modello = bytes(bytes_modello)

    if bytes_modello.startswith(b"\x93NUMPY"):
        return bytes_modello

    array = np.frombuffer(bytes_modello, dtype=np.uint8)
    with io.BytesIO() as bio:
        np.save(bio, array, allow_pickle=False)
        return bio.getvalue()


class RobustFedXgbBagging(FedXgbBagging):
    """FedXgbBagging con normalizzazione payload modello lato server."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def configure_fit(self, server_round, parameters, client_manager):
        start_total = time.perf_counter()
        # t1: Server invia configurazione ai client (timestamp assoluto)
        timestamp_t1_send = time.time()
        print(f"[TIMING_SERVER] R{server_round} t1_send={timestamp_t1_send:.6f} (server invia config)", flush=True)

        start_super = time.perf_counter()
        istruzioni_fit = super().configure_fit(server_round, parameters, client_manager)
        time_super_s = time.perf_counter() - start_super

        # metto il timestamp t1 alla configurazione di OGNI client 
        for _, istruzione_fit in istruzioni_fit:
            # metto il timestamp t1 che il server usa per sincronizzazione
            istruzione_fit.config["server-timestamp-t1"] = float(timestamp_t1_send)

        input_bytes_total = 0
        json_bytes_total = 0
        npy_bytes_total = 0
        for _, istruzione_fit in istruzioni_fit:
            tensori = istruzione_fit.parameters.tensors
            for indice, tensore in enumerate(tensori):
                input_bytes_total += _tensor_nbytes(tensore)
                tensore_json = _ensure_json_model_bytes(tensore)
                json_bytes_total += _tensor_nbytes(tensore_json)
                tensore_npy = _ensure_npy_tensor_bytes(tensore_json)
                npy_bytes_total += _tensor_nbytes(tensore_npy)
                tensori[indice] = tensore_npy  # RITORNO A NPY (richiesto da NumPyClient)

        _append_server_profile(
            Path(__file__).parent / "results" / "server_round_profile.jsonl",
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "timestamp_epoch": timestamp_t1_send, 
                "event": "configure_fit",
                "server_round": int(server_round),
                "num_clients": len(istruzioni_fit),
                "num_tensors": sum(len(istruzione_fit.parameters.tensors) for _, istruzione_fit in istruzioni_fit),
                "input_bytes_total": int(input_bytes_total),
                "json_bytes_total": int(json_bytes_total),
                "npy_bytes_total": int(npy_bytes_total),
                "time_super_s": float(time_super_s),
                "time_total_s": float(time.perf_counter() - start_total),
            },
        )

        return istruzioni_fit

    def aggregate_fit(self, server_round, results, failures):
        start_total = time.perf_counter()
        # t10: Server riceve risultati dai client (timestamp assoluto)
        timestamp_t10_recv = time.time()
        print(f"[TIMING_SERVER] R{server_round} t10_recv={timestamp_t10_recv:.6f} (server riceve {len(results)} risultati)", flush=True)

        input_bytes_total = 0
        json_bytes_total = 0
        for _, fit_res in results:
            tensors = fit_res.parameters.tensors
            for idx, tensor in enumerate(tensors):
                input_bytes_total += _tensor_nbytes(tensor)
                json_tensor = _ensure_json_model_bytes(tensor)
                json_bytes_total += _tensor_nbytes(json_tensor)
                tensors[idx] = json_tensor

        start_super = time.perf_counter()
        out = super().aggregate_fit(server_round, results, failures)
        time_super_s = time.perf_counter() - start_super

        _append_server_profile(
            Path(__file__).parent / "results" / "server_round_profile.jsonl",
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "timestamp_epoch": timestamp_t10_recv,
                "event": "aggregate_fit",
                "server_round": int(server_round),
                "num_results": len(results),
                "num_failures": len(failures) if failures is not None else 0,
                "num_tensors": sum(len(fit_res.parameters.tensors) for _, fit_res in results),
                "input_bytes_total": int(input_bytes_total),
                "json_bytes_total": int(json_bytes_total),
                "time_super_s": float(time_super_s),
                "time_total_s": float(time.perf_counter() - start_total),
            },
        )
        return out

    def configure_evaluate(self, server_round, parameters, client_manager):
        """Normalizza i tensori di evaluate in .npy uint8 compatibile NumPyClient."""
        evaluate_instructions = super().configure_evaluate(server_round, parameters, client_manager)

        for _, evaluate_ins in evaluate_instructions:
            tensors = evaluate_ins.parameters.tensors
            for idx, tensor in enumerate(tensors):
                json_tensor = _ensure_json_model_bytes(tensor)
                npy_tensor = _ensure_npy_tensor_bytes(json_tensor)
                tensors[idx] = npy_tensor

        return evaluate_instructions


def weighted_mae(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """MAE disattivata temporaneamente nel PoC."""
    _ = metrics
    return {"mae_disabled": 1.0}


def _extract_final_mae(storia: Any) -> float | None:
    """MAE disattivata temporaneamente nel PoC."""
    _ = storia
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Avvia il server Flower Bagging in modalita PoC")
    parser.add_argument("--server_address", type=str, required=True, help="Es: 0.0.0.0:8080")
    parser.add_argument("--num_rounds", type=int, default=10, help="Numero di round federati")
    parser.add_argument("--local_epochs", type=int, default=1, help="Round locali per client")
    parser.add_argument("--fraction_train", type=float, default=1.0, help="Frazione client per fit")
    parser.add_argument("--fraction_evaluate", type=float, default=0.0, help="Frazione client per evaluate (0=skip evaluate)")
    parser.add_argument("--min_fit_clients", type=int, default=2, help="Min client per fit")
    parser.add_argument("--min_evaluate_clients", type=int, default=0, help="Min client per evaluate (0=skip evaluate)")
    parser.add_argument("--min_available_clients", type=int, default=2, help="Client minimi connessi")
    parser.add_argument("--test_fraction", type=float, default=0.2, help="Validation split lato client")
    parser.add_argument("--objective", type=str, default="reg:squarederror", help="Objective XGBoost")
    parser.add_argument("--max_depth", type=int, default=6, help="Max depth XGBoost")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate XGBoost")
    parser.add_argument("--subsample", type=float, default=0.8, help="Subsample XGBoost")
    parser.add_argument("--colsample_bytree", type=float, default=0.8, help="Colsample bytree XGBoost")
    args = parser.parse_args()

    print("[BOOT] Avvio processo server Flower Bagging PoC legacy...", flush=True)
    print(f"[BOOT] start_server su {args.server_address} con {args.num_rounds} round...", flush=True)

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    # (output_dir / "server_round_profile.jsonl").write_text("", encoding="utf-8")

    def fit_config(server_round: int) -> dict[str, float | int | str]:
        return {
            "server-round": int(server_round),
            "local-epochs": int(args.local_epochs),
            "train-method": "bagging",
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

    frazione_valutazione = max(0.0, float(args.fraction_evaluate))

    strategy = RobustFedXgbBagging(
        fraction_fit=float(args.fraction_train),
        fraction_evaluate=frazione_valutazione,
        min_fit_clients=int(args.min_fit_clients),
        min_evaluate_clients=int(args.min_evaluate_clients),
        min_available_clients=int(args.min_available_clients),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config if frazione_valutazione > 0.0 else None,
        evaluate_metrics_aggregation_fn=weighted_mae,
    )

    inizio_totale = time.time()
    storia = fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=int(args.num_rounds)),
        strategy=strategy,
    )
    tempo_totale = time.time() - inizio_totale
    final_mae = _extract_final_mae(storia)

    params = {
        "objective": args.objective,
        "max_depth": int(args.max_depth),
        "learning_rate": float(args.learning_rate),
    }
    modello_globale = strategy.global_model
    if modello_globale:
        bst = xgb.Booster(params=params)
        bst.load_model(bytearray(modello_globale))
        model_path = output_dir / "final_model.json"
        bst.save_model(str(model_path))
        print(f"Modello salvato: {model_path}")
    else:
        print("Nessun modello globale prodotto (global_model vuoto).")

    metriche_timing = {
        "approach": "bagging",
        "num_rounds": int(args.num_rounds),
        "total_time": tempo_totale,
        "avg_round_time": (tempo_totale / args.num_rounds) if args.num_rounds > 0 else None,
        "final_mae": final_mae,
        "history_losses_distributed": storia.losses_distributed,
        "history_metrics_distributed": storia.metrics_distributed,
    }

    timing_path = output_dir / "timing_metrics.json"
    with timing_path.open("w", encoding="utf-8") as f:
        json.dump(metriche_timing, f, indent=2)

    print("\n" + "=" * 70)
    print("RIEPILOGO FINALE - BAGGING POC")
    print("=" * 70)
    print(f"⏱️  Tempo totale: {tempo_totale:.2f}s")
    if args.num_rounds > 0:
        print(f"⏱️  Tempo medio/round: {tempo_totale / args.num_rounds:.2f}s")
    print("📉 Final MAE: disattivata")
    print(f"📄 Timing salvato: {timing_path}")


if __name__ == "__main__":
    main()
