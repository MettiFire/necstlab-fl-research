"""
Flower Bagging Benchmark - Client (PoC legacy-compatible)
NECSTLab - Polimi LS2

Client Flower compatibile con API legacy (`start_client` + `NumPyClient`) e
con logging round-level per tempi/bytes in results/structured_metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

import flwr as fl
import numpy as np
import xgboost as xgb

sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import DataLoader, append_client_round_metric

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# questa funzione è un workaround robusto per serializzare booster XGBoost in JSON UTF-8 compatibile con FedXgbBagging, gestendo potenziali incompatibilità tra versioni XGBoost e formati raw. Il client si aspetta sempre un JSON UTF-8 valido, quindi questa funzione garantisce la compatibilità cross-versione.
def _serialize_booster_json(bst: xgb.Booster) -> bytes:
    """Serializza il booster in JSON UTF-8 compatibile con FedXgbBagging."""
    try:
        raw = bst.save_raw(raw_format="json")
        json.loads(bytes(raw).decode("utf-8"))
        return bytes(raw)
    except Exception:
        # Fallback robusto cross-version XGBoost: salva/riapri JSON da file.
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


# esegue il boosting locale e ritorna l'update (bagging o modello completo) insieme al tempo di update. Se train_method è "bagging", ritorna solo le nuove iterazioni; altrimenti, ritorna il booster completo aggiornato.
def _local_boost(
    bst_input: xgb.Booster,
    num_local_round: int,
    train_dmatrix: xgb.DMatrix,
    train_method: str,
) -> tuple[xgb.Booster, float, float]:
    """Esegue boosting locale e ritorna modello, tempo update e tempo slicing."""
    start_update = time.perf_counter()
    for _ in range(num_local_round):
        bst_input.update(train_dmatrix, bst_input.num_boosted_rounds())
    update_time = time.perf_counter() - start_update

    if train_method == "bagging":
        start_slice = time.perf_counter()
        # ✅ FIX: Ritorna il booster intero, NON fare slicing
        sliced = bst_input  
        slice_time = time.perf_counter() - start_slice
        return (sliced, update_time, slice_time)
    
    return bst_input, update_time, 0.0

# Client Flower compatibile con API legacy (`start_client` + `NumPyClient`) e con logging round-level per tempi/bytes in results/structured_metrics.
class XgbBaggingClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_id: int,
        local_epochs: int,
        test_fraction: float,
        objective: str,
        max_depth: int,
        learning_rate: float,
        subsample: float,
        colsample_bytree: float,
    ) -> None:
        self.client_id = client_id
        self.local_epochs = local_epochs
        self.test_fraction = test_fraction
        self.params = {
            "objective": objective,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
        }

    def get_parameters(self, config):
        _ = config
        return []

    def fit(self, parameters, config):
        start_total = time.time()
        start_config = time.perf_counter()

        global_round = int(config.get("server-round", 1))
        local_epochs = int(config.get("local-epochs", self.local_epochs))
        test_fraction = float(config.get("test-fraction", self.test_fraction))
        train_method = str(config.get("train-method", "bagging"))

        params = {
            "objective": str(config.get("objective", self.params["objective"])),
            "max_depth": int(config.get("max-depth", self.params["max_depth"])),
            "learning_rate": float(config.get("learning-rate", self.params["learning_rate"])),
            "subsample": float(config.get("subsample", self.params["subsample"])),
            "colsample_bytree": float(config.get("colsample-bytree", self.params["colsample_bytree"])),
            "nthread": 1,  # limita a 2 thread per client

        }
        config_time = time.perf_counter() - start_config

        start_load = time.time()
        # Risolvi path dati dalla root del progetto (indipendente da dove viene lanciato il client)
        project_root = Path(__file__).parent.parent.parent
        data_dir = str(project_root / "data" / "ready_for_flwr")

        start_loader = time.perf_counter()
        loader = DataLoader(data_dir=data_dir)
        loader_init_time = time.perf_counter() - start_loader

        start_data_load = time.perf_counter()
        train_dmatrix, _, num_train, _ = loader.load_client_data(
            client_id=self.client_id,
            test_fraction=test_fraction,
        )
        data_load_time = time.perf_counter() - start_data_load
        load_time = time.time() - start_load

        incoming_model_bytes = int(parameters[0].nbytes) if parameters else 0
        deserialize_time = 0.0
        run_id = str(config.get("run-id", "legacy"))
        booster_rounds_before = 0
        booster_rounds_after = 0
        local_update_time = 0.0
        local_slice_time = 0.0

        start_train = time.time()
        if global_round == 1 or not parameters or parameters[0].size == 0:
            train_branch = "cold_start_train"
            bst = xgb.train(params, train_dmatrix, num_boost_round=local_epochs)
            booster_rounds_after = int(bst.num_boosted_rounds())
        else:
            train_branch = "incremental_boost"
            start_deserialize = time.time()
            bst = xgb.Booster(params=params)
            bst.load_model(bytearray(parameters[0].tobytes()))
            deserialize_time = time.time() - start_deserialize

            booster_rounds_before = int(bst.num_boosted_rounds())
            bst, local_update_time, local_slice_time = _local_boost(
                bst, local_epochs, train_dmatrix, train_method
            )
            booster_rounds_after = int(bst.num_boosted_rounds())
        train_time = time.time() - start_train

        start_serialize = time.time()
        local_model = _serialize_booster_json(bst)
        model_np = np.frombuffer(local_model, dtype=np.uint8)
        outgoing_model_bytes = int(model_np.nbytes)
        serialize_time = time.time() - start_serialize

        dmatrix_rows = int(train_dmatrix.num_row())
        dmatrix_cols = int(train_dmatrix.num_col())

        total_time = time.time() - start_total

        metrics = {
            "num-examples": int(num_train),
            "train_time": float(train_time),
            "load_time": float(load_time),
            "deserialize_time": float(deserialize_time),
            "serialize_time": float(serialize_time),
            "communication_time_proxy": float(deserialize_time + serialize_time),
            "bytes_received": int(incoming_model_bytes),
            "bytes_sent": int(outgoing_model_bytes),
            "total_time": float(total_time),
            # Profiling dettagliato per debug colli di bottiglia client-side
            "profile_training_branch": train_branch,
            "profile_time_spent_parsing_fit_config_seconds": float(config_time),
            "profile_time_spent_creating_dataloader_seconds": float(loader_init_time),
            "profile_time_spent_loading_client_data_seconds": float(data_load_time),
            "profile_time_spent_updating_booster_seconds": float(local_update_time),
            "profile_time_spent_slicing_bagging_update_seconds": float(local_slice_time),
            "profile_booster_rounds_before_local_fit": int(booster_rounds_before),
            "profile_booster_rounds_after_local_fit": int(booster_rounds_after),
            "profile_booster_rounds_delta_local_fit": int(booster_rounds_after - booster_rounds_before),
            "profile_training_dmatrix_num_rows": int(dmatrix_rows),
            "profile_training_dmatrix_num_columns": int(dmatrix_cols),
            "profile_requested_local_boosting_rounds": int(local_epochs),
        }

        append_client_round_metric(
            approach="flower_bagging",
            client_id=self.client_id,
            round_number=global_round,
            run_id=run_id,
            metric_row=metrics,
        )

        return [model_np], int(num_train), metrics

    def evaluate(self, parameters, config):
        test_fraction = float(config.get("test-fraction", self.test_fraction))
        params = {
            "objective": str(config.get("objective", self.params["objective"])),
            "max_depth": int(config.get("max-depth", self.params["max_depth"])),
            "learning_rate": float(config.get("learning-rate", self.params["learning_rate"])),
        }

        project_root = Path(__file__).parent.parent.parent
        data_dir = str(project_root / "data" / "ready_for_flwr")
        loader = DataLoader(data_dir=data_dir)
        _, valid_dmatrix, _, num_val = loader.load_client_data(
            client_id=self.client_id,
            test_fraction=test_fraction,
        )

        if not parameters or parameters[0].size == 0:
            return float("inf"), int(num_val), {"mae": float("inf"), "num-examples": int(num_val)}

        bst = xgb.Booster(params=params)
        bst.load_model(bytearray(parameters[0].tobytes()))

        y_pred = bst.predict(valid_dmatrix)
        y_true = valid_dmatrix.get_label()
        mae = float(np.mean(np.abs(y_true - y_pred)))

        return mae, int(num_val), {"mae": mae, "num-examples": int(num_val)}

# Entry point per avviare il client con configurazione da command-line, compatibile con `flwr run` e con logging dettagliato dei tempi/bytes in results/structured_metrics per ogni round.
def main() -> None:
    parser = argparse.ArgumentParser(description="Avvia client Flower Bagging PoC (legacy)")
    parser.add_argument("--server_address", type=str, required=True, help="Es: 127.0.0.1:8080")
    parser.add_argument("--client_id", type=int, required=True, help="ID numerico client")
    parser.add_argument("--local_epochs", type=int, default=1, help="Default local epochs")
    parser.add_argument("--test_fraction", type=float, default=0.2, help="Validation split")
    parser.add_argument("--objective", type=str, default="reg:squarederror", help="Objective XGBoost")
    parser.add_argument("--max_depth", type=int, default=6, help="Max depth XGBoost")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate XGBoost")
    parser.add_argument("--subsample", type=float, default=0.8, help="Subsample XGBoost")
    parser.add_argument("--colsample_bytree", type=float, default=0.8, help="Colsample bytree XGBoost")
    args = parser.parse_args()

    client = XgbBaggingClient(
        client_id=args.client_id,
        local_epochs=args.local_epochs,
        test_fraction=args.test_fraction,
        objective=args.objective,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
    )

    fl.client.start_client(
        server_address=args.server_address,
        client=client.to_client(),
    )


if __name__ == "__main__":
    main()
