#!/usr/bin/env python3
"""NVFlare Client Executor with XGBoost training and timing.

Receives task at t2, executes training (deserialize, load, train, serialize),
exits at t9. Sends back metrics with timestamps for latency calculation.
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

from nvflare.apis.executor import Executor
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable


def _add_repo_root_to_path() -> None:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "utils.py").exists():
            sys.path.insert(0, str(parent))
            return


_add_repo_root_to_path()

from utils import DataLoader, append_client_round_metric


class XGBExecutor(Executor):
    """XGBoost FedAvg Executor with timing instrumentation."""

    def __init__(
        self,
        client_id: int = 0,
        local_epochs: int = 1,
        test_fraction: float = 0.2,
    ):
        super().__init__()
        self.client_id = client_id
        self.local_epochs = local_epochs
        self.test_fraction = test_fraction
        self.xgb_params = {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "nthread": 1,
        }
        self.data_loader = None

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal) -> Shareable:
        round_num = shareable.get("round", 1)

        timestamp_t2_recv = time.time()
        self.log_info(f"[TIMING_CLIENT] R{round_num} C{self.client_id} t2_recv={timestamp_t2_recv:.6f}")
        inizio_totale = time.time()

        timestamp_t1_send = shareable.get("server_timestamp_t1", 0)
        latenza_up_ms = 0.0
        if timestamp_t1_send > 0:
            latenza_up_ms = (timestamp_t2_recv - timestamp_t1_send) * 1000
            self.log_info(f"[TIMING_CLIENT] R{round_num} C{self.client_id} latenza_up={latenza_up_ms:.1f}ms")

        if self.data_loader is None:
            radice_progetto = Path(__file__).resolve()
            for parent in radice_progetto.parents:
                if (parent / "data" / "ml_ready_final_fed").exists():
                    percorso_dati = str(parent / "data" / "ml_ready_final_fed")
                    self.data_loader = DataLoader(data_dir=percorso_dati)
                    break
            if self.data_loader is None:
                raise FileNotFoundError("Directory data/ml_ready_final_fed non trovata")

        inizio_caricamento = time.perf_counter()
        matrice_train, _, numero_campioni_train, _ = self.data_loader.load_client_data(
            client_id=self.client_id,
            test_fraction=self.test_fraction,
        )
        tempo_caricamento = time.perf_counter() - inizio_caricamento
        timestamp_t4_data_loaded = time.time()

        tempo_deserializzazione = 0.0
        timestamp_t3_deserialized = timestamp_t4_data_loaded
        byte_modello_in_input = 0

        if round_num == 1:
            booster = None
        else:
            global_model = shareable.get("global_model", b"")
            if global_model and len(global_model) > 0:
                inizio_deserializzazione = time.perf_counter()
                byte_modello_in_input = len(global_model)
                booster = xgb.Booster(params=self.xgb_params)
                booster.load_model(bytearray(global_model))
                tempo_deserializzazione = time.perf_counter() - inizio_deserializzazione
                timestamp_t3_deserialized = time.time()
            else:
                booster = None

        timestamp_t5_training_start = time.time()
        inizio_training = time.perf_counter()

        if round_num == 1 or booster is None:
            booster = xgb.train(self.xgb_params, matrice_train, num_boost_round=self.local_epochs)
        else:
            for _ in range(self.local_epochs):
                booster.update(matrice_train, booster.num_boosted_rounds())

        tempo_training = time.perf_counter() - inizio_training

        timestamp_t6_training_end = time.time()
        timestamp_t7_serialization_start = time.time()
        inizio_serializzazione = time.perf_counter()

        modello_serializzato = self._serialize_booster(booster)
        byte_modello_out = len(modello_serializzato)
        tempo_serializzazione = time.perf_counter() - inizio_serializzazione

        timestamp_t8_serialization_end = time.time()
        timestamp_t9_exit = time.time()

        tempo_totale = time.time() - inizio_totale

        metriche = {
            "round": round_num,
            "client_id": self.client_id,
            "num-examples": int(numero_campioni_train),
            "train_time": float(tempo_training),
            "load_time": float(tempo_caricamento),
            "deserialize_time": float(tempo_deserializzazione),
            "serialize_time": float(tempo_serializzazione),
            "communication_time_proxy": float(tempo_deserializzazione + tempo_serializzazione),
            "bytes_received": int(byte_modello_in_input),
            "bytes_sent": int(byte_modello_out),
            "total_time": float(tempo_totale),
            "server_timestamp_t1": float(timestamp_t1_send),
            "client_timestamp_t2": float(timestamp_t2_recv),
            "client_timestamp_t3": float(timestamp_t3_deserialized),
            "client_timestamp_t4": float(timestamp_t4_data_loaded),
            "client_timestamp_t5": float(timestamp_t5_training_start),
            "client_timestamp_t6": float(timestamp_t6_training_end),
            "client_timestamp_t7": float(timestamp_t7_serialization_start),
            "client_timestamp_t8": float(timestamp_t8_serialization_end),
            "client_timestamp_t9": float(timestamp_t9_exit),
            "latenza_up_ms": float(latenza_up_ms),
        }

        self._save_local_metrics(metriche)

        result = Shareable()
        result["model"] = modello_serializzato
        result["metrics"] = metriche
        result.update(metriche)

        return result

    def _serialize_booster(self, bst: xgb.Booster) -> bytes:
        try:
            raw = bst.save_raw(raw_format="json")
            return bytes(raw)
        except Exception:
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                    tmp_path = tmp.name
                bst.save_model(tmp_path)
                with open(tmp_path, "rb") as f:
                    raw = f.read()
                return raw
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)

    def _save_local_metrics(self, metriche: dict) -> None:
        try:
            percorso_timing = Path(f"/tmp/timing_nvflare_client_{self.client_id}.json")
            round_num = metriche.get("round", 1)

            timing_data = {}
            if percorso_timing.exists():
                timing_data = json.loads(percorso_timing.read_text())

            timing_data[f"R{round_num}"] = metriche
            percorso_timing.write_text(json.dumps(timing_data, indent=2))
        except Exception as e:
            self.log_warning(f"Failed to save local metrics: {e}")