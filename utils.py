"""
Utility comuni per i benchmark di Federated Learning
NECSTLab - Polimi LS2
"""
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from typing import Tuple, Optional
import time
import psutil
import os
import json


class DataLoader:
    """Carica e preprocessa il dataset Garmin per FL"""
    
    def __init__(self, data_dir: str = "./data/ready_for_flwr"):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Directory dati non trovata: {self.data_dir}")
    
    def load_client_data(
        self, 
        client_id: int, 
        test_fraction: float = 0.2
    ) -> Tuple[xgb.DMatrix, xgb.DMatrix, int, int]:
        """
        Carica dati per un client specifico
        
        Args:
            client_id: ID del client (0-8)
            test_fraction: Frazione per validation split
            
        Returns:
            train_dmatrix, valid_dmatrix, num_train, num_val
        """
        data_file = self.data_dir / f"client_{client_id}.csv"
        
        if not data_file.exists():
            raise FileNotFoundError(f"Client data non trovato: {data_file}")
        
        # Carica CSV
        data = pd.read_csv(data_file)
        
        # Rimuovi colonne vuote
        data = data.dropna(axis=1, how='all')
        
        # Seleziona feature (escludi day, label, time_series)
        feature_cols = [
            col for col in data.columns 
            if col not in ['day', 'label'] 
            and not col.endswith('_time_series')
            and data[col].dtype in ['int64', 'float64']
        ]
        
        X = data[feature_cols]
        y = data['label']
        
        # Rimuovi righe con NaN
        valid_idx = X.notna().all(axis=1) & y.notna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Split temporale
        split_idx = int(len(X) * (1 - test_fraction))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Crea DMatrix
        train_dmatrix = xgb.DMatrix(X_train, label=y_train)
        valid_dmatrix = xgb.DMatrix(X_val, label=y_val)
        
        return train_dmatrix, valid_dmatrix, len(X_train), len(X_val)
    
    def load_test_data(self, test_file: str = "./data/x_test.csv") -> xgb.DMatrix:
        """Carica test set centralizzato"""
        test_path = Path(test_file)
        
        if not test_path.exists():
            raise FileNotFoundError(f"Test file non trovato: {test_path}")
        
        data = pd.read_csv(test_path)
        data = data.dropna(axis=1, how='all')
        
        feature_cols = [
            col for col in data.columns 
            if col not in ['day', 'label'] 
            and not col.endswith('_time_series')
            and data[col].dtype in ['int64', 'float64']
        ]
        
        X_test = data[feature_cols]
        
        # Se presente label nel test (per valutazione)
        if 'label' in data.columns:
            y_test = data['label']
            test_dmatrix = xgb.DMatrix(X_test, label=y_test)
        else:
            test_dmatrix = xgb.DMatrix(X_test)
        
        return test_dmatrix


class PerformanceMonitor:
    """Monitor timing e risorse durante benchmark"""
    
    def __init__(self):
        self.metrics = {
            'training_time': [],
            'communication_time': [],
            'aggregation_time': [],
            'memory_usage_mb': [],
            'cpu_percent': [],
        }
        self.start_times = {}
    
    def start_timer(self, name: str):
        """Inizia timer per una fase"""
        self.start_times[name] = time.time()
    
    def stop_timer(self, name: str) -> float:
        """Ferma timer e registra"""
        if name not in self.start_times:
            raise ValueError(f"Timer {name} non avviato")
        
        elapsed = time.time() - self.start_times[name]
        
        if name in self.metrics:
            self.metrics[name].append(elapsed)
        else:
            self.metrics[name] = [elapsed]
        
        del self.start_times[name]
        return elapsed
    
    def record_resource_usage(self):
        """Registra uso CPU e memoria"""
        process = psutil.Process(os.getpid())
        
        self.metrics['memory_usage_mb'].append(
            process.memory_info().rss / 1024 / 1024
        )
        self.metrics['cpu_percent'].append(
            process.cpu_percent(interval=0.1)
        )
    
    def get_summary(self) -> dict:
        """Ritorna summary statistiche"""
        summary = {}
        
        for metric, values in self.metrics.items():
            if len(values) > 0:
                summary[f"{metric}_mean"] = np.mean(values)
                summary[f"{metric}_std"] = np.std(values)
                summary[f"{metric}_total"] = np.sum(values)
        
        return summary
    
    def to_dataframe(self) -> pd.DataFrame:
        """Converti metriche in DataFrame"""
        max_len = max(len(v) for v in self.metrics.values() if len(v) > 0)
        
        # Pad con NaN per lunghezze diverse
        padded = {}
        for key, values in self.metrics.items():
            if len(values) < max_len:
                padded[key] = values + [np.nan] * (max_len - len(values))
            else:
                padded[key] = values
        
        return pd.DataFrame(padded)


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcola Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcola Root Mean Squared Error"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def save_results(
    results: dict, 
    output_path: str, 
    experiment_name: str
):
    """Salva risultati benchmark in CSV"""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Converti in DataFrame
    df = pd.DataFrame([results])
    df['experiment'] = experiment_name
    df['timestamp'] = pd.Timestamp.now()
    
    # Salva
    output_file = output_dir / f"{experiment_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(output_file, index=False)
    
    print(f"✅ Risultati salvati: {output_file}")
    return output_file


def append_client_round_metric(
    approach: str,
    client_id: int,
    round_number: int,
    metric_row: dict,
    output_dir: str = "./results/structured_metrics",
):
    """Append di una metrica round-level per client in formato JSONL.

    Scrive una riga per round in un file dedicato al singolo client, evitando
    conflitti di scrittura quando i client sono in parallelo.
    """

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "approach": approach,
        "client_id": int(client_id),
        "round": int(round_number),
    }
    payload.update(metric_row)

    output_file = out_dir / f"{approach}_client_{client_id}.jsonl"
    with output_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


if __name__ == "__main__":
    # Test data loader
    print("🧪 Test DataLoader...")
    
    try:
        loader = DataLoader()
        train_dm, val_dm, n_train, n_val = loader.load_client_data(client_id=0)
        
        print(f"✅ Client 0 caricato:")
        print(f"   - Training samples: {n_train}")
        print(f"   - Validation samples: {n_val}")
        print(f"   - Features: {train_dm.num_col()}")
        
    except FileNotFoundError as e:
        print(f"⚠️ {e}")
        print("   Crea symlink ai dati prima di testare!")
