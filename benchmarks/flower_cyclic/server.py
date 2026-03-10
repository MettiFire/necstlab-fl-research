"""
Flower Cyclic Benchmark - Server
NECSTLab - Polimi LS2
"""
import numpy as np
import xgboost as xgb
from pathlib import Path
import sys
import warnings
import time
import json

sys.path.append(str(Path(__file__).parent.parent.parent))

from flwr.app import ArrayRecord, Context
from flwr.common.config import unflatten_dict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedXgbCyclic

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Messaggio di loading iniziale
print("⏳ Caricamento Flower Cyclic Server in corso...")
print("   Inizializzazione componenti... (può richiedere qualche secondo)")

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Server principale per Flower Cyclic"""
    
    # Configurazione
    num_rounds = context.run_config.get("num-server-rounds", 10)
    fraction_train = context.run_config.get("fraction-train", 1.0)
    fraction_evaluate = context.run_config.get("fraction-evaluate", 1.0)
    
    # Parametri XGBoost
    cfg = unflatten_dict(context.run_config)
    params = {
        "objective": cfg.get("objective", "reg:squarederror"),
        "max_depth": cfg.get("max-depth", 6),
        "learning_rate": cfg.get("learning-rate", 0.1),
    }
    
    print(f"🔄 Flower Cyclic Server")
    print(f"   Rounds: {num_rounds}")
    print(f"   Fraction train: {fraction_train}")
    print(f"   XGBoost params: {params}")
    
    # Tracking temporale
    timing_metrics = {
        "total_time": 0,
        "rounds": [],
        "approach": "cyclic",
        "num_rounds": num_rounds
    }
    start_total = time.time()
    
    # Modello iniziale vuoto
    global_model = b""
    arrays = ArrayRecord([np.frombuffer(global_model, dtype=np.uint8)])
    
    # Strategia Cyclic
    strategy = FedXgbCyclic(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
    )
    
    # Esegui FL
    print(f"\n🚀 Starting Federated Learning (Cyclic)...")
    start_fl = time.time()
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )
    fl_time = time.time() - start_fl
    
    # Salva modello finale
    bst = xgb.Booster(params=params)
    global_model = bytearray(result.arrays["0"].numpy().tobytes())
    bst.load_model(global_model)
    
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    model_path = output_dir / "final_model.json"
    bst.save_model(str(model_path))
    
    # Salva metriche temporali
    timing_metrics["total_time"] = time.time() - start_total
    timing_metrics["fl_time"] = fl_time
    timing_metrics["avg_round_time"] = fl_time / num_rounds
    
    timing_path = output_dir / "timing_metrics.json"
    with open(timing_path, 'w') as f:
        json.dump(timing_metrics, f, indent=2)
    
    print(f"\n✅ Training completato!")
    print(f"   Modello salvato: {model_path}")
    print(f"   ⏱️  Tempo totale: {timing_metrics['total_time']:.2f}s")
    print(f"   ⏱️  Tempo FL: {fl_time:.2f}s")
    print(f"   ⏱️  Tempo medio/round: {timing_metrics['avg_round_time']:.2f}s")
