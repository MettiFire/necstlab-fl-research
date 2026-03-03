"""
Flower Bagging Benchmark - Server
NECSTLab - Polimi LS2
"""
import numpy as np
import xgboost as xgb
from pathlib import Path
import sys
import warnings

sys.path.append(str(Path(__file__).parent.parent.parent))

from flwr.app import ArrayRecord, Context
from flwr.common.config import unflatten_dict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedXgbBagging

warnings.filterwarnings("ignore", category=DeprecationWarning)

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Server principale per Flower Bagging"""
    
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
    
    print(f"🌸 Flower Bagging Server")
    print(f"   Rounds: {num_rounds}")
    print(f"   Fraction train: {fraction_train}")
    print(f"   XGBoost params: {params}")
    
    # Modello iniziale vuoto
    global_model = b""
    arrays = ArrayRecord([np.frombuffer(global_model, dtype=np.uint8)])
    
    # Strategia Bagging
    strategy = FedXgbBagging(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
    )
    
    # Esegui FL
    print(f"\n🚀 Starting Federated Learning (Bagging)...")
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )
    
    # Salva modello finale
    bst = xgb.Booster(params=params)
    global_model = bytearray(result.arrays["0"].numpy().tobytes())
    bst.load_model(global_model)
    
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    model_path = output_dir / "final_model.json"
    bst.save_model(str(model_path))
    
    print(f"\n✅ Training completato!")
    print(f"   Modello salvato: {model_path}")
