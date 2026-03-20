"""
Flower Bagging Benchmark - Server
NECSTLab - Polimi LS2

In questo file gestisco l'orchestrazione lato server:
- inizializzazione strategia federata (FedXgbBagging)
- avvio dei round FL
- salvataggio del modello finale e delle metriche temporali

L'obiettivo e' misurare in modo trasparente quanto costa il training federato
end-to-end e avere un artefatto finale riproducibile.
"""
import numpy as np
import xgboost as xgb
from pathlib import Path
import sys
import warnings
import time
import json

# Aggiungo la root del progetto per importare moduli locali (`utils`, ecc.)
# quando l'app viene eseguita dalla CLI Flower.
sys.path.append(str(Path(__file__).parent.parent.parent))

from flwr.app import ArrayRecord, Context
from flwr.common.config import unflatten_dict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedXgbBagging

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Stampo un messaggio subito: in ambienti notebook/terminale aiuta a capire
# che il server e' partito e sta caricando dipendenze pesanti.
print("⏳ Caricamento Flower Bagging Server in corso...")
print("   Inizializzazione componenti... (può richiedere qualche secondo)")

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """
    Entry point principale del server Flower.

    Qui imposto la strategia bagging, avvio i round federati e salvo:
    - modello globale finale
    - metriche di timing per l'analisi comparativa
    """
    
    # Leggo i parametri principali dal run-config per mantenere l'esecuzione
    # completamente controllata dal file di configurazione.
    num_rounds = context.run_config.get("num-server-rounds", 10)
    fraction_train = context.run_config.get("fraction-train", 1.0)
    fraction_evaluate = context.run_config.get("fraction-evaluate", 1.0)
    
    # Parametri usati per ricostruire il booster finale e garantire coerenza
    # con il training lato client.
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
    
    # Inizializzo le metriche temporali che poi salvo su JSON.
    # Mi servono per confrontare approcci diversi sul costo runtime.
    timing_metrics = {
        "total_time": 0,
        "rounds": [],
        "approach": "bagging",
        "num_rounds": num_rounds
    }
    start_total = time.time()
    
    # Parto con un modello vuoto: in bagging il server aggrega i contributi
    # dei client round dopo round fino a costruire il modello globale.
    global_model = b""
    arrays = ArrayRecord([np.frombuffer(global_model, dtype=np.uint8)])
    
    # Strategia federata Flower specifica per XGBoost bagging.
    # fraction_train/fraction_evaluate controllano il campionamento client.
    strategy = FedXgbBagging(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
    )
    
    # Avvio il ciclo federato e misuro il tempo della sola fase FL,
    # separandolo dal tempo totale end-to-end.
    print(f"\n🚀 Starting Federated Learning (Bagging)...")
    start_fl = time.time()
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )
    fl_time = time.time() - start_fl
    
    # Ricostruisco il booster globale dai bytes restituiti dalla strategia
    # e salvo il modello per analisi/riproducibilita'.
    bst = xgb.Booster(params=params)
    global_model = bytearray(result.arrays["0"].numpy().tobytes())
    bst.load_model(global_model)
    
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    model_path = output_dir / "final_model.json"
    bst.save_model(str(model_path))
    
    # Compilo e serializzo le metriche temporali principali.
    # avg_round_time e' utile per confronti veloci tra approcci.
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
