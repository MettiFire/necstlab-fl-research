#!/usr/bin/env python3
"""
NVIDIA FLARE XGBoost Benchmark Runner
NECSTLab - Polimi LS2

Esegue il benchmark NVIDIA FLARE con histogram-based aggregation
usando XGBHorizontalRecipe (approccio ufficiale)
"""
import sys
import time
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils import DataLoader

from nvflare.app_opt.xgboost.data_loader import XGBDataLoader
from nvflare.app_opt.xgboost.recipes import XGBHorizontalRecipe
from nvflare.recipe import SimEnv


class SleepQualityDataLoader(XGBDataLoader):
    """Custom data loader per sleep quality dataset"""
    
    def __init__(self, data_dir: str = None, test_fraction: float = 0.2):
        """Inizializzazione con parametri opzionali (chiamato da NVFLARE)"""
        super().__init__()
        self._data_dir = data_dir
        self._test_fraction = test_fraction
        self._client_id = None
        self._data_loader = None
        
    def initialize(self, fl_ctx):
        """Chiamato da NVFLARE dopo __init__ per configurare il DataLoader"""
        # Estrai client ID dal nome del site (es. "site-1" → 0)
        site_name = fl_ctx.get_identity_name()
        self._client_id = int(site_name.split("-")[1]) - 1
        
        # Se data_dir non è stato fornito, usa il percorso di default
        if self._data_dir is None:
            base_dir = Path(__file__).parent
            self._data_dir = str(base_dir.parent.parent / "data" / "ready_for_flwr")
        
        # Inizializza DataLoader
        self._data_loader = DataLoader(data_dir=self._data_dir)
        
    def load_data(self):
        """Carica dati per il client specifico"""
        train_data, valid_data, num_train, num_val = self._data_loader.load_client_data(
            client_id=self._client_id,
            test_fraction=self._test_fraction
        )
        
        print(f"  ✅ Site-{self._client_id + 1}: {num_train} train, {num_val} val")
        
        return {
            "train": train_data,
            "valid": valid_data,
        }


def main():
    """Main entry point"""
    
    print("\n" + "="*70)
    print("NVIDIA FLARE + XGBoost Benchmark")
    print("Histogram-based Aggregation (Federated)")
    print("NECSTLab - Polimi LS2")
    print("="*70 + "\n")
    
    start_total = time.time()
    
    # Parametri XGBoost (matching Flower per confronto equo)
    xgb_params = {
        "objective": "reg:squarederror",
        "max_depth": 6,
        "eta": 0.1,  # learning_rate in nvflare usa 'eta'
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",  # ← HISTOGRAM-BASED (key difference!)
        "nthread": 4,
        "eval_metric": "mae",
    }
    
    # Configurazione per-site con data loaders
    num_clients = 9
    num_rounds = 10
    
    # Path dati (relativo a benchmarks/nvidia_flare/)
    base_dir = Path(__file__).parent
    data_dir = base_dir.parent.parent / "data" / "ready_for_flwr"
    
    print(f"📦 Configurazione Benchmark:")
    print(f"   • Clients: {num_clients}")
    print(f"   • Rounds: {num_rounds}")
    print(f"   • Aggregation: Histogram-based (gradient statistics)")
    print(f"   • Tree method: {xgb_params['tree_method']}")
    print()
    
    print("🔧 Caricamento dati per client:")
    per_site_config = {}
    for client_id in range(num_clients):
        site_name = f"site-{client_id + 1}"
        # DataLoader inizializzato con solo data_dir; client_id estratto in initialize()
        data_loader = SleepQualityDataLoader(
            data_dir=str(data_dir),
            test_fraction=0.2
        )
        per_site_config[site_name] = {"data_loader": data_loader}
    
    print()
    
    # Crea recipe con histogram-based aggregation
    recipe = XGBHorizontalRecipe(
        name="fl_benchmark_nvidia_flare",
        min_clients=num_clients,
        num_rounds=num_rounds,
        use_gpus=False,
        xgb_params=xgb_params,
        per_site_config=per_site_config,
    )
    
    print(f"🚀 Avvio training federato histogram-based...")
    print()
    
    # Esegui simulazione con thread dedicati per ogni client
    # (necessario per XGBoost federato che mantiene stato tra round)
    start_fl = time.time()
    clients = list(per_site_config.keys())
    env = SimEnv(clients=clients, num_threads=len(clients))
    
    try:
        run = recipe.execute(env)
        fl_time = time.time() - start_fl
        total_time = time.time() - start_total
        
        print("\n" + "="*70)
        print(f"✅ BENCHMARK COMPLETATO")
        print("="*70)
        print(f"⏱️  Tempo totale: {total_time:.2f}s")
        print(f"⏱️  Tempo FL: {fl_time:.2f}s")
        print(f"⏱️  Tempo medio per round: {fl_time/num_rounds:.2f}s")
        print()
        print("📊 Job Status:", run.get_status())
        print("📁 Workspace:", run.get_result())
        
        # Salva timing metrics
        base_dir = Path(__file__).parent
        results_dir = base_dir / "results"
        results_dir.mkdir(exist_ok=True)
        
        timing_metrics = {
            "total_time": total_time,
            "fl_time": fl_time,
            "avg_round_time": fl_time / num_rounds,
            "num_rounds": num_rounds,
            "num_clients": num_clients,
            "approach": "histogram_based_aggregation",
            "framework": "nvidia_flare",
            "tree_method": "hist"
        }
        
        timing_file = results_dir / "timing_metrics.json"
        with open(timing_file, 'w') as f:
            json.dump(timing_metrics, f, indent=2)
        
        print(f"\n✅ Timing salvato: {timing_file}")
        print()
        print("💡 Prossimi passi:")
        print(f"   1. Trova modello in: {run.get_result()}")
        print("   2. Valuta performance su test set")
        print("   3. Confronta con Flower Bagging e Cyclic")
        print("   4. Aggiorna notebook analisi comparativa")
        print()
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERRORE durante l'esecuzione:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
