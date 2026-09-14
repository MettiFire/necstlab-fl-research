#!/usr/bin/env python3
"""
NVIDIA FLARE XGBoost Benchmark Runner
NECSTLab - Polimi LS2

Esegue il benchmark NVIDIA FLARE con histogram-based aggregation
usando XGBHorizontalRecipe (approccio ufficiale)

- tengo i parametri XGBoost allineati ai benchmark Flower
- salvo metriche temporali in formato semplice per l'analisi comparativa
"""
import sys
import time
import json
from pathlib import Path

# aggiungo la root del progetto al path per importare utils
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils import DataLoader

from nvflare.app_opt.xgboost.data_loader import XGBDataLoader
from nvflare.app_opt.xgboost.recipes import XGBHorizontalRecipe
from nvflare.recipe import SimEnv


class SleepQualityDataLoader(XGBDataLoader):
    """
    - nvflare vuole un `XGBDataLoader` per ogni site
    - il mio dataset è già gestito da `utils.DataLoader`
    - qui faccio il bridge tra l'API NVFLARE e il formato dati del progetto
    """
    
    def __init__(self, data_dir: str = None, test_fraction: float = 0.2):
        """
        Inizializzo parametri e stato interno del loader.

        `data_dir` opzionale: se non passato, uso il percorso standard del progetto.
        `test_fraction`: quota usata da `DataLoader` per split train/validation.
        """
        super().__init__() 
        self._data_dir = data_dir # se None, `initialize` costruisce il path standard relativo alla root del progetto
        self._test_fraction = test_fraction 
        self._client_id = None 
        self._data_loader = None
        
    def initialize(self, fl_ctx=None, client_id=None, **kwargs):
        """
        Inizializzo il loader in modo compatibile con diverse versioni NVFLARE.

        Nota: a seconda della versione/runtime, NVFLARE può passare:
        - `client_id` diretto
        - oppure `fl_ctx` da cui estrarre il nome site
        """
        
        if client_id is not None:

            # NVFLARE può passare client_id come int o stringa tipo "site-4"
            if isinstance(client_id, str):
                if client_id.startswith("site-"):
                    # Converto da base-1 del nome site a base-0 usata nel dataset.
                    self._client_id = int(client_id.split("-")[1]) - 1
                else: # se è una stringa ma non in formato "site-X", provo a convertire direttamente (es. "4" -> 4)
                    self._client_id = int(client_id)
            else:
                self._client_id = int(client_id)

        elif fl_ctx is not None: # in alcune versioni NVFLARE, il client_id è disponibile solo come parte del contesto `fl_ctx` (es. `fl_ctx.get_identity_name()` restituisce "site-1", "site-2", ecc.)
            # estrai client ID dal nome del site (es. "site-1" → 0)
            site_name = fl_ctx.get_identity_name()
            self._client_id = int(site_name.split("-")[1]) - 1
        else:
            raise ValueError("SleepQualityDataLoader.initialize richiede fl_ctx o client_id") # se non riesco a identificare il client, non posso procedere con il caricamento dati specifico per site.
        
        # Se `data_dir` non è stato fornito, uso il percorso standard nel repo.
        if self._data_dir is None:
            base_dir = Path(__file__).parent
                    self._data_dir = str(base_dir.parent.parent / "data" / "ml_ready_final_fed")
        
        self._data_loader = DataLoader(data_dir=self._data_dir)
        

    
    def load_data(self):
        """
        Carico train/validation per il client corrente.

        Ritorno una tupla `(train_data, valid_data)` come richiesto da NVFLARE per il training federato XGBoost.
        """
        # Carico i dati specifici per il client identificato in `initialize`.
        train_data, valid_data, num_train, num_val = self._data_loader.load_client_data(
            client_id=self._client_id,
            test_fraction=self._test_fraction
        )
        
        # Log esplicito per verificare rapidamente mapping site->dataset e cardinalita'.
        print(f"  ✅ Site-{self._client_id + 1}: {num_train} train, {num_val} val")
        
        return train_data, valid_data


def main():
    """
    Entry point principale del benchmark NVFLARE.

    Flusso generale:
    1) definisco parametri XGBoost e impostazioni federate
    2) creo configurazione per ogni site/client
    3) avvio recipe NVFLARE in simulazione
    4) salvo timing e metadati utili al confronto con Flower
    """
    
    print("\n" + "="*70)
    print("NVIDIA FLARE + XGBoost Benchmark")
    print("Histogram-based Aggregation (Federated)")
    print("NECSTLab - Polimi LS2")
    print("="*70 + "\n")
    
    start_total = time.time()
    
    # Parametri XGBoost allineati a Flower, cosi' il confronto tra framework è il piu' equo possibile
    
    xgb_params = {
        "objective": "reg:squarederror",
        "max_depth": 6,
        "eta": 0.1,  # learning_rate in nvflare usa 'eta', 0.1 è un valore comune per bilanciare velocità di apprendimento e stabilità del training
        "subsample": 0.8, # 0.8 è un valore comune per ridurre overfitting e aumentare la diversità tra i modelli locali
        "colsample_bytree": 0.8, # 0.8 è un valore comune per ridurre overfitting e aumentare la diversità tra i modelli locali
        "tree_method": "hist",  # ← HISTOGRAM-BASED
        "nthread": 4, # uso 4 thread per il training locale, bilanciando velocità e risorse
        "eval_metric": "mae",
    }
    
    # Configurazione federata: stessi 9 client e 10 round dei benchmark Flower.
    num_clients = 9
    num_rounds = 10
    
    # Path dataset locale (relativo alla root del progetto).
    base_dir = Path(__file__).parent
    data_dir = base_dir.parent.parent / "data" / "ml_ready_final_fed"
    
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
        # Creo un data loader per site; il client_id effettivo viene associato
        # da NVFLARE in `initialize` del client.
        data_loader = SleepQualityDataLoader(
            data_dir=str(data_dir), 
            test_fraction=0.2 # stesso split di test usato nei benchmark Flower per coerenza
        )
        per_site_config[site_name] = {"data_loader": data_loader} 
    
    print()
    
    # Recipe ufficiale NVFLARE per FL orizzontale XGBoost.
    # Qui sono definite logica round, aggregazione e orchestrazione dei site.
    recipe = XGBHorizontalRecipe(
        name="fl_benchmark_nvidia_flare",
        min_clients=num_clients,
        num_rounds=num_rounds,
        use_gpus=False, # per confronto equo con Flower, uso CPU
        xgb_params=xgb_params, 
        per_site_config=per_site_config, # passo la configurazione specifica per ogni site, in questo caso solo il data loader, ma potrebbe essere estesa con parametri specifici per site se necessario
    )
    
    print(f"🚀 Avvio training federato histogram-based...")
    print()
    
    # eseguo in SimEnv con un thread per client. simEnv è l'ambiente di simulazione di NVFLARE 
    # XGBoost federato mantiene stato inter-round, quindi un mapping stabile client<->thread aiuta a evitare artefatti runtime.
    start_fl = time.time()
    clients = list(per_site_config.keys())
    env = SimEnv(clients=clients, num_threads=len(clients))
    
    try:
        # Lancio il training federato.
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
        
        # Salvo le metriche temporali e i dati principali in un file JSON nella cartella `results` del benchmark, per facilitare l'analisi comparativa con i benchmark Flower.
        base_dir = Path(__file__).parent
        results_dir = base_dir / "results"
        results_dir.mkdir(exist_ok=True) # creo la cartella se non esiste
        
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
        
        # Salvo le metriche temporali in un file JSON, cosi' il notebook di analisi comparativa può leggerle facilmente senza dover fare parsing dello stdout.
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
        # Gestione errore esplicita con traceback completo:
        # in fase sperimentale utile per debug rapido delle integrazioni NVFLARE
        print(f"\n❌ ERRORE durante l'esecuzione:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Exit code standard: 0 successo, 1 fallimento 
    success = main()
    sys.exit(0 if success else 1)
