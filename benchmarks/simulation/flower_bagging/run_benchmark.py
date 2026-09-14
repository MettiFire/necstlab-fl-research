"""
Run Flower Bagging Benchmark
NECSTLab - Polimi LS2

Esegue benchmark completo di Flower Bagging

Questo script e' il punto di ingresso operativo per i test:
- legge la configurazione comune
- avvia Flower via CLI
- misura tempi/risorse
- salva i risultati in formato analizzabile
"""
import subprocess
import time
import yaml
from pathlib import Path
import sys
import pandas as pd

# Inserisco la root del progetto nel path per importare i moduli condivisi
# anche quando lo script viene lanciato da sottocartelle diverse.
sys.path.append(str(Path(__file__).resolve().parents[3]))

from utils import PerformanceMonitor, save_results


def resolve_data_dir() -> Path:
    """Trova il dataset in una delle posizioni supportate."""
    base_dir = Path(__file__).resolve().parents[3]
    candidates = [
        base_dir / "data" / "ml_ready_final_fed",
        base_dir / "data" / "ready_for_flwr",
        Path.home() / "fl_benchmark" / "data" / "ml_ready_final_fed",
        Path.home() / "fl_benchmark" / "data" / "ready_for_flwr",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def run_flower_bagging_benchmark(config_path: str = "../../config.yaml"):
    """
    Esegue un benchmark completo Flower Bagging e salva un record risultati.

    Perche' tengo tutto in una funzione:
    - posso riusarla in script/automation
    - centralizzo gestione errori e logging
    """
    
    # Leggo la configurazione una sola volta e la uso sia per stampa
    # che per compilare il report finale del run.
    config_file = Path(__file__).resolve().parents[3] / "config.yaml"
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 70)
    print("🌸 FLOWER BAGGING BENCHMARK")
    print("=" * 70)
    print(f"\nConfigurazione:")
    print(f"  - Num clients: {config['dataset']['num_clients']}")
    print(f"  - Num rounds: {config['federated']['num_rounds']}")
    print(f"  - Local epochs: {config['federated']['local_epochs']}")
    print(f"  - XGBoost max_depth: {config['xgboost']['max_depth']}")
    print()
    
    # Istanzio il monitor custom per raccogliere tempi e risorse
    # in modo uniforme rispetto agli altri benchmark.
    monitor = PerformanceMonitor()
    
    # Base directory del progetto: qui sono presenti pyproject.toml,
    # configurazione Flower e import path coerenti.
    base_dir = Path(__file__).resolve().parents[3]  # root del progetto
    
    # Scelgo di avviare Flower tramite CLI (`flwr run .`) cosi' uso
    # direttamente la definizione app nel pyproject.toml.
    cmd = [
        "flwr", "run", "."  # . = usa [tool.flwr.app] dal pyproject.toml
    ]
    
    # Se vuoi override dei parametri:
    # cmd += ["--run-config", f"num-server-rounds={config['federated']['num_rounds']}"]
    
    print("🚀 Avvio Flower simulation...")
    print(f"Command: flwr run . (from {base_dir})")
    print()
    
    # Avvio il timer end-to-end prima del subprocess, cosi' includo
    # startup framework, esecuzione round e teardown finale.
    monitor.start_timer('total_time')
    
    try:
        # Eseguo il benchmark come processo separato:
        # - `check=True` per intercettare errori reali di esecuzione
        # - `cwd=base_dir` per garantire che Flower risolva correttamente app/config
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=str(base_dir)  # Importante: esegui dalla root
        )
        
        # Chiudo timer e raccolgo snapshot risorse a fine run.
        total_time = monitor.stop_timer('total_time')
        monitor.record_resource_usage()
        
        print("\n" + "=" * 70)
        print("✅ BENCHMARK COMPLETATO")
        print("=" * 70)
        print(f"\n⏱️  Tempo totale: {total_time:.2f} secondi")
        
        # Estraggo lo stdout riga per riga per eventuale parsing metriche.
        # Questo blocco e' volutamente semplice: preferisco avere output grezzo
        # stampato ora e migliorare parser in step successivo.
        output_lines = result.stdout.split('\n')
        
        # Cerca MAE finale (da adattare al formato output Flower)
        final_mae = None
        for line in output_lines:
            if 'mae' in line.lower():
                # Parsing semplice, da migliorare
                print(f"   {line.strip()}")
        
        # Creo il record minimale comune a tutti i benchmark.
        results = {
            'approach': 'flower_bagging',
            'total_time_sec': total_time,
            'num_clients': config['dataset']['num_clients'],
            'num_rounds': config['federated']['num_rounds'],
            'local_epochs': config['federated']['local_epochs'],
            'final_mae': final_mae,
        }
        
        # Aggiungo metriche monitor (CPU/memoria/tempi aggregati) al record finale.
        results.update(monitor.get_summary())
        
        # Salvo in cartella results condivisa, cosi' il notebook di analisi
        # trova automaticamente i file senza path speciali.
        results_dir = Path(__file__).resolve().parents[3] / "results"
        save_results(results, str(results_dir), "flower_bagging")
        
        print(f"\n📊 Risultati salvati in {results_dir}")
        
        return results
        
    except subprocess.CalledProcessError as e:
        # Errore runtime del benchmark (config, dipendenze, crash Flower, ...).
        print(f"\n❌ Errore durante esecuzione:")
        print(e.stderr)
        return None
    except KeyboardInterrupt:
        # Gestisco Ctrl+C in modo pulito senza stacktrace rumoroso.
        print(f"\n⚠️ Benchmark interrotto dall'utente")
        return None


if __name__ == "__main__":
    print("\n🧪 Flower Bagging Benchmark - NECSTLab\n")
    
    # Controllo preliminare dati: preferisco fallire subito con messaggio chiaro
    # invece di far partire Flower e scoprire dopo che i CSV non esistono.
    data_dir = resolve_data_dir()
    
    if not data_dir.exists():
        print("⚠️  ATTENZIONE: Directory dati non trovata!")
        print(f"   Path: {data_dir}")
        print("\n   Crea symlink con:")
        print('   cd <root_progetto>/data')
        print('   ln -s /percorso/al/nuovo/dataset/ml_ready_final_fed ml_ready_final_fed')
        sys.exit(1)
    
    # Avvio benchmark completo.
    results = run_flower_bagging_benchmark()
    
    if results:
        print("\n✅ Benchmark completato con successo!")
    else:
        print("\n❌ Benchmark fallito")
