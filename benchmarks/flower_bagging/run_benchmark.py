"""
Run Flower Bagging Benchmark
NECSTLab - Polimi LS2

Esegue benchmark completo di Flower Bagging con profiling.
"""
import subprocess
import time
import yaml
from pathlib import Path
import sys
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import PerformanceMonitor, save_results


def run_flower_bagging_benchmark(config_path: str = "../../config.yaml"):
    """Esegue benchmark Flower Bagging"""
    
    # Carica configurazione
    with open(config_path, 'r') as f:
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
    
    # Performance monitor
    monitor = PerformanceMonitor()
    
    # Metodo: Usando CLI flwr con pyproject.toml config
    base_dir = Path(__file__).parent.parent.parent  # root del progetto
    
    # Usa config dal pyproject.toml (più semplice)
    cmd = [
        "flwr", "run", "."  # . = usa [tool.flwr.app] dal pyproject.toml
    ]
    
    # Se vuoi override dei parametri:
    # cmd += ["--run-config", f"num-server-rounds={config['federated']['num_rounds']}"]
    
    print("🚀 Avvio Flower simulation...")
    print(f"Command: flwr run . (from {base_dir})")
    print()
    
    # Esegui e monitora
    monitor.start_timer('total_time')
    
    try:
        # Run subprocess dalla root del progetto
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=str(base_dir)  # Importante: esegui dalla root
        )
        
        total_time = monitor.stop_timer('total_time')
        monitor.record_resource_usage()
        
        print("\n" + "=" * 70)
        print("✅ BENCHMARK COMPLETATO")
        print("=" * 70)
        print(f"\n⏱️  Tempo totale: {total_time:.2f} secondi")
        
        # Parse output per estrarre metriche
        output_lines = result.stdout.split('\n')
        
        # Cerca MAE finale (da adattare al formato output Flower)
        final_mae = None
        for line in output_lines:
            if 'mae' in line.lower():
                # Parsing semplice, da migliorare
                print(f"   {line.strip()}")
        
        # Summary risultati
        results = {
            'approach': 'flower_bagging',
            'total_time_sec': total_time,
            'num_clients': config['dataset']['num_clients'],
            'num_rounds': config['federated']['num_rounds'],
            'local_epochs': config['federated']['local_epochs'],
            'final_mae': final_mae,
        }
        
        # Aggiungi statistiche monitor
        results.update(monitor.get_summary())
        
        # Salva risultati
        results_dir = Path(__file__).parent.parent.parent / "results"
        save_results(results, str(results_dir), "flower_bagging")
        
        print(f"\n📊 Risultati salvati in {results_dir}")
        
        return results
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Errore durante esecuzione:")
        print(e.stderr)
        return None
    except KeyboardInterrupt:
        print(f"\n⚠️ Benchmark interrotto dall'utente")
        return None


if __name__ == "__main__":
    print("\n🧪 Flower Bagging Benchmark - NECSTLab\n")
    
    # Verifica setup
    data_dir = Path(__file__).parent.parent.parent / "data" / "ready_for_flwr"
    
    if not data_dir.exists():
        print("⚠️  ATTENZIONE: Directory dati non trovata!")
        print(f"   Path: {data_dir}")
        print("\n   Crea symlink con:")
        print('   cd /Users/annamettifogo/Desktop/polimi/necstlab/progetto\\ LS2/fl_benchmark/data')
        print('   ln -s /Users/annamettifogo/Desktop/polimi/1°\\ magistrale/csi/proj4/prova1/dtbagging/ready_for_flwr ready_for_flwr')
        sys.exit(1)
    
    # Run benchmark
    results = run_flower_bagging_benchmark()
    
    if results:
        print("\n✅ Benchmark completato con successo!")
    else:
        print("\n❌ Benchmark fallito")
