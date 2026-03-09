"""
Run Flower Cyclic Benchmark
NECSTLab - Polimi LS2

Esegue benchmark completo di Flower Cyclic con profiling.
"""
import subprocess
import time
import yaml
from pathlib import Path
import sys
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import PerformanceMonitor, save_results


def run_flower_cyclic_benchmark(config_path: str = "../../config.yaml"):
    """Esegue benchmark Flower Cyclic"""
    
    # Carica configurazione
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 70)
    print("🔄 FLOWER CYCLIC BENCHMARK")
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
    
    # Usa config cyclic dal pyproject.toml
    cmd = [
        "flwr", "run", ".",
        "--run-config", "train-method=cyclic"
    ]
    
    # Override serverapp e clientapp per cyclic
    cmd += [
        "--run-config", "serverapp=benchmarks.flower_cyclic.server:app",
        "--run-config", "clientapp=benchmarks.flower_cyclic.client:app"
    ]
    
    print("🚀 Avvio Flower simulation (Cyclic)...")
    print(f"Command: flwr run . --run-config train-method=cyclic")
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
        
        # Cerca MAE finale
        final_mae = None
        for line in output_lines:
            if 'mae' in line.lower():
                print(f"   {line.strip()}")
        
        # Summary risultati
        results = {
            'approach': 'flower_cyclic',
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
        save_results(results, str(results_dir), "flower_cyclic")
        
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
    print("\n🧪 Flower Cyclic Benchmark - NECSTLab\n")
    
    # Verifica setup
    data_dir = Path(__file__).parent.parent.parent / "data" / "ready_for_flwr"
    
    if not data_dir.exists():
        print("⚠️  ATTENZIONE: Directory dati non trovata!")
        print(f"   Path: {data_dir}")
        sys.exit(1)
    
    # Run benchmark
    results = run_flower_cyclic_benchmark()
    
    if results:
        print("\n✅ Benchmark completato con successo!")
    else:
        print("\n❌ Benchmark fallito")
