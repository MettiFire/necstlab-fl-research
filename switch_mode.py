#!/usr/bin/env python3
"""
Helper script per switchare tra Flower Bagging e Cyclic
NECSTLab - Polimi LS2
"""
import sys
from pathlib import Path


def switch_mode(mode: str):
    """Modifica pyproject.toml per usare bagging o cyclic"""
    
    if mode not in ["bagging", "cyclic"]:
        print(f"❌ Modalità '{mode}' non valida. Usa 'bagging' o 'cyclic'")
        sys.exit(1)
    
    pyproject_path = Path(__file__).parent / "pyproject.toml"
    
    # Leggi contenuto
    with open(pyproject_path, 'r') as f:
        content = f.read()
    
    # Sostituisci componenti
    if mode == "bagging":
        content = content.replace(
            'serverapp = "benchmarks.flower_cyclic.server:app"',
            'serverapp = "benchmarks.flower_bagging.server:app"'
        )
        content = content.replace(
            'clientapp = "benchmarks.flower_cyclic.client:app"',
            'clientapp = "benchmarks.flower_bagging.client:app"'
        )
        content = content.replace(
            'train-method = "cyclic"',
            'train-method = "bagging"'
        )
    else:  # cyclic
        content = content.replace(
            'serverapp = "benchmarks.flower_bagging.server:app"',
            'serverapp = "benchmarks.flower_cyclic.server:app"'
        )
        content = content.replace(
            'clientapp = "benchmarks.flower_bagging.client:app"',
            'clientapp = "benchmarks.flower_cyclic.client:app"'
        )
        content = content.replace(
            'train-method = "bagging"',
            'train-method = "cyclic"'
        )
    
    # Scrivi modifiche
    with open(pyproject_path, 'w') as f:
        f.write(content)
    
    emoji = "🌸" if mode == "bagging" else "🔄"
    print(f"\n{emoji} Modalità cambiata a: {mode.upper()}")
    print(f"✅ pyproject.toml aggiornato")
    print(f"\nOra puoi eseguire: flwr run . --stream")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python switch_mode.py [bagging|cyclic]")
        print("\nEsempi:")
        print("  python switch_mode.py bagging")
        print("  python switch_mode.py cyclic")
        sys.exit(1)
    
    switch_mode(sys.argv[1])
