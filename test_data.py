#!/usr/bin/env python3
"""
Quick test per verificare caricamento dati
"""
import sys
from pathlib import Path

print("🧪 Test Caricamento Dati Garmin\n")
print("=" * 60)

# Test 1: Verifica esistenza directory
print("\n1️⃣  Verifica directory dati...")
data_dir = Path("./data/ml_ready_final_fed")
if data_dir.exists():
    print(f"   ✅ Directory trovata: {data_dir}")
else:
    print(f"   ❌ Directory non trovata: {data_dir}")
    sys.exit(1)

# Test 2: Conta cartelle client e CSV
print("\n2️⃣  Verifica struttura client...")
client_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: int(p.name))
print(f"   ✅ Trovate {len(client_dirs)} cartelle client")
for client_dir in client_dirs:
    csv_files = sorted(client_dir.glob("*.csv"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    print(f"      - Client {client_dir.name}: {len(csv_files)} CSV")

# Test 3: Carica librerie necessarie
print("\n3️⃣  Carica librerie (può richiedere ~10 sec)...")
try:
    import pandas as pd
    import numpy as np
    print("   ✅ pandas e numpy caricati")
except ImportError as e:
    print(f"   ❌ Errore importazione: {e}")
    sys.exit(1)

# Test 4: Carica dati client 0
print("\n4️⃣  Test caricamento client 0...")
try:
    client_zero = data_dir / "0"
    csv_files = sorted(client_zero.glob("*.csv"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    df = pd.concat([pd.read_csv(csv_file) for csv_file in csv_files], ignore_index=True)
    
    print(f"   ✅ Dataset caricato:")
    print(f"      - Forma: {df.shape}")
    print(f"      - Colonne: {df.shape[1]}")
    print(f"      - Righe: {df.shape[0]}")
    
    # Conta feature numeriche (escludi metadati)
    feature_cols = [
        col for col in df.columns 
        if col not in ['day', 'label', 'file', 'source_row', 'id', 'client_id'] 
        and not col.endswith('_time_series')
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    
    print(f"      - Feature numeriche: {len(feature_cols)}")
    
    if 'label' in df.columns:
        print(f"      - Range label: [{df['label'].min():.1f}, {df['label'].max():.1f}]")
        print(f"      - Media label: {df['label'].mean():.1f}")
    
except Exception as e:
    print(f"   ❌ Errore caricamento: {e}")
    sys.exit(1)

# Test 5: Verifica XGBoost DMatrix
print("\n5️⃣  Test creazione XGBoost DMatrix...")
try:
    import xgboost as xgb
    
    X = df[feature_cols].dropna()
    y = df.loc[X.index, 'label']
    
    # Split 80/20
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    train_dm = xgb.DMatrix(X_train, label=y_train)
    val_dm = xgb.DMatrix(X_val, label=y_val)
    
    print(f"   ✅ DMatrix create:")
    print(f"      - Train: {train_dm.num_row()} samples, {train_dm.num_col()} features")
    print(f"      - Valid: {val_dm.num_row()} samples, {val_dm.num_col()} features")
    
except Exception as e:
    print(f"   ❌ Errore XGBoost: {e}")
    sys.exit(1)

# Test 6: Verifica tutti i client
print("\n6️⃣  Verifica caricamento tutti i client...")
try:
    client_stats = []
    for client_dir in client_dirs:
        csv_files = sorted(client_dir.glob("*.csv"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
        df = pd.concat([pd.read_csv(csv_file) for csv_file in csv_files], ignore_index=True)
        client_stats.append({
            'client': int(client_dir.name),
            'samples': len(df),
            'features': len(feature_cols)
        })
    
    total_samples = sum(s['samples'] for s in client_stats)
    
    print(f"   ✅ Tutti i client caricati:")
    print(f"      - Totale campioni: {total_samples}")
    print(f"      - Media per client: {total_samples/len(client_stats):.1f}")
    print(f"\n      Dettaglio:")
    for s in client_stats:
        print(f"        Client {s['client']}: {s['samples']} samples")
    
except Exception as e:
    print(f"   ❌ Errore: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ TUTTI I TEST PASSATI!")
print("=" * 60)
print("\n🚀 Il dataset è pronto per i benchmark FL!\n")
