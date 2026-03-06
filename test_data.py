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
data_dir = Path("./data/ready_for_flwr")
if data_dir.exists():
    print(f"   ✅ Directory trovata: {data_dir}")
else:
    print(f"   ❌ Directory non trovata: {data_dir}")
    sys.exit(1)

# Test 2: Conta file client
print("\n2️⃣  Verifica file client...")
client_files = list(data_dir.glob("client_*.csv"))
print(f"   ✅ Trovati {len(client_files)} file client")
for f in sorted(client_files):
    size_kb = f.stat().st_size / 1024
    print(f"      - {f.name}: {size_kb:.1f} KB")

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
    df = pd.read_csv(data_dir / "client_0.csv")
    
    print(f"   ✅ Dataset caricato:")
    print(f"      - Forma: {df.shape}")
    print(f"      - Colonne: {df.shape[1]}")
    print(f"      - Righe: {df.shape[0]}")
    
    # Conta feature numeriche (escludi day, label, time_series)
    feature_cols = [
        col for col in df.columns 
        if col not in ['day', 'label'] 
        and not col.endswith('_time_series')
        and df[col].dtype in ['int64', 'float64']
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
    for i in range(9):
        df = pd.read_csv(data_dir / f"client_{i}.csv")
        client_stats.append({
            'client': i,
            'samples': len(df),
            'features': len(feature_cols)
        })
    
    total_samples = sum(s['samples'] for s in client_stats)
    
    print(f"   ✅ Tutti i 9 client caricati:")
    print(f"      - Totale campioni: {total_samples}")
    print(f"      - Media per client: {total_samples/9:.1f}")
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
