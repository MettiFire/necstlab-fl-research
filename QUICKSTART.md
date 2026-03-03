# Quick Start Guide - FL Benchmark

## Setup Rapido (5 minuti)

### 1. Esegui setup automatico
```bash
cd "/Users/annamettifogo/Desktop/polimi/necstlab/progetto LS2/fl_benchmark"
chmod +x setup.sh
./setup.sh
```

### 2. Attiva environment
```bash
source venv/bin/activate
```

### 3. Verifica installazione
```bash
python utils.py
```

Dovresti vedere:
```
🧪 Test DataLoader...
✅ Client 0 caricato:
   - Training samples: ~88
   - Validation samples: ~22
   - Features: 40
```

---

## Test Primo Benchmark: Flower Bagging

```bash
cd benchmarks/flower_bagging
python run_benchmark.py
```

Questo eseguirà:
- 9 client in parallelo
- 10 rounds federated learning
- Training XGBoost con bagging
- Misurazione tempi e accuratezza
- Salvataggio risultati in `results/`

**Tempo stimato:** ~5-10 minuti

---

## Prossimi Step

### 1. Test Flower Cyclic
```bash
cd benchmarks/flower_cyclic
# (TODO: implementare client/server cyclic)
```

### 2. Test NVIDIA FLARE
```bash
cd benchmarks/nvidia_flare
# (TODO: configurare NVFLARE)
```

### 3. Analisi Comparativa
```bash
jupyter notebook analysis/compare_approaches.ipynb
```

---

## Troubleshooting

### Errore: "Directory dati non trovata"
Verifica symlink:
```bash
ls -la data/
# Dovresti vedere ready_for_flwr -> /path/to/original
```

Se manca, crea manualmente:
```bash
cd data
ln -s "/Users/annamettifogo/Desktop/polimi/1° magistrale/csi/proj4/prova1/dtbagging/ready_for_flwr" ready_for_flwr
```

### Errore: "Module not found"
Reinstalla dipendenze:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Flower simulation non parte
Verifica versione Flower:
```bash
pip show flwr
# Dovrebbe essere >= 1.13.0
```

---

## Info Rapide

**Cosa misura il benchmark?**
- ⏱️ Tempo totale training
- 📡 Overhead comunicazione
- 🎯 Accuratezza (MAE)
- 💾 Uso memoria/CPU

**Dove sono i risultati?**
- `results/` - CSV aggregati
- `benchmarks/*/results/` - Output specifici per approccio

**Come modificare configurazione?**
Edita `config.yaml` - parametri XGBoost, rounds FL, etc.

---

Vedi [README.md](README.md) completo per documentazione estesa.
