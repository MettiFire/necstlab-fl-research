# NVIDIA FLARE XGBoost Benchmark

## Panoramica

Implementazione di **NVIDIA FLARE** per Federated Learning con XGBoost per predizione della qualità del sonno.

## Architettura

```
nvidia_flare/
├── app/
│   ├── config/
│   │   ├── config_fed_server.json    # Configurazione server
│   │   ├── config_fed_client.json    # Configurazione client  
│   │   └── meta.json                 # Metadata applicazione
│   └── custom/                        # Moduli custom (copiati da ../custom/)
├── custom/
│   ├── xgb_trainer.py                # Executor per training client
│   └── xgb_aggregator.py             # Aggregatore server
└── run_benchmark.py                  # Script di esecuzione

workspace/                             # Creato durante esecuzione
└── server/
    └── simulate_job/
        └── results/
            ├── final_model.json       # Modello aggregato finale
            └── timing_metrics.json    # Metriche temporali
```

## Differenze Chiave vs Flower

### NVIDIA FLARE
- **Comunicazione**: Histogram-based (ogni albero)
- **Aggregazione**: Weighted averaging o SecureBoost
- **Paradigma**: Push histograms → Server builds tree
- **Frequenza**: N comunicazioni per round (N = num_trees)
- **Overhead**: Maggiore per singolo albero, ma più sicuro

### Flower Bagging/Cyclic  
- **Comunicazione**: Model weights (fine round)
- **Aggregazione**: FedAvg (bagging) o Sequential (cyclic)
- **Paradigma**: Train locally → Push weights
- **Frequenza**: 1 comunicazione per round
- **Overhead**: Minore, modelli possono essere grandi

## Esecuzione

### Metodo 1: Script integrato

```bash
cd /Users/annamettifogo/Desktop/polimi/necstlab/progetto_LS2/fl_benchmark
source venv/bin/activate
cd benchmarks/nvidia_flare
python run_benchmark.py
```

### Metodo 2: Diretto con nvflare

```bash
# Setup
cd benchmarks/nvidia_flare
python -c "from run_benchmark import setup_app_structure; setup_app_structure()"

# Esegui simulator
nvflare simulator app \
  -w workspace \
  -n 9 \
  -t 9
```

## Parametri Configurabili

### Server (`config_fed_server.json`)
- `num_rounds`: 10 - Numero di round FL
- `min_clients`: 9 - Minimo client necessari

### Client (`config_fed_client.json`)
- `num_boost_round`: 5 - Alberi per round locale
- `max_depth`: 6 - Profondità alberi
- `learning_rate`: 0.1 - Learning rate XGBoost
- `tree_method`: "hist" - Histogram-based

## Risultati

Dopo l'esecuzione, i risultati saranno in:
```
workspace/server/simulate_job/results/
├── final_model.json        # Modello XGBoost finale
├── timing_metrics.json     # Tempi esecuzione
└── model_round_*.json      # Modelli intermedi
```

## Monitoring

Durante l'esecuzione vedrai:
- ⏳ Caricamento dati per ogni client
- 🧠 Training locale per round
- 🔄 Aggregazione modelli sul server
- ✅ Metriche MAE e tempi

## Confronto con Altri Approcci

| Metrica | NVIDIA FLARE | Flower Bagging | Flower Cyclic |
|---------|--------------|----------------|---------------|
| Comunicazioni/round | N (per albero) | 1 | 1 |
| Privacy | Alta (histograms) | Media | Media |
| Overhead | Alto | Medio | Basso |
| Complessità setup | Alta | Media | Media |
| GPU support | Sì | No | No |

## Note Tecniche

- **tree_method: hist** è fondamentale per histogram-based boosting
- NVIDIA FLARE usa **gRPC** per comunicazione (vs Ray in Flower)  
- Supporta **Differential Privacy** e **Secure Aggregation** (non implementati qui)
- Più adatto per **scenari enterprise** con requisiti di sicurezza

## Troubleshooting

**Import Error**:
```bash
pip install nvflare==2.7.1
```

**PYTHONPATH issues

# NVIDIA FLARE Histogram-based (simplified)
for round in rounds:
    for tree in num_trees:
        for client in clients:
            histogram = compute_histogram(data, tree)
            send(histogram)  # ← MORE COMMUNICATION
        global_tree = aggregate_histograms(histograms)
        broadcast(global_tree)
```

## Setup NVIDIA FLARE

### Installazione
```bash
pip install nvflare
```

### Struttura Progetto FLARE
```bash
nvflare provision  # Genera certificati
nvflare simulator  # Per testing locale
```

## Metriche da Confrontare

| Metrica | Flower Bagging | NVIDIA FLARE | Aspettativa |
|---------|----------------|--------------|-------------|
| Training Time | ~5-10 min | ~30-60 min | 6-10x più lento |
| Communication | Low | High | 50-100x più dati |
| Accuracy (MAE) | ~12-13 | **~10-11?** | 10-15% migliore |
| Convergence | Slow | Fast | Meno rounds? |

## Implementazione Steps

1. **Configurare FLARE project**
   ```bash
   nvflare poc prepare -n 9  # 9 clients
   ```

2. **Creare custom trainer** (`custom/xgb_trainer.py`)
   - Implementa histogram sharing
   - Carica dati Garmin
   - Integra con XGBoost

3. **Config files** (JSON)
   - Parametri XGBoost
   - Network settings
   - Histogram bins (default: 256)

4. **Run benchmark** (`run_benchmark.py`)
   - Launch con nvflare simulator
   - Monitor timing
   - Log communication overhead

## Reference

- [NVFLARE Docs](https://nvflare.readthedocs.io/)
- [NVFLARE GitHub](https://github.com/NVIDIA/NVFlare)
- [XGBoost + FLARE Tutorial](https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/xgboost)

## Note Implementazione

**Challenge:**
- FLARE è più complesso di Flower (enterprise framework)
- Richiede setup certificati (anche per simulazione)
- Documentazione meno friendly per ricerca accademica

**Alternativa se troppo complesso:**
- Implementare **manualmente** histogram sharing in Flower
- Modificare client/server per inviare histograms invece di weights
- Questa sarebbe la "via di mezzo" che state cercando!
