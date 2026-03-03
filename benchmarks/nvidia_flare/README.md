# NVIDIA FLARE Benchmark

## TODO: Da Implementare

Questa directory conterrà l'implementazione del benchmark per **NVIDIA FLARE**.

## Cosa è NVIDIA FLARE

Framework enterprise per Federated Learning con supporto avanzato per:
- Histogram-based XGBoost (SecureBoost)
- Differential privacy
- Secure aggregation
- GPU acceleration

## Architettura FLARE

```
Project/
├── config/
│   ├── config_fed_server.json    # Server configuration
│   └── config_fed_client.json    # Client configuration
├── custom/
│   ├── xgb_trainer.py            # Custom XGBoost trainer
│   └── xgb_validator.py          # Custom validator
└── run_benchmark.py              # Launch script
```

## Histogram Sharing

**Differenza chiave vs Flower:**

**Flower Bagging:**
- Scambia: model weights (1x per round)
- Grandezza: ~MB per client
- Quando: Solo a fine training locale

**NVIDIA FLARE:**
- Scambia: gradient histograms (N volte per round, N = num trees)
- Grandezza: ~MB × num_trees × num_features
- Quando: Ad ogni tree construction (dentro ogni boosting iteration)

### Esempio Communication Pattern

```python
# Flower Bagging (simplified)
for round in rounds:
    for client in clients:  # Parallel
        model = train_locally(data)
    global_model = aggregate(models)

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
