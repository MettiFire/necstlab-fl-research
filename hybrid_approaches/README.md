# Hybrid Approaches - Via di Mezzo

## Obiettivo
Trovare approcci intermedi tra:
- **Flower Bagging** (veloce, poco accurato)
- **NVIDIA FLARE** (lento, molto accurato)

Target: **80% accuratezza FLARE, 30% tempo FLARE**

---

## Idee da Esplorare

### 1. Selective Histogram Sharing
**Idea:** Condividi histograms solo per feature più importanti

**Implementazione:**
```python
# Feature importance da primo training locale
importance = model.get_feature_importance()
top_k_features = importance.argsort()[-20:]  # Top 20 features

# Condividi histograms solo per top-K
for feature in top_k_features:
    histogram = compute_histogram(data[:, feature])
    share(histogram)

# Per altre feature: usa approccio bagging (solo model)
```

**Aspettative:**
- Communication: 50% di FLARE (20/40 features)
- Accuratezza: ~90% di FLARE
- Tempo: ~50% di FLARE

---

### 2. Adaptive Communication Strategy
**Idea:** Più comunicazione all'inizio (convergenza), meno dopo (fine-tuning)

**Implementazione:**
```python
for round in range(num_rounds):
    if round < early_phase_rounds:  # e.g., primi 5 rounds
        # Modalità FLARE: histogram sharing
        share_histograms()
    else:
        # Modalità Bagging: solo model weights
        share_model_weights()
```

**Aspettative:**
- Communication: 50% di FLARE (5/10 rounds pesanti)
- Accuratezza: ~95% di FLARE
- Tempo: ~40-50% di FLARE

---

### 3. Compressed Histograms
**Idea:** Riduci dimensione histograms con quantizzazione/aggregazione

**Implementazione:**
```python
# Invece di 256 bins per histogram
histogram_full = np.histogram(data, bins=256)

# Usa 32 bins (8x compressione)
histogram_compressed = np.histogram(data, bins=32)

# O quantizza valori
histogram_quantized = quantize(histogram_full, levels=16)
```

**Aspettative:**
- Communication: ~12-25% di FLARE (8x compressione)
- Accuratezza: ~92% di FLARE (loss per compressione)
- Tempo: ~15-25% di FLARE

---

### 4. Progressive Histogram Refinement
**Idea:** Condividi histograms grossolani prima, raffinati dopo

**Implementazione:**
```python
for round in range(num_rounds):
    # Round iniziali: histogram coarse (pochi bins)
    num_bins = 16 * (2 ** (round // 2))  # 16 → 32 → 64 → 128
    
    histogram = compute_histogram(data, bins=num_bins)
    share(histogram)
```

**Aspettative:**
- Communication: Crescente, ma start leggero
- Accuratezza: Convergenza più smooth
- Tempo: ~30-40% di FLARE

---

### 5. Federated Knowledge Distillation
**Idea:** Teacher model (FLARE-like) + Student model (lightweight)

**Implementazione:**
```python
# Phase 1: Train teacher con histogram sharing (slow)
teacher_model = train_with_histograms()  # Heavy

# Phase 2: Distill in student models (fast)
for client in clients:
    student_model = train_with_teacher_predictions(teacher_model)
```

**Aspettative:**
- Communication: Heavy per teacher, minimal per students
- Accuratezza: ~95% di FLARE
- Tempo: One-time cost + fast student training

---

### 6. Hybrid Parallel-Sequential
**Idea:** Alcuni client bagging (parallel), altri cyclic (sequential)

**Implementazione:**
```python
# Split clients in 3 groups
group_A = clients[:3]  # Cyclic
group_B = clients[3:6]  # Cyclic
group_C = clients[6:]   # Cyclic

# Round 1-3: Group A (cyclic)
# Round 4-6: Group B (cyclic)
# Round 7-9: Group C (cyclic)
# Aggregate: Bagging dei 3 group models
```

**Aspettative:**
- Communication: 3x cyclic (invece di 9x)
- Accuratezza: Meglio di bagging puro
- Tempo: ~1/3 di cyclic completo

---

## Struttura Directory

Ogni idea diventa una subdirectory con:
```
hybrid_approaches/
├── selective_histogram/
│   ├── README.md              # Design doc
│   ├── implementation.py      # Codice
│   ├── run_experiment.py      # Benchmark
│   └── results/
│
├── adaptive_communication/
│   └── ...
│
└── ...
```

---

## Metriche Confronto Obbligatorie

Per ogni approccio ibrido, misurare:

| Metrica | Come Calcolare |
|---------|----------------|
| **Speedup vs FLARE** | `time_flare / time_hybrid` |
| **Accuracy Loss** | `mae_hybrid - mae_flare` |
| **Communication Ratio** | `bytes_hybrid / bytes_flare` |
| **Efficiency Score** | `(accuracy_gain × speedup) / communication_ratio` |

---

## Prossimi Step

1. **Implementa baseline FLARE** (per confronto)
2. **Scegli 2-3 approcci ibridi** più promettenti
3. **Implementa e testa**
4. **Analizza trade-off** (accuracy vs time vs communication)
5. **Paper/Report** con risultati

---

## Note di Ricerca

**Domande da Rispondere:**
- Qual è il bottleneck vero di FLARE? (Communication? Computation?)
- Quali feature sono davvero critiche per accuratezza?
- Quando converge il modello? (primi round = critical?)
- Histogram compression: quale loss di informazione è accettabile?

**Baseline da Stabilire:**
- Tempo assoluto: Bagging (X min) vs FLARE (Y min)
- Accuratezza: Bagging (A MAE) vs FLARE (B MAE)
- Gap da colmare: (Y-X) time, (A-B) accuracy

---

Questa è **la parte più interessante del progetto** - qui c'è innovazione! 🚀
