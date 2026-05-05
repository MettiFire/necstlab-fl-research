# Federated Learning Benchmark: Bagging vs Cyclic vs NVIDIA FLARE

**Progetto:** NECSTLab - Polimi LS2  
**Obiettivo:** Trovare un approccio intermedio tra velocità (Flower) e accuratezza (NVIDIA FLARE)  
**Dataset:** Dati biometrici Garmin (sleep quality prediction)

---

## 🎯 Obiettivi del Benchmark

### 1. **Baseline Performance**
Misurare tempo di esecuzione e accuratezza di:
- **Flower Bagging** (training parallelo + aggregazione predizioni)
- **Flower Cyclic** (training sequenziale)
- **NVIDIA FLARE** (histogram-based federated XGBoost)

### 2. **Analisi Colli di Bottiglia**
Identificare dove si spende il tempo:
- ⏱️ **Training locale** (tempo su ogni client)
- 📡 **Comunicazione** (overhead network, serializzazione)
- 🔄 **Aggregazione** (operazioni server-side)
- 🌳 **Tree construction** (gradient computation)

### 3. **Hybrid Approaches**
Progettare e testare approcci intermedi:
- Selective histogram sharing (solo feature importanti)
- Adaptive communication (più istogrammi nei primi round, meno dopo)
- Compressed histograms (quantizzazione)
- Mixed strategies (bagging + istogrammi parziali)

---

## 📊 Metriche da Confrontare

| Metrica | Descrizione | Unità |
|---------|-------------|-------|
| **Training Time** | Tempo totale per N rounds | secondi |
| **Communication Overhead** | Bytes scambiati per round | MB/round |
| **Model Accuracy (MAE)** | Mean Absolute Error su test set | MAE score |
| **Convergence Speed** | Rounds per raggiungere target MAE | # rounds |
| **Scalability** | Tempo vs numero di client | sec/client |

---

## 🗂️ Struttura del Progetto

```
fl_benchmark/
├── README.md                          # Questo file
├── requirements.txt                   # Dipendenze Python
├── config.yaml                        # Configurazione comune
│
├── data/                              # Dataset
│   ├── README.md                      # Info sul dataset
│   └── [symlink o copia da prova1]
│
├── benchmarks/                        # Implementazioni benchmark
│   ├── flower_bagging/
│   │   ├── client.py                  # Client Flower bagging
│   │   ├── server.py                  # Server Flower bagging
│   │   ├── run_benchmark.py           # Script esecuzione
│   │   └── results/                   # Output specifici
│   │
│   ├── flower_cyclic/
│   │   ├── client.py                  # Client Flower cyclic
│   │   ├── server.py                  # Server Flower cyclic
│   │   ├── run_benchmark.py
│   │   └── results/
│   │
│   └── nvidia_flare/
│       ├── config/                    # Config FLARE
│       ├── custom/                    # Custom trainers
│       ├── run_benchmark.py
│       └── results/
│
├── results/                           # Risultati aggregati
│   ├── benchmark_summary.csv          # Tabella comparativa
│   ├── timing_breakdown.csv           # Dettaglio tempi
│   └── plots/                         # Grafici
│
├── analysis/                          # Analisi e visualizzazioni
│   ├── compare_approaches.ipynb       # Notebook comparazione
│   ├── bottleneck_analysis.ipynb      # Analisi colli di bottiglia
│   └── visualization.py               # Utility plotting
│
└── hybrid_approaches/                 # Nuovi algoritmi intermedi
    ├── selective_histogram/           # Idea 1: istogrammi selettivi
    ├── adaptive_communication/        # Idea 2: comunicazione adattiva
    ├── compressed_histograms/         # Idea 3: compressione
    └── mixed_strategy/                # Idea 4: strategia mista
```

---

## 🚀 Quick Start

- connetti alla vpn
- ``` ssh annamettifogo@10.79.6.127 ```
- pw: necstvmcpu
- naviga con cd nella cartella fl_benchmark
- attiva venv con ```source venv/bin/activate```




comando server:

cd ~/fl_benchmark && source venv/bin/activate
python benchmarks/flower_bagging_poc/server.py \
  --server_address=0.0.0.0:8082 \
  --num_rounds=10 \
  --local_epochs=1 \
  --min_fit_clients=9 \
  --min_evaluate_clients=0 \
  --min_available_clients=9

  comandi client

  cd ~/fl_benchmark && source venv/bin/activate
for i in $(seq 0 8); do
  python benchmarks/flower_bagging_poc/client.py --server_address=127.0.0.1:8082 --client_id=$i &
done
wait



### 3. Run Benchmarks
```bash
# Flower Bagging
python benchmarks/flower_bagging/run_benchmark.py

# Flower Cyclic
python benchmarks/flower_cyclic/run_benchmark.py

# NVIDIA FLARE
python benchmarks/nvidia_flare/run_benchmark.py
```

---

## 🖥️ Accesso alla Macchina del Lab (SSH)

Per eseguire i benchmark o avviare server/client sulla macchina del laboratorio:

1. **Connettiti via SSH**

```bash
ssh <tuo_username>@<ip_macchina_lab>
# oppure, se serve una porta diversa:
ssh -p <porta> <tuo_username>@<ip_macchina_lab>
```

2. **Naviga nella cartella del progetto**

```bash
cd /percorso/alla/cartella/fl_benchmark
```

3. **(Opzionale) Copia file dal tuo PC al lab**

```bash
# Da locale a lab:
scp -r /percorso/locale/fl_benchmark <tuo_username>@<ip_macchina_lab>:/percorso/destinazione/
# Da lab a locale:
scp -r <tuo_username>@<ip_macchina_lab>:/percorso/remoto/fl_benchmark /percorso/locale/
```

4. **Trova l'IP della macchina lab**

```bash
hostname -I
# oppure
ip addr show
```

5. **Avvia server/client come da istruzioni PoC**

---

### 4. Analizza Risultati
```bash
jupyter notebook analysis/compare_approaches.ipynb
```

---

## 📚 Background Teorico

### **Flower Bagging/Cyclic**
- Framework leggero per FL
- Scambia solo **model weights** finali
- Bagging: training parallelo → media predizioni
- Cyclic: training sequenziale client-to-client

**Pro:**
- ⚡ Veloce (comunicazione minima)
- 🪶 Leggero (no dependencies pesanti)
- 🔧 Facile da debuggare

**Contro:**
- 📉 Accuratezza limitata (ogni client vede solo dati locali)
- 🎲 Sensibile a data heterogeneity

### **NVIDIA FLARE**
- Enterprise-grade FL framework
- Scambia **gradient histograms** ad ogni boosting round
- Approssima training centralizzato

**Pro:**
- 🎯 Migliore accuratezza (più informazione condivisa)
- 🔐 Privacy-preserving (istogrammi != dati raw)
- 🏢 Production-ready

**Contro:**
- 🐌 Più lento (molto overhead di comunicazione)
- 💾 Richiede più banda e storage
- 🧩 Più complesso da configurare

### **L'Opportunità: Hybrid Approach**
**Domanda di ricerca:** Esiste un punto intermedio ottimale?

Possibili idee:
1. **Selective histograms:** Solo per top-K feature importanti
2. **Adaptive frequency:** Più comunicazione all'inizio, meno dopo convergenza
3. **Compressed histograms:** Quantizzazione o aggregazione bins
4. **Federated distillation:** Teacher-student con comunicazione pesante/leggera alternata

---

## 📖 References

- [Flower Documentation](https://flower.ai/docs/)
- [NVIDIA FLARE Documentation](https://nvflare.readthedocs.io/)
- [Federated XGBoost Paper](https://arxiv.org/abs/1901.08755)
- [SecureBoost: Histogram-based FL](https://arxiv.org/abs/1901.08755)

---

## 👥 Team

**NECSTLab - Politecnico di Milano**  
Progetto di ricerca LS2 su Federated Learning

---

## 📝 Next Steps

- [ ] Setup NVIDIA FLARE environment
- [ ] Implementare baseline Flower bagging
- [ ] Implementare baseline Flower cyclic
- [ ] Implementare baseline NVIDIA FLARE
- [ ] Raccogliere metriche di timing dettagliate
- [ ] Analizzare communication overhead
- [ ] Progettare approcci ibridi
- [ ] Testare e confrontare
- [ ] Paper/Report finale
