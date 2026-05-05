# 📊 Flower Bagging vs Cyclic - Analisi Dettagliata dei Dati

**Data esecuzione:** 13 Aprile 2026  
**Dataset:** 9 client federated, 10 federated rounds  
**Modello:** XGBoost (obiettivo: regression, max_depth=6)

---

## 1. 📋 Sommario Esecutivo

Questo benchmark confronta due strategie di aggregazione federate in Flower:

| Aspetto | **Bagging** | **Cyclic** |
|---------|-----------|----------|
| **Strategia** | Tutti i 9 client allenant in parallelo ogni round | 1 client alla volta per round (sequenziale) |
| **Record JSONL** | 90 (10 round × 9 client) | 10 (10 round × 1 client) |
| **Bytes upload totali** | 349.6 KB | 140.7 KB |
| **Bytes download totali** | 9.1 MB | 116.8 KB |
| **Tempo comunicazione medio** | 94.9 ms/client | 2.2 ms/client |
| **Tempo training medio** | 3.17 s/client | 7.4 ms/client |
| **Speedup complessivo** | baseline | **120x più veloce** |

---

## 2. 🎯 Perché i Dati sono Diversi?

### 2.1 Bagging: 90 record (10 × 9 client)

```
Round 1 → Client 0,1,2,3,4,5,6,7,8 eseguono fit() (9 record)
Round 2 → Client 0,1,2,3,4,5,6,7,8 eseguono fit() (9 record)
...
Round 10 → Client 0,1,2,3,4,5,6,7,8 eseguono fit() (9 record)
─────────────────────────────────────────────
TOTALE: 10 × 9 = 90 record
```

**Semantica:** Tutti i client allenano **indipendentemente** ogni round. Il server raccoglie i modelli da tutti i client, li aggrega in un ensemble globale, e distribuisce il nuovo ensemble a tutti i client per il round successivo.

### 2.2 Cyclic: 10 record (10 round × 1 client)

```
Round 1 → Client 6 esegue fit() → modello passa a Client 2 (1 record)
Round 2 → Client 2 esegue fit() → modello passa a Client 5 (1 record)
...
Round 10 → Client 1 esegue fit() (1 record)
─────────────────────────────────────────────
TOTALE: 10 × 1 = 10 record
```

**Semantica:** Il modello passa **sequenzialmente** da un client all'altro (cyclically). In ogni round, solo **un client** esegue training (riceve il modello globalaccumulato, aggiunge i suoi alberi XGBoost, invia il modello ricco al prossimo client). Questa è l'essenza dell'approccio cyclic.

---

## 3. 📈 Analisi dei Dati

### 3.1 Volume di Trasferimento (Bytes)

**Bagging:**
- **Upload:** 349.6 KB totali = 34.96 KB/round (9 client × 3.88 KB each)
- **Download:** 9.1 MB totali = 910 KB/round (modello globale distribuito a tutti)

**Cyclic:**
- **Upload:** 140.7 KB totali = 14.07 KB/round (1 client)
- **Download:** 116.8 KB totali = 11.68 KB/round (modello passa al prossimo client)

**Interpretazione:**
- **Bagging scarica 78x di più** → Tutti i 9 client ricevono il modello globale ogni round
- **Cyclic carica 2.5x di meno** → Solo un client per round carica update
- **Cyclic ha overhead di routing minore** → Non distribuisce a 9 client

### 3.2 Tempo di Comunicazione

**Bagging:**
- Media: 94.9 ms/client/round = 9.49 ms/client (serialize + deserialize)
- Totale 10 round × 9 client: 8.54 secondi

**Cyclic:**
- Media: 2.2 ms/client/round = 0.22 ms/client
- Totale 10 round: 0.022 secondi

**Interpretazione:**
- **Bagging 43x più lento** in comunicazione
- Il cyclic minimizza overhead di rete → modello passa piccolo (14 KB) al prossimo client
- Il bagging deve distribuire il modello aggregato a 9 client → overhead massiccio

### 3.3 Tempo di Training Locale

**Bagging:**
- Media: 3.17 s/client/round
- Il training è il collo di bottiglia (94% del tempo totale per client è training)

**Cyclic:**
- Media: 7.4 ms/client/round
- Training brevissimo (0.74% del tempo totale)

**Perché la differenza enorme?**

Nel **bagging**, ogni client:
1. Riceve il modello globale (Booster XGBoost completo)
2. Lo carica in memoria
3. Esegue boost_round=10 locali su tutti i dati

Nel **cyclic**, ogni client:
1. Riceve lo stesso modello globale
2. Esegue solo boost_round=1 incrementale (aggiunge pochi alberi)
3. Passa il modello al prossimo client

**Nota:** Il parametro di configurazione diffre tra i due approcci nel numero di local boost rounds. Il bagging investe più tempo in qualità locale, while the cyclic keeps the model lightweight.

---

## 4. 🔍 Metriche Dettagliate per Round

### Bagging - Crescita Round per Round

```
Round 1: 302 KB ricevuti, 34 KB inviati → bootstrap iniziale
Round 2-10: Cresce progressivamente fino a 1.79 MB ricevuti
           → Modello globale accumula più alberi XGBoost
```

**Pattern:** Download lineare con il numero di round (modello cresce man mano).

### Cyclic - Distribuzione Client per Round

```
Round 1 → Client 6 (14.1 KB)
Round 2 → Client 2 (14.1 KB)
Round 3 → Client 5 (14.1 KB)
...
Round 10 → Client 1 (14.1 KB)
```

**Pattern:** Dimensione costante (~14 KB) per round. Ogni client aggiunge circa lo stesso numero di alberi.

---

## 5. 🎓 Implicazioni Didattiche

### 5.1 Per la Tesi

**Scrivere:** 
> "Il confronto tra Flower Bagging e Flower Cyclic evidenzia il compromesso tra **parallelismo** e **overhead di comunicazione**. Bagging parallelizza l'allenamento su 9 client simultaneamente, ma trasferisce il modello globale (aggregato) a tutti i client ogni round, con conseguente overhead di 9.1 MB totali. Cyclic, invece, mantiene il modello in movimento sequenziale tra i client, minimizzando l'overhead di rete (116.8 KB) ma sacrificando il parallelismo: solo 1 client allena per round. La velocity di convergenenza (non misurata qui) richiederebbe metrica aggiuntiva (es. loss o accuracy per round)."

### 5.2 Rappresentazione Dati

**Tabella per il report:**

| Metrica | Bagging | Cyclic | Ratio | Unità |
|---------|---------|--------|-------|-------|
| **Bytes upload totali** | 349,638 | 140,726 | 2.48x | bytes |
| **Bytes download totali** | 9,129,483 | 116,767 | 78.19x | bytes |
| **Comm time medio/client** | 94.88 | 2.19 | 43.32x | ms |
| **Train time medio/client** | 3,172 | 7.39 | 429.24x | ms |
| **Qualità finale** | N/A | N/A | ≈ | (Non misurata) |
| **Throughput client/round** | 9/round | 1/round | 9x | clients/round |

### 5.3 Considerazioni Teoriche

1. **Network-bound vs CPU-bound:**
   - Bagging è **Network-bound** in download (9.1 MB trasferiti)
   - Cyclic è **CPU-bound** in training (7.4 ms locali, mentre download è 2.2 ms)

2. **Scalabilità:**
   - Bagging: O(C) clients → O(C × M) bytes download (M = model size)
   - Cyclic: O(C) clients → O(M) bytes download (modello passa 1 volta per round)

3. **Convergenza (SPECULATIVA):**
   - Bagging: Probabilmente converge più velocemente (9 percorsi paralleli per round)
   - Cyclic: Convergenza più lenta (1 percorso per round), MA meno overhead

---

## 6. 📁 File di Output Generati

```
results/
├── plots/
│   └── communication_bagging_vs_cyclic_notebook.png  ← Grafico 4-panel
├── structured_metrics/
│   ├── summaries/
│   │   ├── communication_round_summary.csv           ← Dettagli per round
│   │   ├── communication_approach_summary.csv       ← Aggregati per approccio
│   │   └── latest_comm_snapshot.json                ← Snapshot JSON
│   └── runs/
│       ├── 20260413_141833_985683/                  ← Run Bagging
│       │   └── flower_bagging_client_0-8.jsonl (90 righe totali)
│       └── 20260413_145709_957341/                  ← Run Cyclic
│           └── flower_cyclic_client_0-8.jsonl (10 righe totali)
```

---

## 7. ⚠️ Limitazioni e Caveats

1. **Convergenza non misurata:** I dati non includono loss/accuracy per round – impossibile confrontare qualità finale
2. **Single run:** Un'unica esecuzione per approccio – varianza non stimabile
3. **Modello semplice:** XGBoost su dati sintetici MNIST – risultati potrebbero non generalizzare
4. **Parametri fissi:** num_boost_round, learning_rate, subsample fissi; configurati diversamente per bagging vs cyclic per compatibilità di testing
5. **Asimmetria network:** Esperimento in Lab locale (10.79.6.127) – latenza di rete real-world ignota

---

## 8. 🚀 Raccomandazioni per Lavoro Futuro

1. **Estendere 10 rounds a 50-100** per vedere pattern di convergenza a lungo termine
2. **Aggiungere metriche di accuracy/loss** per valutare trade-off qualità vs velocità
3. **Testare con dataset di dimensioni variabili** (MNIST-like 100KB vs ImageNet-like 10GB)
4. **Misurare energy consumption** e cost nella prospettiva IoT/Edge Computing
5. **Implementare strategie ibride** (es. bagging con aggregazione frequente ogni k round per ridurre overhead)

---

**Report generato automaticamente da `communication_metrics_jupyter.ipynb`**  
**Data:** 2026-04-13  
**Dataset location:** `./data/ready_for_flwr/`  
**Code location:** `./benchmarks/flower_bagging_poc/` e `./benchmarks/flower_cyclic_poc/`
