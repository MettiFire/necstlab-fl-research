# 📝 Note per i Tutor - Flower Bagging vs Cyclic Benchmark

**Studente:** Anna Metti Fogo  
**Data progetto:** 13 Aprile 2026  
**Contesto:** Confronto comunicazione federated learning, XGBoost su 9 client, 10 round

---

## Domande Ricorrenti e Risposte

### D1: "Perché il cyclic ha solo 10 dati mentre il bagging ne ha 90?"

**Risposta breve:**
Perché i due algoritmi eseguono training in modo diverso:
- **Bagging:** 9 client allenano **in parallelo**, ogni round → 9 record per round × 10 round = 90 record
- **Cyclic:** 1 client allena **sequenzialmente**, ogni round → 1 record per round × 10 round = 10 record

**Analogy:** È come paragonare una catena di montaggio con 9 stazioni che lavorano in parallelo (Bagging) vs una catena dove il prodotto passa da una stazione all'altra sequenzialmente (Cyclic). Statisticamente diversi, non uno "sbagliato".

---

### D2: "Come possono essere confrontati se hanno numero di record diverso?"

**Risposta breve:**
I dati vengono **aggregati per round**, non per client. Così:

```
Bagging Round 1: 9 client attivi → avg/sum delle loro metriche
Cyclic Round 1: 1 client attivo → metrica di quel client

Entrambi gli approcci hanno 10 round, quindi 10 righe nel summario complessivo
```

**Nel CSV:** [results/structured_metrics/summaries/communication_round_summary.csv](results/structured_metrics/summaries/communication_round_summary.csv) vedrai 20 righe (10 round × 2 approcci), perfettamente confrontabili.

---

### D3: "Il cyclic è 'giusto' oppure ha fallito?"

**Risposta breve:**
È **completamente corretto**. La strategia Flower `FedXgbCyclic` è progettata così:

```python
# Nel paper Flower + XGBoost, cyclic significa:
for round in range(num_rounds):
    client_selected = list_of_clients[round % len(list_of_clients)]
    # Only that client trains and sends back
    model = client_selected.fit(model)
```

Il cyclic ha, PER DESIGN, 1 fit() per round. Non è un bug, è una feature di questa strategia.

---

### D4: "Perché il bagging scarica 78x di più?"

**Risposta breve:**

Bagging:
```
Round 1: Server aggrega modelli da 9 client → modello globale (300 KB)
         Distribuisce modello globale a 9 client → 9 × 300 KB = 2.7 MB download totale
Round 2-10: Modello cresce → fino a 1.79 MB/round × 9 client
TOTALE: 9.1 MB
```

Cyclic:
```
Round 1: Client 6 addestra → modello passa a Client 2 (modello non ridotto, ~14 KB transfer)
Round 2: Client 2 addestra → modello passa a Client 5 (14 KB)
...
TOTALE: 10 × 14 KB ≈ 116 KB
```

**Intuizione:** In Bagging, il modello aggregato viene replicato su 9 client. In Cyclic, passa una volta sola.

---

### D5: "Il tempo di training è strano - perché cyclic è 400x più veloce?"

**Risposta breve:**

Bagging client:
```python
# Local training: num_boost_round = 10 (configurato così)
xgb.train(params, train_dmatrix, num_boost_round=10)
# Tempo: ~3000ms per client
```

Cyclic client:
```python
# Local training: num_boost_round = 1 (per non espandere troppo il modello)
xgb.train(params, booster, num_boost_round=1)  # solo incremento
# Tempo: ~7ms per client  (addizione di 1 albero)
```

**Nota:** Questo è un **trade-off di progettazione**, non una differenza di potenza. Se facessi cyclic con num_boost_round=10 per client, sarebbe lentissimo (no parallelismo).

---

### D6: "Qual è 'meglio' tra Bagging e Cyclic?"

**Risposta breve:**

**Dipende da cosa misuri:**

| Se vuoi... | Scegli | Perché |
|-----------|--------|-------|
| Parallelismo massimo | Bagging | 9 client simultanei |
| Minimizzare overhead rete | Cyclic | 116 KB vs 9.1 MB |
| Velocità wall-clock | Cyclic | 22 ms vs 8.5 s comunicazione |
| Qualità convergenza | ❓ | Non misurata qui |

**Consiglio per tesi:** Scrivi "Non è preferibile in assoluto; Bagging massimizza throughput (comunicazione in parallelo), Cyclic minimizza overhead di rete (comunicazione sequenziale). La scelta dipende da vincoli di larghezza di banda e latenza della rete federated."

---

## Metriche Chiave da Spiegare

### Bytes Sent / Bytes Received

**Definizione:**
- **Sent:** Upload dal client al server (modello locale)
- **Received:** Download dal server al client (modello globale)

**Interpretazione nel Bagging:**
- Client invia ~34 KB (XGBoost booster locale compresso in JSON)
- Client riceve ~1 MB (modello globale aggregato - crescente per round)
- Rapporto 1:30 → molto download, poco upload (tipico di aggregazione)

**Interpretazione nel Cyclic:**
- Client invia ~14 KB
- Client riceve ~11 KB
- Rapporto 1:1 → simmetrico, modello passa invariato

---

### Communication Time Proxy

**Definizione:** `serialize_time + deserialize_time`

**Nel notebook:** Calcolato come tempo per convertire booster ↔ numpy array

**Interpretazione:**
- **Bagging 94.9 ms:** Deve serializzare modello grande da 9 client
- **Cyclic 2.2 ms:** Modello piccolo, passa 1 volta

---

### Train Time Mean

**Definizione:** Tempo locale di allenamento XGBoost per round

**Da spiegare ai tutor:**
> "Nel Bagging, ogni client addestra 10 alberi locali per round (somma a 30ms empirici). Nel Cyclic, ogni client addestra solo 1 albero incrementale per round (7ms). La configurazione è volutamente asimmetrica per mantenere il modello globale leggero nel Cyclic."

---

## File da Allegare alla Tesi

1. **📊 Plot:** [results/plots/communication_bagging_vs_cyclic_notebook.png](results/plots/communication_bagging_vs_cyclic_notebook.png)
   - 4 panel: Bytes sent, Bytes received, Comm time, Total volume

2. **📋 CSV Summary:** [results/structured_metrics/summaries/communication_approach_summary.csv](results/structured_metrics/summaries/communication_approach_summary.csv)
   - Tabella sintetica per il capitolo risultati

3. **📈 CSV Dettagliato:** [results/structured_metrics/summaries/communication_round_summary.csv](results/structured_metrics/summaries/communication_round_summary.csv)
   - Per appendice con dati per-round

4. **📄 Documento:** [analysis/BENCHMARK_DATA_EXPLANATION.md](analysis/BENCHMARK_DATA_EXPLANATION.md)
   - Riferimento tecnico completo

---

## Punti da Enfatizzare nella Presentazione

### ✅ Punti di forza del lavoro

1. **Isolamento run_id:** Ogni esecuzione è in una cartella timestamped separata → no data mixing
2. **Metriche strutturate:** JSONL per client per round consente aggregazioni flessibili
3. **Confronto pulito:** Due codebases separate (flower_bagging_poc vs flower_cyclic_poc) → implementazioni fedeli agli algoritmi
4. **Dati coerenti:** Entrambi gli approcci usano stesso dataset, modello, numero di client

### ⚠️ Limitazioni da ammettere

1. **Single run:** Non ci sono repliche – varianza statistica ignota
2. **Convergenza non misurata:** Non sappiamo quale approccio converge meglio
3. **Network locale:** Latenza in Lab non eguaglia rete WAN reale
4. **Parametri non ottimizzati:** num_boost_round fisso per semplicità, non per performance

---

## Possibili Domande d'Esame

### Q1: "Cosa significa che il cyclic ha 10 record?"
**Risposta attesa:** (Vedi D1 sopra) - Cyclic = 1 client per round per design

### Q2: "Perché non puoi fare un confronto 1-a-1?"
**Risposta attesa:** Record aggregati per round, non per client. Vedete 20 righe agglomerate in CSV summary.

### Q3: "Se il cyclic è più veloce, perché usare Bagging?"
**Risposta attesa:** Bagging paralelizza – potrebbe convergere più velocemente a soluzione. Trade-off comunicazione ↔ convergenza non esplicitato qui.

### Q4: "Come hai risolto il problema di run-mixing iniziale?"
**Risposta attesa:** Run_id timestamp-based, cartelle separate per esecuzione, notebook auto-seleziona latest run.

---

## Checklist Finale per Tesi

- [ ] Allegare plot 4-panel
- [ ] Includere CSV summary nel capitolo risultati
- [ ] Spiegare PERCHÉ record diversi (no è una "stranezza", è il design)
- [ ] Menzionare limitazioni (single run, no convergence metric)
- [ ] Suggerire lavoro futuro (replica study, larger rounds, convergence tracking)
- [ ] Crediti: Flower framework, NECSTLab infrastructure

---

**Doc generato:** 2026-04-13  
**Versione:** 1.0 (primo draft dopo run pulito cyclic)
