# Analisi Federated Learning: Flower Bagging vs Flower Cyclic vs NVIDIA FLARE

Questo documento riassume in modo operativo i risultati estratti dal notebook di confronto e dai file di output dei benchmark, con focus su:

- tempi per fase e colli di bottiglia di comunicazione
- accuratezza (MAE)
- metriche aggiuntive consigliate per arrivare a una decisione finale robusta

## 1) Obiettivo finale (decisione)

Obiettivo pratico: scegliere l'approccio FL migliore per il tuo scenario, bilanciando:

- rapidita' end-to-end
- costo di comunicazione e overhead di orchestrazione
- accuratezza finale
- stabilita' e ripetibilita' tra run

In altre parole, non basta il metodo piu' veloce in un singolo run: serve il miglior compromesso performance/qualita'/scalabilita'.

## 2) Dati usati per questa analisi

Fonti principali:

- `benchmarks/flower_bagging/results/timing_metrics.json`
- `benchmarks/flower_cyclic/results/timing_metrics.json`
- `benchmarks/nvidia_flare/results/timing_metrics.json`
- `benchmarks/nvidia_flare/run_nvflare.log`
- `results/flower_bagging_20260319_094213.csv`
- `analysis/compare_approaches.ipynb`

## 3) Risultati principali (tempi)

### 3.1 Confronto globale

| Approccio | Total Time (s) | FL Time (s) | Avg Round Time (s) | Round | Client |
|---|---:|---:|---:|---:|---:|
| Flower Cyclic | 19.621 | 19.589 | 1.959 | 10 | 9 |
| Flower Bagging | 26.317 | 26.259 | 2.626 | 10 | 9 |
| NVIDIA FLARE | 30.865 | 30.865 | 3.086 | 10 | 9 |

### 3.2 Lettura immediata

- Flower Cyclic è il piu' veloce in questo run.
- Flower Cyclic è circa il 25.4% piu' rapido di Flower Bagging sul tempo totale.
- NVIDIA FLARE è circa il 57.3% piu' lento di Flower Cyclic e circa il 17.3% più lento di Flower Bagging.

## 4) Bottleneck NVFLARE (da log)

Dal log NVFLARE:

- `client configuration took 5.6165 s`
- `client starting took 2.2809 s`
- `Request seq op='allreduce'` osservati: 1089

Scomposizione percentuale su tempo totale NVFLARE:

- client configuration: ~18.2%
- client starting: ~7.4%
- fl_core_estimated (resto): ~74.4%

Interpretazione:

- Il costo di setup client in NVFLARE non e' trascurabile (~25.6% combinato tra configurazione e start).
- L'elevato numero di allreduce e' coerente con una forte pressione sulla comunicazione intra-round.

## 5) Accuratezza (MAE): stato attuale

Nel CSV disponibile il campo `final_mae` risulta vuoto per Flower Bagging nel run letto.

Nel notebook e' quindi usato fallback manuale:

- Flower Bagging: 12.23
- Flower Cyclic: 11.38
- NVIDIA FLARE: non disponibile (NaN)

Conclusione sul punto accuratezza:

- Al momento il confronto MAE non e' conclusivo perche' manca un valore robusto per NVIDIA FLARE e manca un tracciamento consistente dei MAE nei file risultati.

## 6) Cosa puoi gia' concludere con buona confidenza

- Se la priorita' e' il tempo end-to-end nel setup corrente, Flower Cyclic e' il candidato migliore.
- NVFLARE mostra overhead di orchestrazione/comunicazione importante: per renderlo competitivo va misurato se porta un vantaggio di accuratezza o robustezza che compensi il costo tempo.

## 7) Metriche aggiuntive da monitorare (altamente consigliate)

Per il tuo obiettivo ultimo (scelta del miglior compromesso), queste metriche sono le piu' utili.

### 7.1 Metriche di accuratezza/qualita'

1. MAE medio e deviazione standard su piu' run (non un solo run).
2. RMSE e R2 oltre al MAE (per vedere sensibilita' agli outlier e varianza spiegata).
3. MAE per client (non solo globale): identifica client penalizzati.
4. Gap train vs validation (overfitting/fitting debole).

### 7.2 Metriche temporali piu' fini

1. Tempo per round (serie completa, non solo media): p50/p95/p99.
2. Tempo per fase per round:
   - fit locale
   - aggregazione server
   - serializzazione/deserializzazione
   - trasferimento rete
3. Straggler impact:
   - round waiting time (tempo perso in attesa del client piu' lento)
   - differenza tra client fastest/slowest.

### 7.3 Metriche di comunicazione (centrali per i bottleneck)

1. Byte scambiati per round e per client (upload + download).
2. Numero RPC/call per round e loro latenza media/p95.
3. Throughput effettivo (MB/s) durante federated rounds.
4. Retry/error rate di comunicazione.

### 7.4 Metriche di efficienza risorse

1. CPU mean/p95 per client e server durante i round.
2. Memoria peak e memory growth nel tempo (leak detection).
3. Eventuale I/O disco (se presente serializzazione pesante).

### 7.5 Metriche di robustezza e stabilita'

1. Ripetibilita' su N run (almeno 5): media, std, intervallo di confidenza.
2. Sensibilita' a numero client (scalabilita'): 3, 6, 9, 12 client.
3. Sensibilita' al data skew (non-IID piu' forte).
4. Fault tolerance:
   - client dropout simulato
   - ritardo artificiale di 1-2 client.

## 8) Raccomandazione pratica per il prossimo step

Per prendere una decisione finale solida, suggerisco questa pipeline minima:

1. Uniformare logging output tra i 3 approcci (stesso schema JSON/CSV).
2. Salvare sempre:
   - `final_mae`, `rmse`, `r2`
   - `round_time_list`
   - `bytes_up/down_per_round`
   - `cpu_mem_peak_per_client`
3. Ripetere ogni benchmark almeno 5 volte.
4. Confrontare con ranking multi-obiettivo:
   - 50% tempo
   - 35% accuratezza
   - 15% comunicazione/overhead

## 9) Limiti dell'analisi corrente

- Alcuni risultati di accuratezza sono fallback manuali (non completamente tracciati nei file finali).
- Le conclusioni temporali sono forti per questo setup specifico (9 client, 10 round), ma vanno confermate su run ripetuti.

## 10) Stato operativo

Il notebook `analysis/compare_approaches.ipynb` e' pronto per estendere il confronto; conviene ora aggiungere export strutturato delle metriche mancanti per chiudere il cerchio su accuratezza + comunicazione + tempo.
