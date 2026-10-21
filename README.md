# Federated Learning Benchmark

## Progetto

Progetto di ricerca del **NECSTLab - Politecnico di Milano** dedicato allo studio di strategie di **Federated Learning** per la previsione della qualità del sonno a partire da dati biometrici raccolti tramite dispositivi Garmin.

Il progetto confronta approcci con diversi compromessi tra velocità, accuratezza, comunicazione e complessità del sistema:

- **Flower Bagging**: addestramento parallelo dei client e aggregazione dei modelli locali;
- **Flower Cyclic**: addestramento sequenziale, con il modello che passa da un client al successivo;
- **NVIDIA FLARE**: aggregazione federata XGBoost basata sulla condivisione di statistiche e istogrammi dei gradienti.

L'obiettivo generale è individuare un approccio intermedio che riduca l'overhead della comunicazione mantenendo una qualità predittiva vicina a quella ottenibile con un training federato più informativo.

---

## Obiettivi della ricerca

### 1. Confrontare le strategie di Federated Learning

Misurare e confrontare il comportamento dei tre approcci di riferimento mantenendo, per quanto possibile, la stessa configurazione sperimentale:

- stesso dataset federato;
- stesso modello di regressione basato su XGBoost;
- stesso numero di client e round federati;
- stessa suddivisione tra dati di addestramento e validazione.

Il confronto deve evidenziare non soltanto quale metodo sia più veloce, ma quale offra il miglior compromesso tra qualità del modello e costi operativi.

### 2. Analizzare i colli di bottiglia

Separare il costo complessivo del training nelle sue componenti principali:

- **addestramento locale**, eseguito sui singoli client;
- **comunicazione**, inclusi trasferimento e serializzazione dei dati o del modello;
- **aggregazione**, eseguita dal server o dal coordinatore federato;
- **costruzione degli alberi**, legata al calcolo dei gradienti e alle iterazioni di boosting;
- **orchestrazione**, cioè configurazione, avvio e sincronizzazione dei client.

Questa analisi serve a capire se le prestazioni dipendano principalmente dal calcolo locale, dalla rete o dalla logica di coordinamento.

### 3. Valutare qualità, efficienza e scalabilità

Il benchmark considera le seguenti dimensioni:

| Dimensione | Domanda di ricerca |
|---|---|
| Accuratezza | Quanto bene il modello prevede la qualità del sonno? |
| Tempo | Quanto tempo richiede il training federato completo? |
| Comunicazione | Quanti dati devono essere scambiati tra client e server? |
| Convergenza | Quanti round sono necessari per raggiungere una qualità target? |
| Scalabilità | Come cambiano tempi e costi aumentando il numero di client? |
| Robustezza | Quanto il metodo è stabile in presenza di dati non-IID, ritardi o client assenti? |

Le metriche principali sono **MAE**, **RMSE**, tempo totale e tempo per round, byte inviati e ricevuti, tempi di aggregazione e uso delle risorse.

---

## Contesto teorico

### Federated Learning

Nel Federated Learning i dati rimangono presso i client che li possiedono. Il coordinatore organizza il training e raccoglie soltanto modelli, aggiornamenti o statistiche aggregate. In questo modo è possibile addestrare un modello comune senza trasferire i dati biometrici grezzi a un server centrale.

Il problema è trovare il giusto equilibrio tra:

- quantità di informazione condivisa;
- accuratezza del modello globale;
- tempo di calcolo;
- banda e latenza della comunicazione;
- tutela dei dati locali.

La distribuzione dei dati del progetto è federata tra **9 client**. Ogni client contiene dati relativi a gruppi differenti di partecipanti e può quindi introdurre eterogeneità nella distribuzione delle osservazioni.

### Flower Bagging

Nel paradigma Bagging i client lavorano in parallelo sui propri dati e inviano al coordinatore i risultati del training locale. L'approccio limita la quantità di informazione scambiata e rende l'esecuzione relativamente semplice e veloce.

**Vantaggi teorici:**

- parallelismo tra client;
- comunicazione relativamente contenuta;
- implementazione e debug semplificati.

**Limiti teorici:**

- ogni client vede soltanto una parte dei dati;
- la media o aggregazione dei modelli può perdere informazione;
- la qualità può risentire maggiormente della distribuzione non-IID.

### Flower Cyclic

Nel paradigma Cyclic il modello viene aggiornato sequenzialmente dai client. Ogni client riceve il modello corrente, esegue il proprio training locale e lo trasferisce al client successivo o al coordinatore.

**Vantaggi teorici:**

- il modello può incorporare progressivamente l'informazione dei diversi client;
- il flusso di aggiornamento è diretto e leggero;
- il numero di modelli da aggregare può essere ridotto.

**Limiti teorici:**

- l'esecuzione è meno parallela;
- il tempo può dipendere dal client più lento;
- l'ordine dei client può influenzare il risultato finale.

### NVIDIA FLARE e aggregazione basata su istogrammi

NVIDIA FLARE utilizza un approccio più ricco per il Federated XGBoost. Invece di condividere soltanto il modello finale locale, i client possono condividere statistiche aggregate, come gli istogrammi dei gradienti, utili alla costruzione degli alberi.

**Vantaggi teorici:**

- maggiore quantità di informazione disponibile per l'aggregazione;
- comportamento più vicino al training centralizzato;
- possibilità di ottenere una migliore accuratezza su dati eterogenei;
- i dati biometrici grezzi non vengono trasferiti.

**Limiti teorici:**

- maggiore traffico di rete e numero di operazioni di sincronizzazione;
- costi più elevati di serializzazione e aggregazione;
- configurazione e orchestrazione più complesse;
- possibile aumento dei tempi di avvio e del tempo per round.

---

## Dataset e problema predittivo

Il dataset contiene dati biometrici raccolti da **45 partecipanti** che hanno utilizzato un Garmin Vivoactive 5 per circa un mese.

- **999 campioni di training** distribuiti su 9 client;
- **189 campioni di test**;
- circa **40 feature numeriche** derivate da serie temporali biometriche;
- target: **sleep quality score**, espresso su una scala 0-100.

Le feature descrivono, tra gli altri aspetti:

- frequenza cardiaca e variabilità;
- respirazione durante sonno e veglia;
- livelli e picchi di stress;
- attività, passi, calorie e intensità;
- durata e qualità del sonno precedente.

Il problema è trattato come una regressione con obiettivo XGBoost `reg:squarederror`. La valutazione deve considerare sia la qualità globale sia le differenze di errore tra i singoli client.

---

## Domanda di ricerca

> Esiste una strategia federata intermedia capace di mantenere una qualità predittiva vicina a NVIDIA FLARE, riducendo significativamente il tempo di esecuzione e l'overhead di comunicazione?

La domanda nasce dal contrasto tra:

- approcci Flower, più semplici e leggeri ma potenzialmente meno informativi;
- approcci histogram-based, più ricchi dal punto di vista statistico ma più costosi in termini di comunicazione e orchestrazione.

---

## Approcci ibridi da esplorare

Il progetto considera diverse possibili strategie intermedie:

1. **Selective histogram sharing**: condividere gli istogrammi soltanto per le feature più importanti;
2. **Adaptive communication**: usare più comunicazione nei primi round e ridurla dopo l'avvicinamento alla convergenza;
3. **Compressed histograms**: ridurre il numero di bin o quantizzare le statistiche condivise;
4. **Progressive histogram refinement**: iniziare con istogrammi grossolani e aumentarne progressivamente la precisione;
5. **Federated knowledge distillation**: addestrare un modello più ricco e trasferirne la conoscenza a modelli locali più leggeri;
6. **Hybrid parallel-sequential**: combinare gruppi di client addestrati in parallelo e sequenzialmente.

Ogni approccio ibrido deve essere valutato rispetto alle baseline considerando almeno:

- speedup rispetto a NVIDIA FLARE;
- variazione della MAE;
- rapporto tra byte comunicati e byte di NVIDIA FLARE;
- stabilità dei risultati su più esecuzioni;
- comportamento al variare del numero di client e del grado di eterogeneità dei dati.

---

## Risultato atteso

Il risultato finale non è semplicemente individuare il metodo più veloce o quello con la MAE più bassa. L'obiettivo è costruire una valutazione multi-obiettivo che permetta di scegliere la strategia più adatta in base a:

- accuratezza richiesta;
- tempo massimo disponibile;
- banda e latenza della rete;
- numero di client;
- livello di eterogeneità dei dati;
- requisiti di privacy e affidabilità.

Il progetto mira quindi a fornire una base sperimentale per progettare sistemi di Federated Learning efficienti per dati biometrici e scenari distribuiti reali.

---

## Riferimenti

- [Flower Documentation](https://flower.ai/docs/)
- [NVIDIA FLARE Documentation](https://nvflare.readthedocs.io/)
- [Federated XGBoost](https://arxiv.org/abs/1901.08755)
- [SecureBoost: Histogram-based Federated Learning](https://arxiv.org/abs/1901.08755)

**NECSTLab - Politecnico di Milano**  
Progetto di ricerca LS2 su Federated Learning
