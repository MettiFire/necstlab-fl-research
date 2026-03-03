# Flower Cyclic Benchmark

## TODO: Da Implementare

Questa directory conterrà l'implementazione del benchmark per **Flower Cyclic**.

## Differenza vs Bagging

**Bagging:**
```
Round 1: Client0→Model0, Client1→Model1, ..., Client8→Model8
         Server: Average(Model0...Model8) → GlobalModel
```

**Cyclic:**
```
Round 1: Client0 → Model_v1
Round 2: Model_v1 → Client1 → Model_v2
Round 3: Model_v2 → Client2 → Model_v3
...
Round 9: Model_v8 → Client8 → Model_v9
```

## Implementazione Richiesta

### File da creare:
1. `client.py` - Client Flower con strategia cyclic
2. `server.py` - Server Flower con FedXgbCyclic
3. `run_benchmark.py` - Script esecuzione e profiling

### Strategia
Usare `FedXgbCyclic` di Flower invece di `FedXgbBagging`.

### Differenze Implementative

```python
# In server.py
from flwr.serverapp.strategy import FedXgbCyclic

strategy = FedXgbCyclic()  # Invece di FedXgbBagging
```

```python
# In client.py
if train_method == "cyclic":
    bst = bst_input  # Ritorna modello completo (non slice)
```

## Aspettative Performance

**Tempo:** 
- Più lento di bagging (sequenziale vs parallelo)
- 9x tempo singolo client (se perfettamente sequenziale)

**Accuratezza:**
- Potenzialmente migliore di bagging (ogni client vede "conoscenza" dei precedenti)
- Ma: rischio di overfitting se dati client troppo diversi

## Reference

Vedi implementazione bagging in `../flower_bagging/` per struttura analoga.
