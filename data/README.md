# Dataset Garmin - Sleep Quality Prediction

## Origine Dataset
Dati biometrici raccolti da 45 partecipanti che indossano Garmin Vivoactive 5 per circa un mese.

## Struttura
- **Training:** 999 campioni distribuiti su 9 client
- **Test:** 189 campioni (x_test.csv nel progetto originale)
- **Client:** 9 gruppi (5 utenti ciascuno)

## Features (40 totali)
Derivate da time-series biometriche:
- **Heart Rate:** min, max, average, resting, variability
- **Respiration:** durante sonno, durante veglia, rate medio
- **Stress:** livelli durante giorno, picchi, durata
- **Activity:** passi, calorie, intensità
- **Sleep:** durata, qualità precedente

## Valori Speciali
- `-1`: Valore mancante (sensore non disponibile)
- `-2`: Valore invalido (errore sensore)
- `null`: Gap nei dati

## Setup
Per usare il dataset in questo progetto:

### Opzione 1: Symlink (consigliato)
```bash
cd /Users/annamettifogo/Desktop/polimi/necstlab/progetto\ LS2/fl_benchmark/data
ln -s /Users/annamettifogo/Desktop/polimi/1°\ magistrale/csi/proj4/prova1/dtbagging/ready_for_flwr ready_for_flwr
ln -s /Users/annamettifogo/Desktop/polimi/1°\ magistrale/csi/proj4/prova1/dtbagging/x_test.csv x_test.csv
```

### Opzione 2: Copia
```bash
cp -r /Users/annamettifogo/Desktop/polimi/1°\ magistrale/csi/proj4/prova1/dtbagging/ready_for_flwr ./
cp /Users/annamettifogo/Desktop/polimi/1°\ magistrale/csi/proj4/prova1/dtbagging/x_test.csv ./
```

## File Attesi
```
data/
├── ready_for_flwr/
│   ├── client_0.csv
│   ├── client_1.csv
│   ├── ...
│   └── client_8.csv
└── x_test.csv
```

## Caricamento Dati
Vedi `utils/data_loader.py` per funzioni di caricamento standardizzate.
