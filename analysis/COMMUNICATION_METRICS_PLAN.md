# Communication Metrics Plan

## Goal

Misurare e confrontare il costo di comunicazione e trasmissione per:

- Flower Bagging
- Flower Cyclic

## Flow

1. Ogni client misura per round:
- bytes ricevuti dal server
- bytes inviati al server
- tempo di deserializzazione
- tempo di serializzazione
- communication time proxy = deserialize + serialize

2. Ogni client salva una riga JSONL in:
- results/structured_metrics/flower_bagging_client_<id>.jsonl
- results/structured_metrics/flower_cyclic_client_<id>.jsonl

3. Lo script di report aggrega i JSONL e produce:
- CSV round-level
- CSV approach-level
- grafico comparativo PNG

## Quick Run

Esegui benchmark Bagging e Cyclic, poi:

python analysis/communication_metrics_report.py

Output principali:

- results/structured_metrics/communication_round_summary.csv
- results/structured_metrics/communication_approach_summary.csv
- results/plots/communication_bagging_vs_cyclic.png

## Notes

Il tempo di rete puro non e misurabile direttamente in simulation locale.
Per questo usiamo un proxy robusto lato client:

communication_time_proxy = deserialize_time + serialize_time

In modalita PoC su macchina lab, questo proxy diventa piu vicino al costo reale end-to-end.
