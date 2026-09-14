# NVIDIA FLARE PoC

Questa cartella contiene il flusso PoC NVIDIA FLARE per il benchmark LS2.

Punti di ingresso:
- [run_benchmark.py](run_benchmark.py): prepara/avvia/ferma/pulisce il POC
- [metrics_pipeline.py](metrics_pipeline.py): normalizza metriche NVFlare e crea CSV comparabili con Flower

Nota sulla struttura: il job effettivo e' [jobs/xgb_fedavg_poc](jobs/xgb_fedavg_poc). Dentro quel job, i file usati da NVFlare sono solo `server_app/` e `client_app/`; la vecchia copia `app/` e gli altri residui legacy non fanno parte del flusso attivo.

## Workflow end-to-end

1) Attiva ambiente e prepara workspace

```bash
source venv/bin/activate
python benchmarks/nvidia_flare_poc/run_benchmark.py benchmark
```

Questo step:
- usa `results/nvflare_poc_workspace` come workspace POC locale
- esegue `nvflare config -pw ...`, `nvflare poc prepare -n 9`, `nvflare poc prepare-jobs-dir`
- salva manifest in `results/nvidia_flare_poc_manifest.json`
- stampa `run_id` da riusare per le metriche

2) Avvia servizi POC

```bash
python benchmarks/nvidia_flare_poc/run_benchmark.py start
```

Alternativa equivalente:

```bash
nvflare poc start
```

3) Sottometti il job NVFlare

La cartella jobs locale e':

```bash
benchmarks/nvidia_flare_poc/jobs
```

Con CLI NVFlare puoi usare, ad esempio:

```bash
nvflare job submit -j benchmarks/nvidia_flare_poc/jobs/<job_dir>
```

4) Ferma e pulisci (a fine run)

```bash
python benchmarks/nvidia_flare_poc/run_benchmark.py stop
python benchmarks/nvidia_flare_poc/run_benchmark.py clean
```

## Pipeline metriche comparabile con Flower

Dopo il run NVFlare, estrai e aggrega metriche con lo stesso schema Flower (`bytes_sent`, `bytes_received`, `train_time`, serialize/deserialize):

```bash
python benchmarks/nvidia_flare_poc/metrics_pipeline.py all \
	--run-id <run_id_stampato_dal_runner> \
	--source-dir results/nvflare_poc_workspace \
	--approach nvidia_flare_poc \
	--site-one-indexed
```

Output principali:
- `results/structured_metrics/runs/<run_id>/nvidia_flare_poc_client_<id>.jsonl`
- `results/structured_metrics/summaries/communication_round_summary_nvidia_flare_poc_<run_id>.csv`
- `results/structured_metrics/summaries/communication_approach_summary_nvidia_flare_poc_<run_id>.csv`
- `results/structured_metrics/summaries/communication_approach_summary_with_nvflare_<run_id>.csv` (merge con CSV Flower se presente)

## Allineamento metodologico con Flower

Per confronto equo Flower vs NVFlare, mantieni fissi:
- semantica round
- numero client attivi per round
- definizione metriche comunicazione (bytes sent/received, serialize/deserialize, train time)
