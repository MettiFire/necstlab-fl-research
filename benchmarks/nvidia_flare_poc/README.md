# NVIDIA FLARE PoC

Questa cartella contiene la versione "proof of concept" (PoC) del benchmark NVIDIA FLARE, con workspace locale dentro il repository per renderla visibile e gestibile da VS Code.

Il punto di ingresso è [run_benchmark.py](run_benchmark.py).

## Cosa fa il runner

- legge la configurazione comune da [config.yaml](../../config.yaml)
- prepara la workspace NVFlare dentro `nvflare_poc_workspace`
- esegue `nvflare config -pw ...`, `nvflare poc prepare` e `nvflare poc prepare-jobs-dir`
- salva un manifest con i parametri della run in `results/nvidia_flare_poc_manifest.json`
- stampa i comandi corretti per `start`, `stop` e `clean`

## Uso

```bash
source venv/bin/activate
python benchmarks/nvidia_flare_poc/run_benchmark.py
```

Comandi opzionali:

```bash
python benchmarks/nvidia_flare_poc/run_benchmark.py prepare
python benchmarks/nvidia_flare_poc/run_benchmark.py start
python benchmarks/nvidia_flare_poc/run_benchmark.py stop
python benchmarks/nvidia_flare_poc/run_benchmark.py clean
```

## Flusso NVFlare POC

Il runner usa questo flusso:

```bash
nvflare config -pw nvflare_poc_workspace
nvflare poc prepare -n 9
nvflare poc prepare-jobs-dir -j benchmarks/nvidia_flare_poc/jobs
nvflare poc start
```

In questa versione della CLI, `nvflare poc start` avvia il workspace POC locale. Se vuoi fermare o pulire:

```bash
nvflare poc stop
nvflare poc clean
```

## Nota sulla workspace

La workspace viene creata nel progetto, non in `/tmp`, così puoi vederla in VS Code e tenere tutto sotto controllo dentro `fl_benchmark`.
