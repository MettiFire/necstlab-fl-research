# NVFlare POC Workspace

Questa cartella e' mantenuta per compatibilita' con run precedenti.

La workspace predefinita attuale del benchmark NVIDIA FLARE POC e':

- `results/nvflare_poc_workspace`

I file operativi vengono generati da `benchmarks/nvidia_flare_poc/run_benchmark.py` tramite:

- `nvflare config -pw results/nvflare_poc_workspace`
- `nvflare poc prepare -n 9`
- `nvflare poc prepare-jobs-dir -j benchmarks/nvidia_flare_poc/jobs`

Dopo il prepare, qui compariranno i contenuti del POC creati da NVFlare.
