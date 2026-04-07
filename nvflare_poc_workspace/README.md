# NVFlare POC Workspace

Questa cartella contiene la workspace locale usata dal benchmark NVIDIA FLARE POC.

I file operativi vengono generati da `benchmarks/nvidia_flare_poc/run_benchmark.py` tramite:

- `nvflare config -pw nvflare_poc_workspace`
- `nvflare poc prepare -n 9`
- `nvflare poc prepare-jobs-dir -j benchmarks/nvidia_flare_poc/jobs`

Dopo il prepare, qui compariranno i contenuti del POC creati da NVFlare.
