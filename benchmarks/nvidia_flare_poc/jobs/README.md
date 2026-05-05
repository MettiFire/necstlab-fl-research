# NVFlare Jobs

Metti qui i job NVFlare usati dal PoC.

Quando il contenuto e' pronto, rilancia:

```bash
nvflare poc prepare-jobs-dir -j benchmarks/nvidia_flare_poc/jobs
```

Per rendere le metriche confrontabili con Flower, fai in modo che il job produca
record JSON/JSONL con almeno questi campi (nomi consigliati):

- `round`
- `client_id` oppure `site_id`
- `bytes_sent`
- `bytes_received`
- `train_time`
- `serialize_time`
- `deserialize_time`
- `total_time`

Poi normalizza e aggrega con:

```bash
python benchmarks/nvidia_flare_poc/metrics_pipeline.py all --run-id <run_id>
```