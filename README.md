# llm-opinion-dynamics

Treat the LLM as a particle in a statistical-mechanics system: ask the same arithmetic question many times, vary the sampling temperature `T`, and study how the output distribution (single agent) and the inter-agent dynamics (a small "society" of agents) change as a function of both `T` and **model size**. The whole study sweeps the `qwen2.5` family from `0.5b` to `14b` via local Ollama.

The probe question is

```
T_c / e = 2.269 / 2.718 ≈ 0.8347
```

— a quantity neither memorisable nor a clean rounded constant — so the variance in the output isolates noise in the *computation step*, not knowledge uncertainty.

The headline finding: across the sweep, the genuine control parameter is **model size, not sampling temperature**. The Ising / Naming-Game analogy fits the small models (0.5B, 1.5B) and breaks down for the larger ones (≥3B), where the population is so confident that the inter-agent coupling becomes a null operation.

**See [`REPORT.md`](REPORT.md) for the full study with figures, tables, and discussion.**

---

## Repo contents

| File | Purpose |
|------|---------|
| [`REPORT.md`](REPORT.md) | The write-up: setup, results across `0.5b → 14b`, capacity-driven phase diagram interpretation. |
| `temperature_sweep.py` | Single-agent sweep. For a given model, samples `N` independent answers per temperature `T`, persists raw results and plots histograms / mean / discard-rate figures. |
| `opinion_dynamics.py` | Multi-agent sweep (CrewAI). For a given model and temperature, runs `N` agents over `R` rounds where each agent sees peers' previous-round answers and may revise. Persists per-round JSON + a per-(model, T) figure. |
| `aggregate_plots.py` | Cross-temperature aggregation: variance-vs-round panels and summary tables built from the per-run JSONs. |
| `send_job.sh` | SLURM submission template (CINECA Leonardo) — boots Ollama inside the allocation, then runs the two sweep scripts. Generic enough to adapt to other clusters; paths resolve from `SLURM_SUBMIT_DIR`. |
| `data/` | All raw outputs that back `REPORT.md`: histogram / mean / variance PNGs and per-(model, T) JSONs for every model in `{0.5b, 1.5b, 3b, 7b, 14b}` × every temperature in the grid. |

---

## Reproducing

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/download) running locally, with the qwen2.5 family pulled:
  ```bash
  for tag in 0.5b 1.5b 3b 7b 14b; do ollama pull qwen2.5:$tag; done
  ```
- A few Python packages: `requests`, `matplotlib`, `crewai`. Install with:
  ```bash
  pip install requests matplotlib crewai
  ```

### Single-agent sweep

```bash
python temperature_sweep.py \
    -model qwen2.5:7b \
    -temperatures 0.1 0.3 0.7 1.2 2.0 3.5 6.0 10.0 \
    -num_samples 150 \
    -workers 4
```

Writes raw rows to `data/temperature_sweep_qwen2.5_7b.json` and figures `histograms_qwen2.5_7b.png`, `mean_and_discards_qwen2.5_7b.png`.

### Multi-agent sweep

```bash
for T in 0.1 0.3 0.7 1.2 2.0 3.5 6.0 10.0; do
    python opinion_dynamics.py \
        -model qwen2.5:7b \
        -num_agents 4 \
        -num_rounds 10 \
        -temperature $T
done
```

Each invocation writes `data/opinion_dynamics_qwen2.5_7b_T<T>_N4.{json,png}`.

### Cross-temperature aggregation

After both sweeps have populated `data/`:

```bash
python aggregate_plots.py
```

Produces `data/variance_vs_round_<model>_N4.png` panels and the summary tables used in `REPORT.md`.

### On a SLURM cluster

`send_job.sh` is a working template for CINECA Leonardo (boost partition, 4 GPUs, exclusive node) that boots a private Ollama server inside the allocation. Adapt the `#SBATCH --account=` line and the cluster-specific module/conda lines and submit with `sbatch send_job.sh` from the repo root.

---

## Notes & caveats

- Single-agent: 150 samples × 8 temperatures × 5 models, `think` disabled, `num_predict = 16`.
- Multi-agent: `N = 4` agents, `R = 10` rounds, peer answers from previous round (self excluded). A round-snapshot read makes the within-round dispatch parallel.
- Seeds are not fixed; the per-run JSONs in `data/` are the authoritative artefact and `aggregate_plots.py` rebuilds every figure from them.
- For model sizes ≥ 3B, the population locks onto the correct answer at round 0 at every temperature — the multi-agent coupling has nothing to act on. The "phase transition" picture only holds for the smaller models.

## License

MIT — see `LICENSE`.
