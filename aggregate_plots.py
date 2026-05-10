"""Aggregate variance-vs-round across saved opinion-dynamics runs.

`opinion_dynamics.py` writes one JSON per (model, temperature, N) run.
This script is the post-processing step: it globs every
`opinion_dynamics*.json` in `./data/`, groups the runs by (model, N),
and emits one PNG per group with the per-round variance plotted as
a function of round number, one curve per temperature.

The output is the main diagnostic for whether a population reaches
consensus (variance → 0) or stays scattered (variance flat or
growing). On the symlog y-axis a frozen run (σ² = 0) and a turbulent
run (σ² ≈ 1) are visible on the same plot.
"""

import json
import math
import re
import glob
from pathlib import Path

import matplotlib.pyplot as plt


DATA_DIR = Path(__file__).resolve().parent / "data"


def _model_slug(model: str) -> str:
    """Turn an Ollama model name into a filesystem-safe slug.

    CrewAI prefixes Ollama model names with `ollama/` for litellm; we drop
    that. The remaining tag may contain `:` (e.g. `qwen2.5:7b`) which is
    invalid on some filesystems, so we squash any non-`[A-Za-z0-9._-]`
    char to an underscore. This must match the slug used by
    `opinion_dynamics.py` for the per-run JSON/PNG so the globbing in
    `load_runs` actually picks the files up.
    """
    model_tag = model.split("/", 1)[1] if model.startswith("ollama/") else model
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_tag)


def load_runs(pattern: str = str(DATA_DIR / "opinion_dynamics*.json")) -> list[dict]:
    """Load every saved opinion-dynamics run as a list of dicts.

    Each JSON is expected to be the dump produced by
    `opinion_dynamics.py:__main__`, so it carries the keys we rely on
    later: `model`, `temperature`, `num_agents`, `variances` (one entry
    per round, including round 0). Sorted by path for deterministic
    output filenames when several runs share the same (model, N).
    """
    runs = []
    for p in sorted(glob.glob(pattern)):
        with open(p) as f:
            runs.append(json.load(f))
    return runs


def plot_variance_by_temperature(runs: list[dict], model: str, num_agents: int, log_scale: bool = True) -> Path | None:
    """Plot variance(round) for one (model, N) group, one curve per T.

    Returns the path of the written PNG, or None if no run matches the
    requested (model, num_agents) filter — the caller uses this to skip
    "empty" combinations silently when sweeping over the full grid.
    """
    # Filter the global run list down to the requested (model, N) and
    # order by temperature so the legend reads from cold to hot.
    selected = sorted(
        (r for r in runs if r.get("num_agents") == num_agents and r.get("model") == model),
        key=lambda r: r["temperature"],
    )
    if not selected:
        return None

    plt.figure(figsize=(8, 5))
    for r in selected:
        # NaN-safe pass-through: a failed round (all parses returned NaN
        # so variance is NaN) is plotted as a gap rather than crashing.
        variances = [v if (v is not None and not math.isnan(v)) else float("nan")
                     for v in r["variances"]]
        # X-axis is the round index. variances[0] is round 0 (independent
        # answers, no peer exposure yet), variances[k] is after k debate
        # rounds.
        rounds = list(range(len(variances)))
        plt.plot(rounds, variances, marker="o", label=f"T={r['temperature']}")

    if log_scale:
        # symlog (not pure log) lets exact-zero variance — very common
        # for ≥7b models that reach perfect consensus — plot at y=0
        # instead of being silently dropped. linthresh=1e-8 sets the
        # width of the linear region around zero; below that everything
        # collapses to the axis, above it the scale is logarithmic.
        plt.yscale("symlog", linthresh=1e-8)
    plt.xlabel("Round")
    plt.ylabel("Variance across agents" + (" (symlog)" if log_scale else ""))
    plt.title(f"Variance vs round, model={model}, N={num_agents}")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    # Filename mirrors the per-run convention (slug + N) so a sweep
    # over models and Ns produces non-colliding outputs.
    out = DATA_DIR / f"variance_vs_round_{_model_slug(model)}_N{num_agents}.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out


if __name__ == "__main__":
    # 1. Load every run currently saved on disk.
    runs = load_runs()
    if not runs:
        print(f"No opinion_dynamics_T*_N*.json files found in {DATA_DIR}.")
        raise SystemExit(0)

    # 2. Discover the (model, N) grid actually present in the data —
    #    we don't hard-code it, so adding a new model or rerunning at
    #    a new N just shows up the next time this script runs.
    Ns = sorted({r["num_agents"] for r in runs})
    models = sorted({r["model"] for r in runs if r.get("model")})
    print(f"Loaded {len(runs)} runs covering N values: {Ns} and models: {models}")

    # 3. Emit one figure per (model, N). Empty cells (e.g. model X has
    #    no N=2 runs) silently return None and are skipped.
    for model in models:
        for N in Ns:
            out = plot_variance_by_temperature(runs, model, N)
            if out is not None:
                print(f"  wrote {out}")
