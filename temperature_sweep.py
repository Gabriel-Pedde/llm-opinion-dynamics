# Project 2 - Physics of Agents: Opinion Dynamics in an LLM Population
#
# Single-agent baseline for the opinion-dynamics study: the same numerical
# question (T_c / e for the 2D Ising model) is asked `num_samples` times
# *independently* at each sampling temperature. The temperature acts like
# a thermodynamic temperature on the token distribution — low T concentrates
# probability on the argmax, high T flattens it — so we expect the spread
# (and the discard rate from malformed replies) to grow with T.

import json
import re
import requests
import os
from concurrent.futures import ThreadPoolExecutor

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def _model_slug(model: str) -> str:
    model_tag = model.split("/", 1)[1] if model.startswith("ollama/") else model
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_tag)

# Pooled HTTP session — reuses the TCP connection across every chat() call,
# saving the per-request handshake (tens of ms on localhost, more on remote).
_SESSION = requests.Session()


# Chat method to submit a prompt to Ollama and get the assistant's reply as a string.
def chat(messages: list[dict],
         model: str = "qwen3.6:35b",
         temperature: float = 0.7,
         url: str = "http://localhost:11438/api/chat",
         timeout: int = 300) -> str:
    """One round-trip to the Ollama /api/chat endpoint.

    Args:
        messages: list of {'role': 'system'|'user'|'assistant', 'content': str}
        model: Ollama model tag
        temperature: softmax scaling of the token distribution (>=0)

    Returns:
        The assistant's reply as a plain string.
    """
    payload = {
        "model":      model,
        "messages":   messages,
        "stream":     False,
        # `think: false` disables the chain-of-thought phase on thinking models
        # (qwen3, deepseek-r1, ...).
        "think":      False,
        # Pin the model in VRAM so subsequent calls skip the reload cost.
        "keep_alive": "30m",
        "options":    {
            "temperature": temperature,
            # Prompt is ~60 tokens; 512 is plenty and shrinks the KV cache.
            "num_ctx":     512,
            "num_predict": 16,
            # Cut generation as soon as the model emits a newline pair —
            # the answer is a single float, anything after is wasted tokens.
            "stop":        ["\n\n"],
        },
    }
    r = _SESSION.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["message"]["content"]


_FLOAT_DECIMAL = re.compile(r"-?\d+\.\d+(?:[eE][-+]?\d+)?")
_FLOAT_ANY     = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def parser(messages: list[dict]) -> float | None:
    """Extract a numeric value from the assistant's last reply.

    Handles bare numbers, leading prose ("approximately 0.835"), prefixes
    ("T_c/e = 0.83", "≈ 0.83"), and trailing punctuation. Returns None when
    no number can be extracted so the caller can record a discard.
    """
    text = messages[-1]["content"]
    m = _FLOAT_DECIMAL.search(text) or _FLOAT_ANY.search(text)
    if m is None:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def histogram(data: list[float], bins: int = 20) -> dict:
    """Compute a histogram of the data.

    Bins are evenly spaces within the fixed range [0.5, 1.2] and samples outside that range are dropped.
    Returns a dict with the bin edges and counts, suitable for plotting as a bar chart.     
    """
    if not data:
        return {"bins": [], "counts": []}

    # Fixed range centred on the true value (~0.8347). Hard-coding the bounds
    # keeps every per-temperature subplot on the same x-axis, which makes the
    # broadening-with-T pattern visible at a glance. Samples outside [0.5, 1.2]
    # are dropped — they are almost always parser noise (e.g. extracting the
    # leading digit of "2.269") rather than legitimate answers.
    min_val, max_val = 0.5, 1.2
    bin_edges = [min_val + i * (max_val - min_val) / bins for i in range(bins + 1)]
    counts = [0] * bins
    for x in data:
        if x < min_val or x > max_val:
            continue
        for i in range(bins):
            if bin_edges[i] <= x < bin_edges[i + 1]:
                counts[i] += 1
                break
    return {"bins": bin_edges, "counts": counts}

# Requires scikit-learn for GaussianMixture.
def plot_gmm(estimates: list[float], hist: dict, k_max: int = 4) -> dict | None:
    """Fit a Gaussian Mixture Model with BIC-selected K and overlay it.

    Motivation: at intermediate temperatures the response distribution is
    often multi-modal — e.g. a peak at the correct 0.834 and another at a
    rounded 0.83 or a memorised 0.85 — and a single mean+std hides this.
    BIC penalises extra components, so K is chosen automatically and a
    unimodal histogram does not get over-fit.

    Each component is plotted as its own Gaussian, scaled so the peak
    height equals the histogram count of the bin containing the
    component mean. This keeps the curves visually comparable to the
    bars instead of using a density-rescaled mixture PDF.

    Returns a dict with the chosen K, weights, means, stds, and BIC,
    or None if a fit was not possible.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.mixture import GaussianMixture

    X = np.asarray(estimates, dtype=float).reshape(-1, 1)
    n = len(X)
    if n < 2 or np.ptp(X) == 0:
        return None

    k_upper = min(k_max, n, len(set(estimates)))
    best_gmm, best_bic, best_k = None, np.inf, 1
    for k in range(1, k_upper + 1):
        try:
            gmm = GaussianMixture(n_components=k, covariance_type='full',
                                  random_state=0, n_init=3).fit(X)
            bic = gmm.bic(X)
            if bic < best_bic:
                best_bic, best_gmm, best_k = bic, gmm, k
        except Exception:
            continue

    if best_gmm is None:
        return None

    bin_edges = hist["bins"]
    counts    = hist["counts"]
    n_bins    = len(counts)

    def bin_count_at(mu: float) -> int:
        if mu < bin_edges[0]:
            return counts[0]
        if mu >= bin_edges[-1]:
            return counts[-1]
        for j in range(n_bins):
            if bin_edges[j] <= mu < bin_edges[j + 1]:
                return counts[j]
        return counts[-1]

    means_arr   = best_gmm.means_.ravel()
    stds_arr    = np.sqrt(best_gmm.covariances_.ravel())
    weights_arr = best_gmm.weights_

    for i, (mu, sigma) in enumerate(zip(means_arr, stds_arr)):
        target = bin_count_at(mu)
        if target <= 0:
            continue
        x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
        y = target * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        label = f'GMM (K={best_k})' if i == 0 else None
        plt.plot(x, y, color='purple', label=label)
        plt.axvline(mu, color='green', linestyle='--', alpha=0.5)

    return {
        "k": best_k,
        "weights": weights_arr.tolist(),
        "means":   means_arr.tolist(),
        "stds":    stds_arr.tolist(),
        "bic":     float(best_bic),
    }


import matplotlib.pyplot as plt
import argparse # For command-line arguments.
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    argp = argparse.ArgumentParser(description="Temperature sweep for LLM estimates of T_c / e.")
    argp.add_argument("-model", type=str, default="qwen3.6:35b", help="Ollama model tag to use for the sweep.")
    argp.add_argument("-temperatures", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0], help="List of temperatures to test.")
    argp.add_argument("-num_samples", type=int, default=50, help="Number of independent estimates to collect per temperature.")
    argp.add_argument("-workers", type=int, default=4, help="Parallel /api/chat requests per temperature.")
    args = argp.parse_args()

    MODEL = args.model
    T = args.temperatures
    num_samples = args.num_samples
    WORKERS = args.workers
    TRUE_VALUE = 0.8347

    # Prompt to be submitted. Both numerical inputs are spelled out in the
    # user message so the task is purely arithmetic (2.269 / 2.718 = 0.8347...);
    # the model is not being tested on whether it remembers the constants.
    # "/no_think" disables chain-of-thought on Qwen3-style thinking models.
    messages = [
        {"role": "system", "content": "You are a concise physics tutor."},
        {"role": "user",   "content": "/no_think The critical temperature of the 2D Ising model is T_c = 2.269 and Euler's number is e = 2.718. Compute the ratio T_c / e. Give only a numerical value, without explanation."
         },
    ]

    print("Model used:", MODEL)
    means    = []
    stds     = []
    discards = []
    biases   = []
    raw_rows = []
    count = 0

    plt.figure(figsize=(14, 10))
    plt.suptitle(f"Histograms of Estimates at Different Temperatures, Model: {MODEL}  Num Samples: {num_samples}", fontsize=16)
    def _one_sample(args):
        run_index, temperature = args
        raw_text = chat(messages, model=MODEL, temperature=temperature)
        reply = parser([{"role": "assistant", "content": raw_text}])
        return run_index, raw_text, reply

    for temp in T:
        print(f"Collecting estimates for temperature {temp}...")
        count += 1
        estimates = []
        discard_rate = 0
        # Fire `WORKERS` /api/chat requests at a time. Samples are i.i.d.
        # so order doesn't matter; we sort by run_index when persisting.
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            results = list(ex.map(_one_sample, [(i, temp) for i in range(num_samples)]))
        for run_index, raw_text, reply in sorted(results, key=lambda r: r[0]):
            raw_rows.append({
                "temperature":     temp,
                "run_index":       run_index,
                "raw_text":        raw_text,
                "extracted_value": reply,
            })
            if reply is not None:
                estimates.append(reply)
            else:
                discard_rate += 1
        discard_rate /= num_samples
        # Discard rate = fraction of replies the parser could not turn into a
        # number. It is itself a soft proxy for sampling noise — high T tends
        # to produce more verbose / off-format outputs.
        mean = sum(estimates) / len(estimates) if estimates else float('nan')
        std = (sum((x - mean) ** 2 for x in estimates) / len(estimates)) ** 0.5 if estimates else float('nan')
        means.append(mean)
        stds.append(std)
        discards.append(discard_rate)
        biases.append(mean - TRUE_VALUE if estimates else float('nan'))
        histogram_data = histogram(estimates)
        print(f"Temperature: {temp:.1f}, Mean Estimate: {means[-1]:.4f}, Std Dev: {stds[-1]:.4f}, Discard Rate: {discard_rate:.2%}")
        plt.subplot(2, len(T) // 2 + len(T) % 2, count)
        plt.bar(histogram_data["bins"][:-1], [int(x) for x in histogram_data["counts"]], width=(histogram_data["bins"][1] - histogram_data["bins"][0]), align='edge')

        if stds[-1] > 0 and len(estimates) >= 2:
            gmm_info = plot_gmm(estimates, histogram_data)
            if gmm_info is not None:
                print(f"  GMM K={gmm_info['k']}, means={[f'{m:.4f}' for m in gmm_info['means']]}, "
                      f"weights={[f'{w:.2f}' for w in gmm_info['weights']]}, BIC={gmm_info['bic']:.1f}")
        plt.axvline(TRUE_VALUE, color='red', linestyle='--', label='True Value')
        plt.legend()
        plt.title(f"Temperature: {temp:.1f}")
        plt.xlabel("Estimated Value")
        plt.ylabel("Count")
        plt.tight_layout()
    model_slug = _model_slug(MODEL)
    plt.savefig(os.path.join(DATA_DIR, f"histograms_{model_slug}.png"))

    # Plot 1 (spec): Mean estimate +/- std vs T, with the true value as a horizontal line.
    plt.figure(figsize=(14, 5))
    plt.suptitle(f"Mean Estimate and Discard Rate vs Temperature, Model: {MODEL}  Num Samples: {num_samples}", fontsize=14)
    plt.subplot(1, 2, 1)
    plt.errorbar(T, means, yerr=stds, marker='o', color='blue', ecolor='lightgray',
                 elinewidth=3, capsize=1, capthick=2, barsabove=True)
    plt.axhline(TRUE_VALUE, color='red', linestyle='--', label=f'True Value ({TRUE_VALUE})')
    plt.legend()
    plt.title("Mean Estimate vs Temperature")
    plt.xlabel("Temperature")
    plt.ylabel("Estimate of T_c / e")
    plt.subplot(1, 2, 2)
    plt.plot(T, discards, marker='o', color='red')
    plt.title("Discard Rate vs Temperature")
    plt.xlabel("Temperature")
    plt.ylabel("Discard Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, f"mean_and_discards_{model_slug}.png"))

    summary = {
        str(temp): {
            "mean":         means[i],
            "std":          stds[i],
            "discard_rate": discards[i],
            "bias":         biases[i],
        }
        for i, temp in enumerate(T)
    }
    out_path = os.path.join(DATA_DIR, f"temperature_sweep_{model_slug}.json")
    with open(out_path, "w") as f:
        json.dump({
            "model":        MODEL,
            "num_samples":  num_samples,
            "temperatures": T,
            "true_value":   TRUE_VALUE,
            "rows":         raw_rows,
            "summary":      summary,
        }, f, indent=2)
    print(f"Wrote raw data to {out_path}")

