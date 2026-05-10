# Part 2 - Multi-Agent Opinion Dynamics
#
# Several LLM "agents" are asked the same numerical question (T_c / e for the
# 2D Ising model). After an independent round 0, every later round each agent
# sees the others' previous estimates and is invited to revise its own. We
# track the per-round variance to see whether the population converges
# (consensus) or stays scattered, as a function of sampling temperature.

import os
os.environ["OTEL_SDK_DISABLED"]        = "true"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

import warnings
warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def _model_slug(model: str) -> str:
    model_tag = model.split("/", 1)[1] if model.startswith("ollama/") else model
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_tag)

BASE_URL = "http://localhost:11438"

import re
import requests
from concurrent.futures import ThreadPoolExecutor
from crewai import Agent, Task, Crew, Process, LLM

# Pooled HTTP session reused by every direct Ollama call below.
_SESSION = requests.Session()


def _direct_chat(model: str, system: str, user: str, temperature: float,
                 num_predict: int = 32, timeout: int = 300) -> str:
    """Direct call to Ollama /api/chat, bypassing CrewAI's Task/Crew machinery.

    The debate question is small ("give one float") so CrewAI's per-task
    overhead (instantiating Task+Crew, telemetry plumbing, agent
    orchestration) dominates the actual inference for fast models, and
    for slow models it blocks the parallelism we want to add. Going
    straight to the HTTP endpoint lets us:
      * disable the chain-of-thought phase on thinking models (think=False)
      * cap output to ~32 tokens (one float) instead of 4000
      * keep the model resident in VRAM between calls
      * fire all agents in a round concurrently from a thread pool
    """
    # CrewAI prefixes the model with "ollama/" for litellm; strip it for
    # the native Ollama endpoint.
    model_tag = model.split("/", 1)[1] if model.startswith("ollama/") else model
    payload = {
        "model":      model_tag,
        "messages":   [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream":     False,
        "think":      False,
        "keep_alive": "30m",
        "options": {
            "temperature": temperature,
            "num_ctx":     1024,
            "num_predict": num_predict,
            "stop":        ["\n\n"],
        },
    }
    r = _SESSION.post(f"{BASE_URL}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["message"]["content"]

_FLOAT_DECIMAL = re.compile(r"-?\d+\.\d+(?:[eE][-+]?\d+)?")
_FLOAT_ANY     = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _parse_float(text: str) -> float:
    """Extract a numerical value from a model output string. Prefer a decimal."""
    match = _FLOAT_DECIMAL.search(text) or _FLOAT_ANY.search(text)
    if match is None:
        return float('nan')
    return float(match.group(0))

def variance_of_estimates(estimates: list[float]) -> float:
    """Compute the variance of the estimates.

    Variance is the order parameter of the simulation: a vanishing variance
    flags consensus, a persistently large variance flags polarization.
    """
    if not estimates:
        return float('nan')
    # NaN-safe mean: `x == x` filters out failed parses without importing numpy.
    mean = sum([x for x in estimates if x == x]) / len(estimates)
    return sum((x - mean) ** 2 for x in estimates) / len(estimates)


def _persona_system(agent) -> str:
    """Build the system prompt from a CrewAI Agent's role/goal/backstory."""
    return (
        f"You are a {agent.role}. {agent.backstory}\n"
        f"Goal: {agent.goal}\n"
        "Answer with a single numerical value only, no explanation, no units."
    )


def run_debate(agents: list, question: str, num_rounds: int = 4,
               threshold: float = 0.0001, temperature: float = 0.7):
    """Run a multi-agent opinion dynamics simulation.

    Agents within a round are queried *in parallel* via a thread pool: at
    round k each agent only sees round k-1's outputs, so the N calls are
    independent. Wall time per round drops from O(N * t_inference) to
    roughly t_inference (assuming OLLAMA_NUM_PARALLEL >= N on the server).

    Returns (history, variances, consensus_round). `variances[k]` is the
    population variance of the N estimates at round k. `consensus_round` is
    the first round whose variance drops below `threshold`, or None.
    """
    history = {}
    variances = []
    consensus_round = None

    names = [f"Agent{i+1}" for i in range(len(agents))]
    # Cache the model tag once; every agent shares the same LLM here.
    model = agents[0].llm.model if agents else ""

    def _ask(agent_idx: int, user_prompt: str) -> float:
        agent = agents[agent_idx]
        raw = _direct_chat(
            model=model,
            system=_persona_system(agent),
            user=user_prompt,
            temperature=temperature,
        )
        return _parse_float(raw)

    def _run_round(round_num: int, prompt_for: callable) -> dict:
        with ThreadPoolExecutor(max_workers=len(agents)) as ex:
            futures = {
                ex.submit(_ask, i, prompt_for(i)): names[i]
                for i in range(len(agents))
            }
            return {futures[f]: f.result() for f in futures}

    # Round 0: independent answers — every agent sees the same prompt.
    history[0] = _run_round(0, lambda i: question)
    variances.append(variance_of_estimates(list(history[0].values())))
    if consensus_round is None and variances[-1] < threshold:
        consensus_round = 0

    # Rounds 1..num_rounds: each agent sees the others' last estimates.
    for round_num in range(1, num_rounds + 1):
        prev = history[round_num - 1]

        def _prompt(i: int, _prev=prev) -> str:
            others = "\n".join(
                f"{other}: {answer:.4f}"
                for other, answer in _prev.items()
                if other != names[i]
            )
            return (
                f"Your colleagues computed T_c / e as:\n{others}\n\n"
                "Consider their answers and give your own best estimate. "
                "Give only a single numerical value."
            )

        history[round_num] = _run_round(round_num, _prompt)
        variances.append(variance_of_estimates(list(history[round_num].values())))
        if consensus_round is None and variances[-1] < threshold:
            consensus_round = round_num

    return history, variances, consensus_round



def plot_from_history(history: dict, temperature: float, num_agents: int, model: str):
    """Plot the opinion dynamics from the history dictionary returned by run_debate.

    The output filename embeds T and N so multi-run sweeps don't overwrite
    each other (matches the JSON naming convention).
    """
    import matplotlib.pyplot as plt

    rounds = sorted(history.keys())
    names = sorted(next(iter(history.values())).keys())

    plt.figure(figsize=(15, 10))
    for name in names:
        estimates = [history[round][name] for round in rounds]
        plt.plot(rounds, estimates, marker='o', label=name)

    plt.xlabel("Round")
    plt.ylabel("Estimate of T_c / e")
    plt.title(f"Opinion Dynamics Simulation (T={temperature}, N={num_agents})")
    plt.axhline(0.8347, color='red', linestyle='--', alpha=0.5, label="True Value (T_c / e)")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(DATA_DIR, f"opinion_dynamics_{_model_slug(model)}_T{temperature}_N{num_agents}.png"))
        


import argparse
import json
import time
# ----------------------------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Opinion dynamics simulation with CrewAI agents.")
    parser.add_argument("-model", type=str, default="qwen2.5:1.5b", help="Ollama model tag to use for the simulation.")
    parser.add_argument("-num_agents", type=int, default=2, help="Number of agents in the simulation.")
    parser.add_argument("-num_rounds", type=int, default=4, help="Number of rounds to simulate.")
    parser.add_argument("-temperature", type=float, default=0.7, help="Sampling temperature; nonzero is required for diverse opinions.")
    args = parser.parse_args()

    MODEL = "ollama/" + args.model
    NUM_AGENTS = args.num_agents
    NUM_ROUNDS = args.num_rounds
    TEMPERATURE = args.temperature

    # ---------------------------------------------------------------------------
    #  Setup
    # ---------------------------------------------------------------------------
    llm = LLM(
        model=MODEL,
        base_url=BASE_URL,
        temperature=TEMPERATURE,
        # Single float fits in a handful of tokens. 
        max_tokens=32,
    )


    # A homogeneous population: every agent shares the same persona, so any
    # disagreement comes from sampling stochasticity rather than from
    # differing priors baked into the prompt.
    PERSONAS = [
        ("Pedantic Physicist",
         "You compute carefully to four decimal places and refuse to round.")
    ]

    agents = []
    for i in range(NUM_AGENTS):
        role, backstory = PERSONAS[i % len(PERSONAS)]
        agents.append(Agent(
            llm=llm,
            role=role,
            goal="Estimate the value of T_c / e for the 2D Ising model, where T_c is the critical temperature and e is Euler's number.",
            backstory=backstory,
        ))
    print(f"LLM: {llm.model}")

    # Both inputs are supplied in the prompt so the task reduces to a single
    # division (2.269 / 2.718 ~= 0.8347). The "/no_think" prefix suppresses
    # the chain-of-thought phase on Qwen3-style thinking models.
    question = "/no_think The critical temperature of the 2D Ising model is T_c = 2.269 and Euler's number is e = 2.718. " \
               "Compute the ratio T_c / e. " \
               "Give only a numerical value, without explanation."

    t0 = time.time()
    history, variances, consensus_round = run_debate(
        agents=agents, question=question, num_rounds=NUM_ROUNDS,
        temperature=TEMPERATURE,
    )
    dt = time.time() - t0
    print(f"\n=== Crew finished in {dt:.1f}s ===\n")

    # 2.269 / 2.718 = 0.8347... — the reference value the population should
    # converge to if the model can reliably divide the two given numbers.
    TRUE_VALUE = 0.8347
    final_round = max(history.keys())
    final_estimates = list(history[final_round].values())
    final_mean = sum(final_estimates) / len(final_estimates)
    final_accuracy = abs(final_mean - TRUE_VALUE)

    print(f"Final estimates (round {final_round}): {history[final_round]}")
    print(f"Final mean: {final_mean:.4f}  |  |mean - {TRUE_VALUE}| = {final_accuracy:.4f}")
    print(f"Variance per round: {[round(v, 6) for v in variances]}")
    print(f"Consensus round: {consensus_round}")

    rows = [
        {
            "temperature": TEMPERATURE,
            "num_agents":  NUM_AGENTS,
            "round":       round_num,
            "agent_id":    agent_id,
            "estimate":    estimate,
        }
        for round_num, round_estimates in history.items()
        for agent_id, estimate in round_estimates.items()
    ]
    model_slug = _model_slug(MODEL)
    out_stem = f"opinion_dynamics_{model_slug}_T{TEMPERATURE}_N{NUM_AGENTS}"
    out_path = os.path.join(DATA_DIR, f"{out_stem}.json")
    with open(out_path, "w") as f:
        json.dump({
            "model":           MODEL,
            "temperature":     TEMPERATURE,
            "num_agents":      NUM_AGENTS,
            "num_rounds":      NUM_ROUNDS,
            "true_value":      TRUE_VALUE,
            "rows":            rows,
            "variances":       variances,
            "consensus_round": consensus_round,
            "final_mean":      final_mean,
            "final_accuracy":  final_accuracy,
            "wall_time_s":     dt,
        }, f, indent=2)
    print(f"Wrote raw data to {out_path}")

    plot_from_history(history, temperature=TEMPERATURE, num_agents=NUM_AGENTS, model=MODEL)
