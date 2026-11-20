"""Expériences : AG de référence vs AG + jeu (réplicateur), 30 runs par config.

Méthodes :
  ga      : AG classique, mu = 0.05, p_c = 0.8 (paramètres « standard »)
  ga_low  : AG classique, mu = 0.01, p_c = 0.8 (AG à faible mutation)
  hybrid  : AG + jeu Exploration/eXploitation, dynamique du réplicateur
  fixed   : mêmes opérateurs E/X, proportion fixe x_E = 0.5 (sans jeu)
  random  : mêmes opérateurs E/X, x_E tiré au hasard (sans apprentissage)
"""
import csv
import os
import time
from metaheuristic.ga import GeneticAlgorithm
from game_theory.hybrid import HybridGameGA
from problems.continuous import ContinuousProblem
from problems.knapsack import KnapsackProblem
from problems.tsp import TSPProblem, parse_tsplib

METHODS = ("ga", "ga_low", "hybrid", "fixed", "random")
BERLIN52_OPT = 7542.0


def get_problems(tsplib_dir="tsplib"):
    knap = KnapsackProblem(n_items=50, seed=42)
    problems = [
        ("rastrigin_d10", ContinuousProblem("rastrigin", dim=10)),
        ("rosenbrock_d10", ContinuousProblem("rosenbrock", dim=10)),
        ("knapsack_50", knap),
        ("berlin52", TSPProblem(parse_tsplib(
            os.path.join(tsplib_dir, "berlin52.tsp")))),
    ]
    return problems, knap


def success_thresholds(knap):
    """Seuil de succès (minimisation) par problème."""
    return {
        "rastrigin_d10": 5.0,                      # optimum 0
        "rosenbrock_d10": 20.0,                    # optimum 0
        "knapsack_50": 0.99 * knap.optimum(),      # à 1 % de l'optimum (DP)
        "berlin52": 1.40 * BERLIN52_OPT,           # à 40 % de l'optimum connu
    }


def convergence_speed(history, threshold, censor):
    """Première génération où le seuil est atteint ; si jamais atteint, la
    valeur est CENSURÉE à n_generations (convention documentée)."""
    for t, v in enumerate(history):
        if v <= threshold:
            return t
    return censor


def build(method, seed, n_generations, pop_size):
    kw = dict(pop_size=pop_size, n_generations=n_generations, seed=seed)
    if method == "ga":
        return GeneticAlgorithm(**kw)
    if method == "ga_low":
        return GeneticAlgorithm(mu=0.01, **kw)
    mode = {"hybrid": "replicator", "fixed": "fixed", "random": "random"}[method]
    return HybridGameGA(mode=mode, **kw)


def run_one(method, problem, seed, n_generations=200, pop_size=100):
    algo = build(method, seed, n_generations, pop_size)
    t0 = time.perf_counter()
    res = algo.run(problem)
    res["time"] = time.perf_counter() - t0
    return res


def main(n_runs=30, n_generations=200, pop_size=100,
         tsplib_dir="tsplib", out_csv="results.csv"):
    problems, knap = get_problems(tsplib_dir)
    thr = success_thresholds(knap)
    fieldnames = ["problem", "method", "seed", "best_fitness", "time",
                  "conv_speed", "success", "history", "history_xE"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pname, prob in problems:
            for method in METHODS:
                for seed in range(n_runs):
                    res = run_one(method, prob, seed, n_generations, pop_size)
                    writer.writerow({
                        "problem": pname, "method": method, "seed": seed,
                        "best_fitness": res["best_fitness"],
                        "time": res["time"],
                        "conv_speed": convergence_speed(
                            res["history"], thr[pname], n_generations),
                        "success": int(res["best_fitness"] <= thr[pname]),
                        "history": ";".join(f"{v:.6f}" for v in res["history"]),
                        "history_xE": ";".join(
                            f"{v:.4f}" for v in res.get("history_xE", [])),
                    })
                f.flush()
            print(f"{pname} terminé", flush=True)
    print(f"Résultats écrits dans {out_csv}")


if __name__ == "__main__":
    main()
