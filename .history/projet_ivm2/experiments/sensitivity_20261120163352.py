import csv
import numpy as np
from game_theory.hybrid import HybridGameGA
from experiments.run_experiments import get_problems

# (nom, valeurs, fonction valeur -> kwargs). Valeur par défaut : voir hybrid.py
PARAMS = [
    ("eta", [0.5, 1, 2, 4, 8, 16], lambda v: {"eta": v}),
    ("mu_E", [0.02, 0.05, 0.1, 0.2], lambda v: {"mu_E": v}),
    ("mu_X", [0.001, 0.005, 0.01, 0.02], lambda v: {"mu_X": v}),
    ("x_max", [0.6, 0.8, 0.95, 0.99],
     lambda v: {"x_max": v, "x_min": round(1 - v, 2)}),
]


def main(n_runs=10, n_generations=200, seed_base=100, out_csv="sensitivity.csv"):
    problems, _ = get_problems()
    rows = []
    for pname, prob in problems:
        for name, values, mk in PARAMS:
            for v in values:
                fits = [HybridGameGA(n_generations=n_generations,
                                     seed=seed_base + s, **mk(v)
                                     ).run(prob)["best_fitness"]
                        for s in range(n_runs)]
                fits = np.array(fits)
                rows.append({"problem": pname, "param": name, "value": v,
                             "mean": fits.mean(), "std": fits.std(ddof=1),
                             "min": fits.min()})
            print(pname, name, "ok", flush=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["problem", "param", "value",
                                          "mean", "std", "min"])
        w.writeheader()
        w.writerows(rows)
    print(f"Sensibilité écrite dans {out_csv}")


if __name__ == "__main__":
    main()
