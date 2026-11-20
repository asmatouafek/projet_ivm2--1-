"""Figures : convergence moyenne, évolution de x_E, sensibilité."""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"ga": "tab:blue", "ga_low": "tab:green", "fixed": "tab:gray",
          "random": "tab:purple", "hybrid": "tab:red"}
LABELS = {"ga": "AG (mu=0.05)", "ga_low": "AG (mu=0.01)",
          "fixed": "E/X fixe (x_E=0.5)", "random": "E/X aléatoire",
          "hybrid": "Hybride (réplicateur)"}
PROBLEMS = ["rastrigin_d10", "rosenbrock_d10", "knapsack_50", "berlin52"]


def load(path, col):
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row[col]:
                data.setdefault((row["problem"], row["method"]), []).append(
                    np.array([float(v) for v in row[col].split(";")]))
    return data


def plot_convergence(hist, problem, out):
    plt.figure(figsize=(7, 4.5))
    for m in ("ga", "ga_low", "fixed", "random", "hybrid"):
        h = np.array(hist[(problem, m)])
        mean, std = h.mean(0), h.std(0, ddof=1)
        lw = 2.4 if m == "hybrid" else 1.4
        plt.plot(mean, label=LABELS[m], color=COLORS[m], lw=lw)
        if m in ("ga", "hybrid"):
            plt.fill_between(range(len(mean)), mean - std, mean + std,
                             alpha=0.15, color=COLORS[m])
    plt.xlabel("Génération")
    plt.ylabel("Meilleure fitness (moyenne de 30 runs)")
    plt.title(f"Convergence moyenne — {problem}")
    if problem == "rastrigin_d10" or problem == "rosenbrock_d10":
        plt.yscale("log")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_xE(xE, problem, out, x_min=0.05, x_max=0.95):
    h = np.array(xE[(problem, "hybrid")])
    mean, std = h.mean(0), h.std(0, ddof=1)
    plt.figure(figsize=(7, 4.5))
    for r in h[:5]:
        plt.plot(r, color="tab:red", lw=0.6, alpha=0.35)
    plt.plot(mean, color="tab:red", lw=2.4, label="x_E moyen (30 runs)")
    plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.15,
                     color="tab:red", label="± 1 écart-type")
    plt.axhline(0.5, color="gray", ls="--", lw=1, label="x_E initial = 0.5")
    plt.axhline(x_min, color="k", ls=":", lw=0.8)
    plt.axhline(x_max, color="k", ls=":", lw=0.8)
    plt.ylim(0, 1)
    plt.xlabel("Génération")
    plt.ylabel("Proportion de la stratégie Exploration x_E")
    plt.title(f"Évolution de x_E (traits fins : 5 runs) — {problem}")
    plt.legend(fontsize=8, loc="center right")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_sensitivity(path="sensitivity.csv", out="sensitivity.png"):
    rows = list(csv.DictReader(open(path)))
    params = []
    for r in rows:
        if r["param"] not in params:
            params.append(r["param"])
    fig, axes = plt.subplots(len(PROBLEMS), len(params), figsize=(13, 9))
    for i, p in enumerate(PROBLEMS):
        for j, par in enumerate(params):
            ax = axes[i, j]
            sub = [r for r in rows if r["problem"] == p and r["param"] == par]
            x = [float(r["value"]) for r in sub]
            m = np.array([float(r["mean"]) for r in sub])
            s = np.array([float(r["std"]) for r in sub])
            ax.errorbar(range(len(x)), m, yerr=s, marker="o", capsize=3,
                        color="tab:red", lw=1.2)
            ax.set_xticks(range(len(x)))
            ax.set_xticklabels([f"{v:g}" for v in x], fontsize=7)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title(par)
            if j == 0:
                ax.set_ylabel(p.replace("_", " "), fontsize=8)
    plt.suptitle("Sensibilité : valeur finale moyenne ± écart-type "
                 "(10 runs par point, minimisation)")
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()


if __name__ == "__main__":
    hist = load("results.csv", "history")
    xE = load("results.csv", "history_xE")
    for p in PROBLEMS:
        plot_convergence(hist, p, f"conv_{p}.png")
        plot_xE(xE, p, f"xE_{p}.png")
    try:
        plot_sensitivity()
    except FileNotFoundError:
        print("sensitivity.csv absent : figure de sensibilité ignorée")
