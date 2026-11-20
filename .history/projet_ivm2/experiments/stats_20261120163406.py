import csv
import numpy as np
from scipy.stats import wilcoxon

REFS = ("ga", "ga_low", "fixed", "random")
ORDER = ("ga", "ga_low", "fixed", "random", "hybrid")


def load_results(path):
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            data.setdefault((row["problem"], row["method"]), []).append({
                "seed": int(row["seed"]),
                "best_fitness": float(row["best_fitness"]),
                "time": float(row["time"]),
                "conv_speed": int(row["conv_speed"]),
                "success": int(row["success"]),
            })
    for rows in data.values():
        rows.sort(key=lambda r: r["seed"])
    return data


def summary_rows(data):
    rows = []
    for p in sorted({k[0] for k in data}):
        for m in ORDER:
            r = data[(p, m)]
            v = np.array([x["best_fitness"] for x in r])
            rows.append({
                "problem": p, "method": m, "mean": v.mean(), "std": v.std(ddof=1),
                "min": v.min(),
                "success_pct": 100 * np.mean([x["success"] for x in r]),
                "conv_mean": np.mean([x["conv_speed"] for x in r]),
                "time": np.mean([x["time"] for x in r]),
            })
    return rows


def wilcoxon_rows(data, alpha=0.05):
    rows = []
    for p in sorted({k[0] for k in data}):
        hy = np.array([x["best_fitness"] for x in data[(p, "hybrid")]])
        for ref in REFS:
            rf = np.array([x["best_fitness"] for x in data[(p, ref)]])
            try:
                stat, pv = wilcoxon(hy, rf)
            except ValueError:
                stat, pv = float("nan"), 1.0
            diff = np.median(hy - rf)
            rows.append({
                "problem": p, "ref": ref, "W": stat, "p": pv,
                "median_diff": diff,
                "better": "hybride" if diff < 0 else "référence",
                "signif_005": pv < alpha,
                "signif_bonf": pv < alpha / len(REFS),
            })
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    data = load_results("results.csv")
    s = summary_rows(data)
    print(f"{'problème':15s} {'méthode':8s} {'moy':>10s} {'std':>9s} {'min':>10s}"
          f" {'succ%':>6s} {'conv':>6s} {'t(s)':>6s}")
    for r in s:
        print(f"{r['problem']:15s} {r['method']:8s} {r['mean']:10.2f} "
              f"{r['std']:9.2f} {r['min']:10.2f} {r['success_pct']:6.1f} "
              f"{r['conv_mean']:6.1f} {r['time']:6.2f}")
    write_csv(s, "summary.csv")
    print("\n=== Wilcoxon : hybride vs références (seuil Bonferroni 0.05/4) ===")
    w = wilcoxon_rows(data)
    for r in w:
        print(f"{r['problem']:15s} vs {r['ref']:7s} W={r['W']:7.1f} "
              f"p={r['p']:.2e} meilleur={r['better']:9s} "
              f"{'*' if r['signif_bonf'] else ('(.)' if r['signif_005'] else 'ns')}")
    write_csv(w, "wilcoxon.csv")
