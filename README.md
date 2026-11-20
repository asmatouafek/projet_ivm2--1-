# Projet IVM2 — Théorie des jeux dans une métaheuristique

AG hybride dont la proportion d'exploration x_E est pilotée par un **jeu
évolutionnaire à deux stratégies (Exploration / eXploitation)** et la
**dynamique du réplicateur**.

## Installation (Python >= 3.8)
    pip install -r requirements.txt

## Structure
- `metaheuristic/` : AG de base (`ga.py`) et opérateurs (`operators.py`)
- `game_theory/`   : `game.py` (gains), `replicator.py` (dynamique),
                     `controller.py` (réplicateur / aléatoire / fixe),
                     `hybrid.py` (AG + jeu)
- `problems/`      : Rastrigin, Rosenbrock, sac à dos (optimum exact par DP),
                     TSP + parseur TSPLIB (`tsplib/berlin52.tsp`)
- `experiments/`   : `run_experiments.py`, `stats.py`, `sensitivity.py`, `plots.py`

## Exécution (depuis la racine du projet)
    python -m experiments.run_experiments   # -> results.csv (5 méthodes x 4 problèmes x 30 runs)
    python -m experiments.stats             # -> summary.csv, wilcoxon.csv
    python -m experiments.sensitivity       # -> sensitivity.csv
    python -m experiments.plots             # -> conv_*.png, xE_*.png, sensitivity.png

## Méthodes comparées
`ga` (mu=0.05), `ga_low` (mu=0.01), `fixed` (E/X, x_E=0.5 constant),
`random` (E/X, x_E aléatoire), `hybrid` (E/X, x_E par réplicateur).
`fixed` et `random` sont des ablations : elles isolent l'apport du jeu.

## Reproductibilité
Seeds 0..29 pour l'évaluation finale, 100..109 pour la sensibilité.
À seed égal, toutes les méthodes partent de la même population initiale
(comparaisons appariées pour le test de Wilcoxon).
