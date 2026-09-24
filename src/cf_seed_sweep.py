"""DiCE seed sweep for the counterfactual archetypes (revision analysis).

The 80 sampled at-risk Mathematics students are fixed (seeded sampling); only
the DiCE random-search seed changes. For each seed we record coverage,
timeouts, archetype names/sizes/recipes and silhouette scores, and we measure
how often two seeds put the same student in the same named archetype.

Writes results_v7/extras/cf_seed_sweep.json.

Run from the project root:  python src/cf_seed_sweep.py [--max-at-risk N]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import pandas as pd
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from counterfactuals import generate_archetypes  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# RnD keeps results_v* at the project root; the public repository under results/.
RESULTS = next((d for d in (ROOT, os.path.join(ROOT, 'results'))
                if os.path.isdir(os.path.join(d, 'results_v6'))), ROOT)
SEEDS = [0, 42, 123, 2024, 2025]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-at-risk', type=int, default=80)
    ap.add_argument('--out', default=os.path.join(RESULTS, 'results_v7', 'extras',
                                                  'cf_seed_sweep.json'))
    args = ap.parse_args()

    df_mat = pd.read_csv(os.path.join(ROOT, 'data', 'student-mat.csv'), sep=';')
    runs, labels = {}, {}
    for seed in SEEDS:
        t0 = time.time()
        metrics, archetypes, deltas, sil = generate_archetypes(
            df_mat, n_archetypes=2, cf_per_student=1,
            max_at_risk=args.max_at_risk, method='random',
            enforce_monotonic=True, dice_seed=seed)
        names = {a.cluster_id: a.name for a in archetypes}
        seed_dir = os.path.join(os.path.dirname(args.out), 'cf_seeds', str(seed))
        os.makedirs(seed_dir, exist_ok=True)
        if not deltas.empty:
            deltas.to_csv(os.path.join(seed_dir, 'cf_deltas.csv'), index=False)
        with open(os.path.join(seed_dir, 'archetypes.json'), 'w') as fh:
            json.dump({
                'metrics': dict(n_at_risk=metrics.n_at_risk,
                                n_with_cf=metrics.n_with_cf,
                                coverage=metrics.coverage,
                                validity=metrics.validity,
                                mean_sparsity=metrics.mean_sparsity,
                                mean_proximity=metrics.mean_proximity,
                                n_timeouts=metrics.n_timeouts),
                'archetypes': [dict(cluster_id=a.cluster_id, name=a.name,
                                    n_students=a.n_students,
                                    dominant_features=a.dominant_features,
                                    recipe=(a.example_cf or {}).get('recipe', ''))
                               for a in archetypes],
                'silhouette_scores': {str(k): v for k, v in sil.items()},
                'dice_seed': seed,
            }, fh, indent=2)
        labels[seed] = ({int(i): names[int(c)] for i, c in
                         zip(deltas['_orig_idx'], deltas['_archetype'])}
                        if not deltas.empty else {})
        runs[str(seed)] = {
            'seconds': round(time.time() - t0, 1),
            'n_at_risk': metrics.n_at_risk, 'n_with_cf': metrics.n_with_cf,
            'coverage': metrics.coverage, 'n_timeouts': metrics.n_timeouts,
            'mean_sparsity': metrics.mean_sparsity,
            'mean_proximity': metrics.mean_proximity,
            'archetypes': [dict(name=a.name, n_students=a.n_students,
                                dominant_features=a.dominant_features)
                           for a in archetypes],
            'silhouette': {str(k): v for k, v in sil.items()},
            'best_k': int(max(sil, key=sil.get)) if sil else None,
        }
        print(f"[seed {seed}] cov={metrics.coverage:.3f} "
              f"timeouts={metrics.n_timeouts} "
              f"archetypes={[(a.name, a.n_students) for a in archetypes]} "
              f"best_k={runs[str(seed)]['best_k']} "
              f"({runs[str(seed)]['seconds']}s)", flush=True)

    pairs = []
    for a, b in itertools.combinations(SEEDS, 2):
        common = sorted(set(labels[a]) & set(labels[b]))
        if not common:
            continue
        la = [labels[a][i] for i in common]
        lb = [labels[b][i] for i in common]
        pairs.append({'seeds': [a, b], 'n_common': len(common),
                      'name_agreement': sum(x == y for x, y in zip(la, lb)) / len(common),
                      'adjusted_rand': adjusted_rand_score(la, lb)})
    out = {'seeds': SEEDS, 'runs': runs, 'pairwise': pairs}
    if pairs:
        ag = [p['name_agreement'] for p in pairs]
        out['name_agreement_min'] = min(ag)
        out['name_agreement_mean'] = sum(ag) / len(ag)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
