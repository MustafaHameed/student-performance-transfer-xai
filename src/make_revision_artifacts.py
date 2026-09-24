"""Statistics, figures and LaTeX macros for the VTSE revision (results_v7).

Reads   results_v7/transfer/<policy>/*.json, results_v7/xapi/, results_v7/extras/
        and the policy-independent analyses carried over from results_v6
        (calibration, fairness, counterfactual archetypes, grade-inclusive boost).
Writes  results_v7/cliffs_delta.json
        <package>/figures/{transfer_methods_rmse,reverse_transfer,archetype_runs}.png
        <package>/_results_macros.tex

Run from the project root:  python src/make_revision_artifacts.py
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _results_dir(name: str) -> Path:
    # RnD keeps results_v* at the project root; the public repository keeps
    # them under results/.
    for cand in (ROOT / name, ROOT / 'results' / name):
        if cand.exists():
            return cand
    return ROOT / name


V6, V7 = _results_dir('results_v6'), _results_dir('results_v7')
# Where the manuscript macros and figures go: $PAPER_DIR, else the revision
# package if present (RnD layout), else ./paper_outputs (public repository).
_default_pkg = ROOT / 'submission_package' / 'vtse_revision_20260922'
PKG = Path(os.environ.get('PAPER_DIR') or (
    _default_pkg if _default_pkg.exists() else ROOT / 'paper_outputs'))
FIG = PKG / 'figures'

METHODS = ['target_only', 'pooled', 'warm_start', 'tradaboost_r2',
           'tradaboost_r2_cb', 'coral', 'gated']
LABEL = {'target_only': 'Target-only', 'pooled': 'Pooled',
         'warm_start': 'Warm-start', 'tradaboost_r2': 'TrAdaBoost.R2 (tree)',
         'tradaboost_r2_cb': 'TrAdaBoost.R2 (CatBoost)', 'coral': 'CORAL',
         'gated': 'Gated'}
SHORT = {'target_only': 'TO', 'pooled': 'Pool', 'warm_start': 'WS',
         'tradaboost_r2': 'TRtree', 'tradaboost_r2_cb': 'TRcb',
         'coral': 'Coral', 'gated': 'Gate'}

# Reference palette (dataviz skill, validated): blue / orange categorical,
# muted ink for everything else.
BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#b9b7b0', '#e4e3df'

# Analyses that do not depend on the source policy are carried over as-is.
CARRY_OVER = ['calibration.json', 'fairness.json', 'shap_bootstrap.json',
              'counterfactual/archetypes.json', 'counterfactual/cf_deltas.csv',
              'g1g2_boost/g1g2_boost_results.json']


def load(path: Path):
    return json.load(open(path)) if path.exists() else None


def cliffs(a, b) -> float:
    """Positive when a has the lower RMSE on more paired folds."""
    d = np.asarray(b) - np.asarray(a)
    return float(((d > 0).sum() - (d < 0).sum()) / len(d))


def cliffs_matrix(fold_rmse: dict) -> dict:
    ms = [m for m in METHODS if m in fold_rmse]
    mat = [[0.0 if i == j else cliffs(fold_rmse[a], fold_rmse[b])
            for j, b in enumerate(ms)] for i, a in enumerate(ms)]
    return {'methods': ms, 'matrix': mat}


def _style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_rmse(d: dict, out: Path):
    s = d['summary']
    ms = sorted([m for m in METHODS if m in s], key=lambda m: s[m]['rmse']['mean'])
    mu = [s[m]['rmse']['mean'] for m in ms]
    ci = [s[m]['rmse']['ci95'] for m in ms]
    # Dot-and-whisker rather than bars: the axis does not start at zero.
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, (m, v, c) in enumerate(zip(ms, mu, ci)):
        col = BLUE if m == 'gated' else INK2
        ax.errorbar(v, i, xerr=c, fmt='o', color=col, ecolor=col,
                    elinewidth=1.4, capsize=3, markersize=7,
                    markeredgecolor='white', markeredgewidth=1.5)
    to = s['target_only']['rmse']['mean']
    ax.axvline(to, color=INK2, linestyle='--', linewidth=1)
    ax.set_yticks(range(len(ms)))
    ax.set_yticklabels([LABEL[m] for m in ms])
    ax.invert_yaxis()
    lo = min(m - c for m, c in zip(mu, ci))
    hi = max(m + c for m, c in zip(mu, ci))
    ax.set_xlim(lo - 0.15, hi + 0.25)
    for i, (m, v, c) in enumerate(zip(ms, mu, ci)):
        ax.text(v + c + 0.02, i, f'{v:.3f}', va='center', fontsize=8.5, color=INK)
    ax.set_xlabel('RMSE on the Mathematics target (lower is better; 95% CI, 50 folds)',
                  fontsize=9, color=INK2)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_reverse(fwd: dict, rev: dict, out: Path):
    ms = [m for m in METHODS if m != 'target_only' and m in fwd['summary']
          and m in rev['summary']]

    def gain(d, m):
        s = d['summary']
        return s[m]['rmse']['mean'] - s['target_only']['rmse']['mean']
    y = np.arange(len(ms))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    h = 0.38
    ax.barh(y - h / 2, [gain(fwd, m) for m in ms], height=h, color=BLUE,
            edgecolor='white', linewidth=2,
            label='Por$\\rightarrow$Math (target $n=395$)')
    ax.barh(y + h / 2, [gain(rev, m) for m in ms], height=h, color=ORANGE,
            edgecolor='white', linewidth=2,
            label='Math$\\rightarrow$Por (target $n=649$)')
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[m] for m in ms])
    ax.invert_yaxis()
    ax.set_xlabel('RMSE change vs. target-only (negative = transfer helps)',
                  fontsize=9, color=INK2)
    ax.legend(fontsize=8, frameon=False, loc='lower right')
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


ARCH_COLOR = {'Attendance-focused': BLUE, 'Lifestyle-moderation': ORANGE,
              'Study-intensity': '#1baf7a'}


def fig_archetype_runs(runs: list[tuple[str, dict]], out: Path):
    """Stacked bars of archetype sizes per DiCE run; counts labelled on
    every segment (aqua is below 3:1 contrast, so labels carry identity)."""
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    seen = set()
    for i, (label, r) in enumerate(runs):
        left = 0
        segs = sorted(r['archetypes'], key=lambda a: -a['n_students'])
        segs.append({'name': 'No counterfactual',
                     'n_students': r['n_at_risk'] - r['n_with_cf']})
        for a in segs:
            n = a['n_students']
            col = ARCH_COLOR.get(a['name'], MUTED)
            lab = a['name'] if a['name'] not in seen else None
            seen.add(a['name'])
            ax.barh(i, n, left=left, color=col, edgecolor='white',
                    linewidth=2, height=0.66, label=lab)
            ax.text(left + n / 2, i, str(n), ha='center', va='center',
                    fontsize=8, color='white' if col in (BLUE, ORANGE) else INK)
            left += n
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels([r[0] for r in runs])
    ax.invert_yaxis()
    ax.set_xlabel('At-risk Mathematics students (of 80 sampled)',
                  fontsize=9, color=INK2)
    ax.legend(fontsize=8, frameon=False, ncol=2, loc='upper center',
              bbox_to_anchor=(0.5, -0.2))
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def f3(x):
    return f'{x:.3f}'


def sci(p):
    if p is None:
        return 'n/a'
    if p >= 0.001:
        return f'{p:.3f}'
    m, e = f'{p:.2e}'.split('e')
    return f'{m}\\times10^{{{int(e)}}}'


def main():
    for rel in CARRY_OVER:
        dst = V7 / rel
        if not (V6 / rel).exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(V6 / rel, dst)

    runs = {}
    for pol in ('fold_exclusive', 'disjoint', 'full'):
        for kind, name in (('fwd', 'transfer_cv.json'), ('rev', 'transfer_cv_reverse.json'),
                           ('g12', 'transfer_cv_with_g1g2.json')):
            d = load(V7 / 'transfer' / pol / name)
            if d:
                runs[(pol, kind)] = d
    print('runs found:', sorted(runs))

    M: dict[str, str] = {}

    def put(name, val):
        assert name.isalpha(), name
        M[name] = val

    fwd = runs.get(('fold_exclusive', 'fwd'))
    if fwd:
        s = fwd['summary']
        for m in METHODS:
            k = SHORT[m]
            for met, tag in (('rmse', 'Rmse'), ('mae', 'Mae'), ('r2', 'Rsq')):
                put(f'f{k}{tag}', f3(s[m][met]['mean']))
                put(f'f{k}{tag}CI', f3(s[m][met]['ci95']))
        g = fwd['gate_decisions']
        put('fGateChose', str(sum(x['chose_transfer'] for x in g)))
        put('fGateFallback', str(sum(not x['chose_transfer'] for x in g)))
        put('fNFolds', str(len(g)))
        ss = fwd['source_size']
        put('fSrcMin', str(ss['min']))
        put('fSrcMax', str(ss['max']))
        to, gt = s['target_only']['rmse']['mean'], s['gated']['rmse']['mean']
        put('fGateDelta', f3(gt - to))
        put('fGateRel', f'{100 * (to - gt) / to:.1f}')
        for m in ('warm_start', 'coral', 'pooled'):
            put(f'f{SHORT[m]}Rel',
                f"{100 * (to - s[m]['rmse']['mean']) / to:.1f}")
        st = fwd['statistical_tests']
        put('fFriedChi', f"{st['friedman']['chi2']:.2f}")
        put('fFriedP', sci(st['friedman']['p']))
        wil = st['wilcoxon_bonferroni']

        def wp(a, b):
            r = wil.get(f'{a}__vs__{b}') or wil.get(f'{b}__vs__{a}')
            return None if r is None else r.get('p_bonferroni')
        for a, b in [('target_only', 'gated'), ('target_only', 'warm_start'),
                     ('pooled', 'warm_start'), ('pooled', 'gated'),
                     ('warm_start', 'coral'), ('pooled', 'coral'),
                     ('target_only', 'coral'), ('target_only', 'pooled'),
                     ('target_only', 'tradaboost_r2'),
                     ('target_only', 'tradaboost_r2_cb'),
                     ('tradaboost_r2', 'tradaboost_r2_cb')]:
            put(f'fP{SHORT[a]}{SHORT[b]}', sci(wp(a, b)))
        cm = cliffs_matrix(fwd['_fold_rmse'])
        json.dump({**cm, 'interpretation': 'entry (i,j) > 0: method i has lower '
                   'RMSE on more paired folds (Romano et al. 2006 thresholds)'},
                  open(V7 / 'cliffs_delta.json', 'w'), indent=2)
        ix = {m: i for i, m in enumerate(cm['methods'])}
        for a in METHODS:
            for b in METHODS:
                if a != b:
                    v = cm['matrix'][ix[a]][ix[b]]
                    put(f'cd{SHORT[a]}{SHORT[b]}', f'{v:+.2f}')
        # table rows
        rows = []
        for m in METHODS:
            k = SHORT[m]
            cells = ' & '.join(f'${M[f"f{k}{x}"]} \\pm {M[f"f{k}{x}CI"]}$'
                               for x in ('Rmse', 'Mae', 'Rsq'))
            name = f'\\textbf{{{LABEL[m]}}}' if m == 'gated' else LABEL[m]
            rows.append(f'{name} & {cells} \\\\')
        put('TransferRows', '\n'.join(rows))
        FIG.mkdir(parents=True, exist_ok=True)
        fig_rmse(fwd, FIG / 'transfer_methods_rmse.png')

    for pol, pre in (('disjoint', 'd'), ('full', 'u')):
        d = runs.get((pol, 'fwd'))
        if d:
            s = d['summary']
            for m in METHODS:
                put(f'{pre}{SHORT[m]}Rmse', f3(s[m]['rmse']['mean']))
            put(f'{pre}GateChose', str(sum(x['chose_transfer'] for x in d['gate_decisions'])))
            put(f'{pre}SrcN', str(d['config']['source_rows']))
            to, gt = s['target_only']['rmse']['mean'], s['gated']['rmse']['mean']
            put(f'{pre}GateRel', f'{100 * (to - gt) / to:.1f}')
            wil = d['statistical_tests']['wilcoxon_bonferroni']
            for a, b in [('target_only', 'gated'), ('target_only', 'coral'),
                         ('target_only', 'pooled'), ('target_only', 'warm_start'),
                         ('warm_start', 'coral'), ('coral', 'gated'),
                         ('pooled', 'coral')]:
                r = wil.get(f'{a}__vs__{b}') or wil.get(f'{b}__vs__{a}')
                put(f'{pre}P{SHORT[a]}{SHORT[b]}',
                    sci(r.get('p_bonferroni') if r else None))
            for m in ('coral', 'pooled', 'warm_start'):
                put(f'{pre}{SHORT[m]}Rel',
                    f"{100 * (to - s[m]['rmse']['mean']) / to:.1f}")

    rev = runs.get(('fold_exclusive', 'rev'))
    if rev:
        s = rev['summary']
        for m in METHODS:
            put(f'r{SHORT[m]}Rmse', f3(s[m]['rmse']['mean']))
        put('rGateChose', str(sum(x['chose_transfer'] for x in rev['gate_decisions'])))
        to, gt = s['target_only']['rmse']['mean'], s['gated']['rmse']['mean']
        put('rGateDelta', f3(gt - to))
        put('rGateRel', f'{100 * (to - gt) / to:.1f}')
        put('rSrcMin', str(rev['source_size']['min']))
        put('rSrcMax', str(rev['source_size']['max']))
        wil = rev['statistical_tests']['wilcoxon_bonferroni']
        for a, b in [('target_only', 'gated'), ('target_only', 'coral'),
                     ('target_only', 'warm_start'), ('target_only', 'pooled')]:
            r = wil.get(f'{a}__vs__{b}') or wil.get(f'{b}__vs__{a}')
            put(f'rP{SHORT[a]}{SHORT[b]}',
                sci(r.get('p_bonferroni') if r else None))
        if fwd:
            fig_reverse(fwd, rev, FIG / 'reverse_transfer.png')

    for pol, pre in (('fold_exclusive', 'g'), ('disjoint', 'gd'), ('full', 'gu')):
        d = runs.get((pol, 'g12'))
        if d:
            s = d['summary']
            for m in ('target_only', 'warm_start', 'gated'):
                put(f'{pre}{SHORT[m]}Rmse', f3(s[m]['rmse']['mean']))
                put(f'{pre}{SHORT[m]}Mae', f3(s[m]['mae']['mean']))
            put(f'{pre}GateChose', str(sum(x['chose_transfer'] for x in d['gate_decisions'])))
            wil = d['statistical_tests']['wilcoxon_bonferroni']
            r = wil.get('target_only__vs__gated')
            put(f'{pre}PTOGate', sci(r.get('p_bonferroni') if r else None))

    x = load(V7 / 'xapi' / 'external_validation.json')
    if x:
        for k, n in (('auc', 'Auc'), ('accuracy', 'Acc'), ('f1_macro', 'Fone')):
            put(f'x{n}', f3(x[k]))
            put(f'x{n}Lo', f3(x[f'{k}_ci95_lo']))
            put(f'x{n}Hi', f3(x[f'{k}_ci95_hi']))

    e = load(V7 / 'extras' / 'shap_stability_exact.json')
    if e:
        P = e['spearman_p_matrix']
        put('pPorMath', f3(P[0][1]))
        put('pPorX', f3(P[0][2]))
        put('pMathX', f3(P[1][2]))
        put('critRho', f3(e['critical_abs_rho_alpha05']))
        put('minPtwo', sci(e['min_p_two_sided']))
        put('minPone', sci(e['min_p_one_sided']))
        B = e['bootstrap']
        for pair, n in (('Por_vs_Math', 'PorMath'), ('Por_vs_xAPI', 'PorX'),
                        ('Math_vs_xAPI', 'MathX')):
            put(f'ci{n}Lo', f3(B[pair]['bootstrap_ci95_lo']))
            put(f'ci{n}Hi', f3(B[pair]['bootstrap_ci95_hi']))
    p = load(V7 / 'extras' / 'shap_por_only_sensitivity.json')
    if p:
        put('porOnlyN', str(p['n_portuguese_only']))
        put('porOnlyRho', f3(p['rho_por_only_vs_math']))
        put('porOnlyP', f3(p['p_two_sided_exact']))
    c = load(V7 / 'extras' / 'cf_diagnostics.json')
    if c:
        cp, up = c['covered_profile'], c['uncovered_profile']
        put('cfPredRisk', str(c['n_predicted_at_risk_in_sample']))
        put('cfUncFailMean', f'{up["failures_mean"]:.2f}')
        put('cfCovFailMean', f'{cp["failures_mean"]:.2f}')
        put('cfUncFailShare', f'{100 * up["share_with_prior_failure"]:.0f}')
        put('cfCovFailShare', f'{100 * cp["share_with_prior_failure"]:.0f}')
        put('cfUncAbsMed', f'{up["absences_median"]:.1f}')
        put('cfCovAbsMed', f'{cp["absences_median"]:.1f}')
        put('cfUncAbsMean', f'{up["absences_mean"]:.2f}')
        put('cfCovAbsMean', f'{cp["absences_mean"]:.2f}')
        put('cfUncRisk', f'{up["predicted_risk_mean"]:.2f}')
        put('cfCovRisk', f'{cp["predicted_risk_mean"]:.2f}')
        k3 = c['k3_clusters']
        put('kThreeSizes', ', '.join(str(k['n']) for k in k3))
        w = c['worked_example']
        put('wexProb', f'{w["predicted_risk_probability"]:.2f}')

    eda = load(V7 / 'extras' / 'eda_stats.json')
    if eda:
        tag = {'Portuguese': 'Por', 'Mathematics': 'Math', 'xAPI-Edu': 'X'}
        cname = {'failure_risk': 'Fail', 'engagement': 'Eng',
                 'high_absence': 'Abs', 'higher_intent': 'Higher',
                 'par_education': 'Par'}
        for dom, t in tag.items():
            d = eda[dom]
            put(f'eda{t}Risk', f"{d['at_risk_pct']:.1f}")
            for cc, n in cname.items():
                put(f'eda{t}{n}', f"{d['cohens_d'][cc]:.2f}")
            big = sum(abs(v) >= 0.5 for v in d['cohens_d'].values())
            put(f'eda{t}Big', ['no', 'one', 'two', 'three', 'four', 'five',
                               'six', 'seven', 'eight'][big])

    fair = load(V7 / 'fairness.json')
    if fair:
        pg = fair['Portuguese']['by_gender']
        put('fairGapM', f3(pg['M']['at_risk_rate'] - pg['M']['predicted_at_risk_rate']))
        put('fairGapF', f3(pg['F']['at_risk_rate'] - pg['F']['predicted_at_risk_rate']))
        rows = []
        grp = {'M': 'Male', 'F': 'Female', 'low': 'low', 'mid': 'mid', 'high': 'high'}
        for dom in ('Portuguese', 'Mathematics', 'xAPI-Edu'):
            d = fair[dom]
            pe = 'Par.\\ edu.' if dom != 'xAPI-Edu' else 'Parent level'
            entries = [(grp[g], d['by_gender'][g]) for g in ('M', 'F')] + \
                      [(f'{pe} {grp[t]}', d['by_par_education_tertile'][t])
                       for t in ('low', 'mid', 'high')]
            for i, (lab, v) in enumerate(entries):
                head = f'\\multirow{{5}}{{*}}{{{dom}}}' if i == 0 else ''
                rows.append(f"{head} & {lab} & {v['n']} & {v['at_risk_rate']:.3f} & "
                            f"{v['predicted_at_risk_rate']:.3f} & {v['recall']:.3f} & "
                            f"{v['precision']:.3f} & {v['fpr']:.3f} \\\\")
            if dom != 'xAPI-Edu':
                rows.append('\\midrule')
        put('FairRows', '\n'.join(rows))

    sw = load(V7 / 'extras' / 'cf_seed_sweep.json')
    arch_v6 = load(V7 / 'counterfactual' / 'archetypes.json')
    if sw and arch_v6:
        R = sw['runs']
        nwc = [r['n_with_cf'] for r in R.values()]
        att = [a['n_students'] for r in R.values() for a in r['archetypes']
               if a['name'] == 'Attendance-focused']
        sec = [a['n_students'] for r in R.values() for a in r['archetypes']
               if a['name'] != 'Attendance-focused']
        k2 = [r['silhouette']['2'] for r in R.values()]
        ab = [dict(a['dominant_features']).get('absences') for r in R.values()
              for a in r['archetypes'] if a['name'] == 'Attendance-focused']
        go = [dict(a['dominant_features']).get('goout') for r in R.values()
              for a in r['archetypes'] if a['name'] == 'Attendance-focused']
        put('swCovMinN', str(min(nwc)))
        put('swCovMaxN', str(max(nwc)))
        put('swCovMin', f3(min(nwc) / 80))
        put('swCovMax', f3(max(nwc) / 80))
        put('swAttMin', str(min(att)))
        put('swAttMax', str(max(att)))
        put('swSecMin', str(min(sec)))
        put('swSecMax', str(max(sec)))
        put('swSilTwoMin', f3(min(k2)))
        put('swSilTwoMax', f3(max(k2)))
        put('swAbsMin', f'{min(abs(x) for x in ab):.1f}')
        put('swAbsMax', f'{max(abs(x) for x in ab):.1f}')
        put('swGoMin', f'{min(abs(x) for x in go):.1f}')
        put('swGoMax', f'{max(abs(x) for x in go):.1f}')
        put('swSpMin', f'{min(r["mean_sparsity"] for r in R.values()):.2f}')
        put('swSpMax', f'{max(r["mean_sparsity"] for r in R.values()):.2f}')
        put('swBestKs', ', '.join(sorted({str(r['best_k']) for r in R.values()})))
        ag = [p['name_agreement'] for p in sw['pairwise']]
        ar = [p['adjusted_rand'] for p in sw['pairwise']]
        put('swAgreeMin', f'{min(ag):.2f}')
        put('swAgreeMax', f'{max(ag):.2f}')
        put('swAgreeMean', f'{np.mean(ag):.2f}')
        put('swAriMin', f'{min(ar):.2f}')
        put('swAriMax', f'{max(ar):.2f}')
        if c:
            ps = c['per_seed_uncovered']
            put('swUncMin', str(min(v['n_uncovered'] for v in ps.values())))
            put('swUncMax', str(max(v['n_uncovered'] for v in ps.values())))

        def rec(a):
            return '; '.join(f'{f} {v:+.1f}' for f, v in a['dominant_features'])
        rows = []
        mv6 = arch_v6['metrics']
        a6 = sorted(arch_v6['archetypes'], key=lambda a: -a['n_students'])
        runs_fig = [('Original (unseeded)', {**mv6, 'archetypes': a6})]
        sil6 = arch_v6['silhouette_scores']
        rows.append(('Original (unseeded)', mv6['n_with_cf'], mv6['mean_sparsity'],
                     a6, sil6['2'], max(sil6, key=sil6.get)))
        for sd, r in R.items():
            aa = sorted(r['archetypes'], key=lambda a: -a['n_students'])
            runs_fig.append((f'Seed {sd}', r))
            rows.append((f'Seed {sd}', r['n_with_cf'], r['mean_sparsity'], aa,
                         r['silhouette']['2'], str(r['best_k'])))
        tex = []
        for lab, nw, sp, aa, s2, bk in rows:
            arch = ' / '.join(f"{a['name']} {a['n_students']}" for a in aa)
            tex.append(f'{lab} & {nw} & {sp:.2f} & {arch} & {s2:.3f} & {bk} \\\\')
        put('CfSeedRows', '\n'.join(tex))
        fig_archetype_runs(runs_fig, FIG / 'archetype_runs.png')

    lines = ['% Auto-generated by src/make_revision_artifacts.py from results_v7.',
             '% Do not edit by hand.']
    for k, v in M.items():
        lines.append(f'\\newcommand{{\\{k}}}{{{v}}}')
    PKG.mkdir(parents=True, exist_ok=True)
    (PKG / '_results_macros.tex').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print(f'wrote {len(M)} macros to {PKG / "_results_macros.tex"}')


if __name__ == '__main__':
    main()
