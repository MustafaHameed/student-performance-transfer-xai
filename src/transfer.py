"""Transfer-learning strategies for cross-course student risk prediction.

Three strategies + one fallback gate:

    A) warm_start_catboost    -- CatBoost init_model fine-tuning
    B) tradaboost_r2          -- Pardoe & Stone (2010) TrAdaBoost.R2
    C) coral_align            -- Sun et al. (2016) CORAL feature alignment

    negative_transfer_gate    -- per-fold fallback to target-only when
                                 transfer hurts on inner validation.

All strategies expect numeric ndarray inputs (already feature-engineered
and encoded) and return scikit-learn-style fitted estimators with a
.predict(X) method. Source = Portuguese, target = Mathematics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor


SEED = 42


# ──────────────────────────────────────────────────────────
# A. CatBoost warm-start fine-tuning
# ──────────────────────────────────────────────────────────

def warm_start_catboost(X_source: np.ndarray, y_source: np.ndarray,
                        X_target: np.ndarray, y_target: np.ndarray,
                        source_params: dict | None = None,
                        finetune_iterations: int = 200,
                        finetune_lr: float = 0.03) -> CatBoostRegressor:
    """Train a source CatBoost model, then continue training on target with
    a smaller learning rate and fewer trees (init_model fine-tuning)."""
    src_params = dict(
        iterations=400, depth=6, learning_rate=0.05,
        l2_leaf_reg=3.0, random_seed=SEED, verbose=False,
        allow_writing_files=False,
    )
    if source_params:
        src_params.update(source_params)

    src = CatBoostRegressor(**src_params)
    src.fit(X_source, y_source)

    tgt = CatBoostRegressor(
        iterations=finetune_iterations,
        depth=src_params['depth'],
        learning_rate=finetune_lr,
        l2_leaf_reg=src_params['l2_leaf_reg'],
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,
    )
    tgt.fit(X_target, y_target, init_model=src)
    return tgt


# ──────────────────────────────────────────────────────────
# B. TrAdaBoost.R2 (Pardoe & Stone, 2010)
# ──────────────────────────────────────────────────────────

class TrAdaBoostR2:
    """TrAdaBoost.R2 for regression transfer.

    Combines source + target into one weighted training set. At each round:
    train a weak learner, compute per-sample errors, then update weights:
      target samples -> increase weight if wrong (boost-like)
      source samples -> decrease weight if wrong (suppress mismatched source)

    Final prediction is the weighted median of the second-half learners
    (per Pardoe & Stone). Weak learner: shallow DecisionTreeRegressor by
    default; base_learner='catboost' swaps in a small CatBoost regressor so
    the strategy shares the base-learner family of the other two.
    """

    def __init__(self, n_estimators: int = 30, max_depth: int = 6,
                 random_state: int = SEED, base_learner: str = 'tree',
                 catboost_iterations: int = 100):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.base_learner = base_learner
        self.catboost_iterations = catboost_iterations
        self.estimators_: list = []
        self.estimator_weights_: list[float] = []
        self.n_target_: int = 0

    def fit(self, X_source: np.ndarray, y_source: np.ndarray,
            X_target: np.ndarray, y_target: np.ndarray):
        n_s = X_source.shape[0]
        n_t = X_target.shape[0]
        n = n_s + n_t
        self.n_target_ = n_t

        X = np.vstack([X_source, X_target])
        y = np.concatenate([y_source, y_target])

        weights = np.ones(n) / n
        beta_source = 1.0 / (1.0 + np.sqrt(2.0 * np.log(n_s) / self.n_estimators))

        rng = np.random.default_rng(self.random_state)

        for t in range(self.n_estimators):
            if self.base_learner == 'catboost':
                est = CatBoostRegressor(iterations=self.catboost_iterations,
                                        depth=self.max_depth, learning_rate=0.1,
                                        l2_leaf_reg=3.0,
                                        random_seed=self.random_state + t,
                                        verbose=False, allow_writing_files=False)
            else:
                est = DecisionTreeRegressor(max_depth=self.max_depth,
                                            random_state=self.random_state + t)
            sample_idx = rng.choice(n, size=n, replace=True, p=weights)
            est.fit(X[sample_idx], y[sample_idx])
            preds = est.predict(X)

            errors = np.abs(preds - y)
            err_max = errors.max() if errors.max() > 0 else 1.0
            adj = errors / err_max

            target_weight_sum = weights[n_s:].sum()
            target_err = float(np.sum(weights[n_s:] * adj[n_s:]) /
                               max(target_weight_sum, 1e-12))
            target_err = min(max(target_err, 1e-6), 0.5 - 1e-6)

            beta_t = target_err / (1.0 - target_err)

            new_weights = weights.copy()
            new_weights[:n_s] *= beta_source ** adj[:n_s]
            new_weights[n_s:] *= beta_t ** (-adj[n_s:])
            new_weights /= new_weights.sum()
            weights = new_weights

            self.estimators_.append(est)
            self.estimator_weights_.append(np.log(1.0 / beta_t))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Weighted-median over the second half of learners (Pardoe & Stone)
        start = self.n_estimators // 2
        ests = self.estimators_[start:]
        wts = np.array(self.estimator_weights_[start:])
        if wts.sum() <= 0:
            preds = np.column_stack([e.predict(X) for e in ests])
            return preds.mean(axis=1)

        all_preds = np.column_stack([e.predict(X) for e in ests])
        out = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            order = np.argsort(all_preds[i])
            sorted_w = wts[order]
            cumw = np.cumsum(sorted_w)
            half = cumw[-1] / 2.0
            j = int(np.searchsorted(cumw, half))
            j = min(j, len(order) - 1)
            out[i] = all_preds[i][order[j]]
        return out


def tradaboost_r2(X_source: np.ndarray, y_source: np.ndarray,
                  X_target: np.ndarray, y_target: np.ndarray,
                  n_estimators: int = 30, max_depth: int = 6,
                  base_learner: str = 'tree') -> TrAdaBoostR2:
    model = TrAdaBoostR2(n_estimators=n_estimators, max_depth=max_depth,
                         base_learner=base_learner)
    model.fit(X_source, y_source, X_target, y_target)
    return model


# ──────────────────────────────────────────────────────────
# C. CORAL alignment + standard model
# ──────────────────────────────────────────────────────────

class CoralAdaptedModel:
    """Sun et al. CORAL: align source covariance to target, then fit
    a standard CatBoost on the union of (aligned source, target).
    """

    def __init__(self, base_params: dict | None = None):
        self.base_params = base_params or dict(
            iterations=400, depth=6, learning_rate=0.05, l2_leaf_reg=3.0,
            random_seed=SEED, verbose=False, allow_writing_files=False,
        )
        self.A_: np.ndarray | None = None
        self.mean_s_: np.ndarray | None = None
        self.mean_t_: np.ndarray | None = None
        self.model_: CatBoostRegressor | None = None

    def _coral_transform(self, X_source: np.ndarray, X_target: np.ndarray) -> np.ndarray:
        eps = 1e-3
        d = X_source.shape[1]
        Xs_c = X_source - X_source.mean(axis=0, keepdims=True)
        Xt_c = X_target - X_target.mean(axis=0, keepdims=True)
        cov_s = (Xs_c.T @ Xs_c) / max(X_source.shape[0] - 1, 1) + eps * np.eye(d)
        cov_t = (Xt_c.T @ Xt_c) / max(X_target.shape[0] - 1, 1) + eps * np.eye(d)

        Us, Ss, _ = np.linalg.svd(cov_s)
        Ut, St, _ = np.linalg.svd(cov_t)
        whiten = Us @ np.diag(1.0 / np.sqrt(Ss)) @ Us.T
        recolor = Ut @ np.diag(np.sqrt(St)) @ Ut.T
        self.A_ = whiten @ recolor

        self.mean_s_ = X_source.mean(axis=0)
        self.mean_t_ = X_target.mean(axis=0)
        return (X_source - self.mean_s_) @ self.A_ + self.mean_t_

    def fit(self, X_source: np.ndarray, y_source: np.ndarray,
            X_target: np.ndarray, y_target: np.ndarray):
        X_s_aligned = self._coral_transform(X_source, X_target)
        X_combined = np.vstack([X_s_aligned, X_target])
        y_combined = np.concatenate([y_source, y_target])
        self.model_ = CatBoostRegressor(**self.base_params)
        self.model_.fit(X_combined, y_combined)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)


def coral_align(X_source: np.ndarray, y_source: np.ndarray,
                X_target: np.ndarray, y_target: np.ndarray,
                base_params: dict | None = None) -> CoralAdaptedModel:
    model = CoralAdaptedModel(base_params=base_params)
    model.fit(X_source, y_source, X_target, y_target)
    return model


# ──────────────────────────────────────────────────────────
# Target-only baseline for comparison + the gate
# ──────────────────────────────────────────────────────────

def target_only_catboost(X_target: np.ndarray, y_target: np.ndarray,
                         params: dict | None = None) -> CatBoostRegressor:
    p = dict(
        iterations=400, depth=6, learning_rate=0.05, l2_leaf_reg=3.0,
        random_seed=SEED, verbose=False, allow_writing_files=False,
    )
    if params:
        p.update(params)
    m = CatBoostRegressor(**p)
    m.fit(X_target, y_target)
    return m


@dataclass
class GateDecision:
    chose_transfer: bool
    transfer_inner_rmse: float
    target_only_inner_rmse: float


def _inner_rmse(fit_fn, X_t: np.ndarray, y_t: np.ndarray,
                n_splits: int = 3, **fit_kwargs) -> float:
    """Return mean RMSE of fit_fn(X_train, y_train, **fit_kwargs).predict(X_val)
    across an inner KFold on the target only. fit_kwargs is forwarded as-is."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scores = []
    for tr, va in kf.split(X_t):
        model = fit_fn(X_target=X_t[tr], y_target=y_t[tr], **fit_kwargs)
        scores.append(np.sqrt(mean_squared_error(y_t[va], model.predict(X_t[va]))))
    return float(np.mean(scores))


def negative_transfer_gate(transfer_fit_fn,
                           X_source: np.ndarray, y_source: np.ndarray,
                           X_target: np.ndarray, y_target: np.ndarray,
                           n_inner_splits: int = 3,
                           source_keys: np.ndarray | None = None,
                           target_keys: np.ndarray | None = None
                           ) -> tuple[object, GateDecision]:
    """Choose transfer if it beats target-only on inner CV; otherwise fall back.

    transfer_fit_fn must accept (X_source, y_source, X_target, y_target)
    and return a fitted model. Returns (chosen_model, decision).

    When source_keys/target_keys are given, source rows belonging to the
    students in each inner validation split are dropped for that split, so
    the inner comparison is free of the same student-identity leak that the
    outer loop removes.
    """
    kf = KFold(n_splits=n_inner_splits, shuffle=True, random_state=SEED)
    transfer_scores = []
    target_only_scores = []
    for tr, va in kf.split(X_target):
        if source_keys is not None and target_keys is not None:
            keep = ~np.isin(source_keys, target_keys[va])
            Xs_in, ys_in = X_source[keep], y_source[keep]
        else:
            Xs_in, ys_in = X_source, y_source
        m_t = transfer_fit_fn(Xs_in, ys_in, X_target[tr], y_target[tr])
        transfer_scores.append(np.sqrt(mean_squared_error(
            y_target[va], m_t.predict(X_target[va]))))

        m_o = target_only_catboost(X_target[tr], y_target[tr])
        target_only_scores.append(np.sqrt(mean_squared_error(
            y_target[va], m_o.predict(X_target[va]))))

    rmse_transfer = float(np.mean(transfer_scores))
    rmse_target_only = float(np.mean(target_only_scores))
    chose_transfer = rmse_transfer <= rmse_target_only

    if chose_transfer:
        chosen = transfer_fit_fn(X_source, y_source, X_target, y_target)
    else:
        chosen = target_only_catboost(X_target, y_target)

    return chosen, GateDecision(
        chose_transfer=chose_transfer,
        transfer_inner_rmse=rmse_transfer,
        target_only_inner_rmse=rmse_target_only,
    )


# ──────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────

def main():
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_loader import StudentPerformanceDataset

    rng = np.random.default_rng(SEED)

    ds_por = StudentPerformanceDataset(data_dir='./data')
    ds_por.download_and_load(subject_subset='portuguese')
    X_s, y_s = ds_por.preprocess(target='G3', include_prior_grades=False,
                                  scale_numeric=False)

    ds_mat = StudentPerformanceDataset(data_dir='./data')
    ds_mat.download_and_load(subject_subset='math')
    X_t_full, y_t_full = ds_mat.preprocess(target='G3',
                                           include_prior_grades=False,
                                           scale_numeric=False)

    # Reconcile column count: take intersection of feature columns
    n_feat = min(X_s.shape[1], X_t_full.shape[1])
    X_s = X_s[:, :n_feat]
    X_t_full = X_t_full[:, :n_feat]

    idx = np.arange(X_t_full.shape[0])
    rng.shuffle(idx)
    split = int(len(idx) * 0.7)
    tr, te = idx[:split], idx[split:]
    X_t, y_t = X_t_full[tr], y_t_full[tr]
    X_e, y_e = X_t_full[te], y_t_full[te]

    print(f"\n[smoke] source={X_s.shape}  target_train={X_t.shape}  target_eval={X_e.shape}")

    print("\n[A] warm-start CatBoost ...")
    m_a = warm_start_catboost(X_s, y_s, X_t, y_t,
                              finetune_iterations=80, finetune_lr=0.05)
    rmse_a = float(np.sqrt(mean_squared_error(y_e, m_a.predict(X_e))))
    print(f"    RMSE = {rmse_a:.3f}")

    print("\n[B] TrAdaBoost.R2 ...")
    m_b = tradaboost_r2(X_s, y_s, X_t, y_t, n_estimators=15, max_depth=5)
    rmse_b = float(np.sqrt(mean_squared_error(y_e, m_b.predict(X_e))))
    print(f"    RMSE = {rmse_b:.3f}")

    print("\n[C] CORAL ...")
    m_c = coral_align(X_s, y_s, X_t, y_t,
                      base_params=dict(iterations=200, depth=6,
                                        learning_rate=0.05, l2_leaf_reg=3.0,
                                        random_seed=SEED, verbose=False,
                                        allow_writing_files=False))
    rmse_c = float(np.sqrt(mean_squared_error(y_e, m_c.predict(X_e))))
    print(f"    RMSE = {rmse_c:.3f}")

    print("\n[D] target-only baseline ...")
    m_d = target_only_catboost(X_t, y_t,
                               params=dict(iterations=200, depth=6,
                                            learning_rate=0.05))
    rmse_d = float(np.sqrt(mean_squared_error(y_e, m_d.predict(X_e))))
    print(f"    RMSE = {rmse_d:.3f}")

    print("\n[Gate] negative-transfer gate (warm-start) ...")
    chosen, decision = negative_transfer_gate(
        lambda Xs, ys, Xt, yt: warm_start_catboost(
            Xs, ys, Xt, yt, finetune_iterations=80, finetune_lr=0.05),
        X_s, y_s, X_t, y_t, n_inner_splits=3,
    )
    rmse_gate = float(np.sqrt(mean_squared_error(y_e, chosen.predict(X_e))))
    print(f"    decision={decision}, eval RMSE = {rmse_gate:.3f}")


if __name__ == '__main__':
    main()
