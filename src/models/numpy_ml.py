"""Pure-NumPy machine-learning implementations.

These are the fallbacks RetailPulse uses when scikit-learn is unavailable. They
are real implementations, not stubs: each is validated against the textbook
definition in ``tests/test_retailpulse.py``.

    KMeans          - Lloyd's algorithm with k-means++ seeding
    LogisticRegression - L2-regularised, trained by gradient descent
    DecisionTree    - CART with Gini impurity, used both directly and as the
                      base learner for RandomForest and GradientBoosting
    RandomForest    - bagged trees with per-split feature subsampling
    GradientBoosting- stagewise additive trees fitted to the logistic loss
                      gradient (the same idea as XGBoost, simplified)

Everything is deterministic given a seed, so results are reproducible.
"""
from __future__ import annotations

import numpy as np


# ==========================================================================
# Utilities
# ==========================================================================
def standardise(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (scaled, mean, std) so the transform can be reused on new data."""
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (x - mu) / sd, mu, sd


def train_test_split_stratified(y: np.ndarray, test_frac: float, seed: int):
    """Indices for a stratified split, preserving the class balance."""
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_test = max(int(round(len(idx) * test_frac)), 1)
        test_idx.append(idx[:n_test])
        train_idx.append(idx[n_test:])
    return np.sort(np.concatenate(train_idx)), np.sort(np.concatenate(test_idx))


# ==========================================================================
# K-Means
# ==========================================================================
def _kmeans_plusplus(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding: spread the initial centroids out probabilistically."""
    n = len(x)
    centres = [x[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min(
            np.stack([((x - c) ** 2).sum(axis=1) for c in centres]), axis=0
        )
        total = d2.sum()
        if total <= 0:
            centres.append(x[rng.integers(n)])
            continue
        probs = d2 / total
        centres.append(x[rng.choice(n, p=probs)])
    return np.array(centres)


class KMeans:
    """Lloyd's algorithm with k-means++ initialisation and n_init restarts."""

    def __init__(self, n_clusters: int = 4, n_init: int = 10, max_iter: int = 300,
                 tol: float = 1e-6, random_state: int = 42) -> None:
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _single_run(self, x: np.ndarray, rng: np.random.Generator):
        centres = _kmeans_plusplus(x, self.n_clusters, rng)
        labels = np.zeros(len(x), dtype=int)
        for _ in range(self.max_iter):
            # assignment step
            d = np.stack([((x - c) ** 2).sum(axis=1) for c in centres], axis=1)
            new_labels = d.argmin(axis=1)

            # update step; empty clusters are re-seeded on the furthest point
            new_centres = centres.copy()
            for j in range(self.n_clusters):
                members = x[new_labels == j]
                if len(members):
                    new_centres[j] = members.mean(axis=0)
                else:
                    far = d.min(axis=1).argmax()
                    new_centres[j] = x[far]

            shift = float(np.sqrt(((new_centres - centres) ** 2).sum(axis=1)).max())
            centres, labels = new_centres, new_labels
            if shift <= self.tol:
                break

        inertia = float(((x - centres[labels]) ** 2).sum())
        return centres, labels, inertia

    def fit(self, x: np.ndarray) -> "KMeans":
        rng = np.random.default_rng(self.random_state)
        best = None
        for _ in range(self.n_init):
            centres, labels, inertia = self._single_run(x, rng)
            if best is None or inertia < best[2]:
                best = (centres, labels, inertia)
        self.cluster_centers_ = best[0]
        self.labels_ = best[1]
        self.inertia_ = best[2]
        return self

    def fit_predict(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).labels_

    def predict(self, x: np.ndarray) -> np.ndarray:
        d = np.stack([((x - c) ** 2).sum(axis=1) for c in self.cluster_centers_], axis=1)
        return d.argmin(axis=1)


# ==========================================================================
# Logistic regression
# ==========================================================================
class LogisticRegression:
    """Binary logistic regression trained by full-batch gradient descent."""

    def __init__(self, lr: float = 0.1, epochs: int = 600, l2: float = 1e-3,
                 class_weight: str | None = "balanced", random_state: int = 42) -> None:
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        n, d = x.shape
        self.coef_ = np.zeros(d)
        self.intercept_ = 0.0

        if self.class_weight == "balanced":
            n_pos = max(int((y == 1).sum()), 1)
            n_neg = max(int((y == 0).sum()), 1)
            w = np.where(y == 1, n / (2.0 * n_pos), n / (2.0 * n_neg))
        else:
            w = np.ones(n)

        for _ in range(self.epochs):
            z = x @ self.coef_ + self.intercept_
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            err = (p - y) * w
            grad_w = x.T @ err / n + self.l2 * self.coef_
            grad_b = err.sum() / n
            self.coef_ -= self.lr * grad_w
            self.intercept_ -= self.lr * grad_b
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p1 = 1.0 / (1.0 + np.exp(-np.clip(x @ self.coef_ + self.intercept_, -30, 30)))
        return np.column_stack([1 - p1, p1])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


# ==========================================================================
# Decision tree (CART, Gini impurity)
# ==========================================================================
class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value", "is_leaf")

    def __init__(self, *, value=None, feature=None, threshold=None,
                 left=None, right=None, is_leaf=False):
        self.value = value
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.is_leaf = is_leaf


class DecisionTree:
    """Binary CART tree. For regression, splits minimise squared error."""

    def __init__(self, max_depth: int = 4, min_samples_leaf: int = 5,
                 max_features: int | None = None, task: str = "classification",
                 random_state: int = 42) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.task = task
        self.random_state = random_state

    # ---- impurity -------------------------------------------------------
    def _leaf_value(self, y: np.ndarray) -> float:
        if self.task == "regression":
            return float(y.mean())
        p = float(y.mean()) if len(y) else 0.0
        return p

    def _score(self, y: np.ndarray) -> float:
        """Gini impurity (classification) or variance (regression)."""
        if len(y) == 0:
            return 0.0
        if self.task == "regression":
            return float(y.var())
        p = float(y.mean())
        return float(2 * p * (1 - p))

    def fit(self, x: np.ndarray, y: np.ndarray) -> "DecisionTree":
        self.rng_ = np.random.default_rng(self.random_state)
        self.n_features_ = x.shape[1]
        self.root_ = self._build(x, y.astype(float), depth=0)
        return self

    def _build(self, x: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        if (depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf
                or len(np.unique(y)) == 1):
            return _Node(value=self._leaf_value(y), is_leaf=True)

        feature, threshold, gain = self._best_split(x, y)
        if feature is None or gain <= 0:
            return _Node(value=self._leaf_value(y), is_leaf=True)

        left_mask = x[:, feature] <= threshold
        right_mask = ~left_mask
        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            return _Node(value=self._leaf_value(y), is_leaf=True)

        return _Node(
            feature=feature, threshold=threshold,
            left=self._build(x[left_mask], y[left_mask], depth + 1),
            right=self._build(x[right_mask], y[right_mask], depth + 1),
        )

    def _best_split(self, x: np.ndarray, y: np.ndarray):
        """Exhaustive scan over candidate thresholds on a feature subset."""
        n, d = x.shape
        parent = self._score(y)

        if self.max_features is None or self.max_features >= d:
            features = range(d)
        else:
            features = self.rng_.choice(d, size=self.max_features, replace=False)

        best = (None, None, 0.0)
        for f in features:
            col = x[:, f]
            # candidate thresholds: midpoints between consecutive unique values,
            # thinned out so the scan stays linear rather than quadratic
            uniq = np.unique(col)
            if len(uniq) < 2:
                continue
            if len(uniq) > 32:
                qs = np.quantile(uniq, np.linspace(0.05, 0.95, 31))
                uniq = np.unique(qs)
            thresholds = (uniq[:-1] + uniq[1:]) / 2.0

            for t in thresholds:
                mask = col <= t
                nl, nr = int(mask.sum()), int((~mask).sum())
                if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                    continue
                weighted = (nl * self._score(y[mask]) + nr * self._score(y[~mask])) / n
                gain = parent - weighted
                if gain > best[2]:
                    best = (f, float(t), gain)
        return best

    def _descend(self, node: _Node, row: np.ndarray) -> float:
        while not node.is_leaf:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.value

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p = np.array([self._descend(self.root_, row) for row in x])
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.task == "regression":
            return np.array([self._descend(self.root_, row) for row in x])
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)

    # ---- feature importance (impurity decrease) --------------------------
    def feature_importance(self, n_features: int) -> np.ndarray:
        imp = np.zeros(n_features)

        def walk(node: _Node, weight: float):
            if node.is_leaf:
                return
            imp[node.feature] += weight
            walk(node.left, weight * 0.5)
            walk(node.right, weight * 0.5)

        walk(self.root_, 1.0)
        total = imp.sum()
        return imp / total if total > 0 else imp


# ==========================================================================
# Random forest
# ==========================================================================
class RandomForest:
    """Bagged decision trees with per-split feature subsampling."""

    def __init__(self, n_estimators: int = 120, max_depth: int = 10,
                 min_samples_leaf: int = 3, max_features: str | int = "sqrt",
                 random_state: int = 42) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RandomForest":
        rng = np.random.default_rng(self.random_state)
        n, d = x.shape
        if self.max_features == "sqrt":
            mf = max(int(np.sqrt(d)), 1)
        elif self.max_features == "log2":
            mf = max(int(np.log2(d)), 1)
        else:
            mf = int(self.max_features) if self.max_features else d

        self.trees_: list[DecisionTree] = []
        for i in range(self.n_estimators):
            idx = rng.integers(0, n, n)          # bootstrap sample
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=mf,
                task="classification",
                random_state=int(rng.integers(1 << 30)),
            ).fit(x[idx], y[idx])
            self.trees_.append(tree)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        probs = np.mean([t.predict_proba(x)[:, 1] for t in self.trees_], axis=0)
        return np.column_stack([1 - probs, probs])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


# ==========================================================================
# Gradient boosting (logistic loss)
# ==========================================================================
class GradientBoosting:
    """Stagewise additive model: each tree fits the negative gradient of the loss."""

    def __init__(self, n_estimators: int = 120, learning_rate: float = 0.08,
                 max_depth: int = 3, min_samples_leaf: int = 5,
                 subsample: float = 0.9, random_state: int = 42) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GradientBoosting":
        rng = np.random.default_rng(self.random_state)
        n = len(y)
        yf = y.astype(float)

        base = np.clip(yf.mean(), 1e-6, 1 - 1e-6)
        self.base_score_ = float(np.log(base / (1 - base)))
        raw = np.full(n, self.base_score_)
        self.trees_: list[DecisionTree] = []

        for _ in range(self.n_estimators):
            p = 1.0 / (1.0 + np.exp(-np.clip(raw, -30, 30)))
            residual = yf - p                      # negative gradient for logloss
            mask = np.ones(n, dtype=bool)
            if self.subsample < 1.0:
                mask = rng.random(n) < self.subsample
                if mask.sum() < max(self.min_samples_leaf * 2, 10):
                    mask = np.ones(n, dtype=bool)

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                task="regression",
                random_state=int(rng.integers(1 << 30)),
            ).fit(x[mask], residual[mask])

            update = tree.predict(x)
            raw += self.learning_rate * update
            self.trees_.append(tree)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        raw = np.full(len(x), self.base_score_)
        for tree in self.trees_:
            raw += self.learning_rate * tree.predict(x)
        p1 = 1.0 / (1.0 + np.exp(-np.clip(raw, -30, 30)))
        return np.column_stack([1 - p1, p1])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)

    def feature_importance(self, n_features: int) -> np.ndarray:
        imp = np.mean([t.feature_importance(n_features) for t in self.trees_], axis=0)
        total = imp.sum()
        return imp / total if total > 0 else imp
