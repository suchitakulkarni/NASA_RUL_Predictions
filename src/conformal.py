# src/conformal.py
import logging
import os

import joblib
import numpy as np

logger = logging.getLogger(__name__)


class ConformalPredictor:
    """
    Split-conformal predictor with cluster-stratified nonconformity quantiles.

    Nonconformity score: absolute residual |y - y_hat|.
    The finite-sample corrected quantile ceil((n+1)*(1-alpha))/n is used per
    cluster so coverage holds marginally at the target level.
    A global fallback quantile handles operating-condition clusters that were
    unseen at calibration time.
    """

    def __init__(self):
        self.quantiles_per_cluster = {}
        self.global_quantile = None
        self.alpha = None

    def calibrate(self, model, X_cal, y_cal, clusters_cal, alpha=0.1):
        """
        Compute per-cluster nonconformity quantiles on a held-out calibration set.

        Parameters
        ----------
        model        : fitted RULModel with a predict(X) method
        X_cal        : (n, p) feature matrix for calibration samples
        y_cal        : (n,) true RUL values
        clusters_cal : (n,) integer operating-condition cluster per sample
        alpha        : miscoverage level — 0.1 targets 90% coverage

        Returns self.
        """
        self.alpha = alpha
        y_cal = np.asarray(y_cal, dtype=float)
        clusters_cal = np.asarray(clusters_cal, dtype=int)

        y_pred = model.predict(X_cal)
        scores = np.abs(y_cal - y_pred)

        logger.info(
            "Calibration: %d samples, alpha=%.2f, "
            "score range [%.2f, %.2f], mean=%.2f",
            len(scores), alpha, scores.min(), scores.max(), scores.mean(),
        )

        target_level = 1.0 - alpha

        for c in np.unique(clusters_cal):
            mask = clusters_cal == c
            cs = scores[mask]
            n = len(cs)
            level = min(np.ceil((n + 1) * target_level) / n, 1.0)
            q = float(np.quantile(cs, level))
            self.quantiles_per_cluster[int(c)] = q
            logger.info(
                "Cluster %d: n=%d, corrected_level=%.4f, quantile=%.4f",
                int(c), n, level, q,
            )

        n_all = len(scores)
        level_all = min(np.ceil((n_all + 1) * target_level) / n_all, 1.0)
        self.global_quantile = float(np.quantile(scores, level_all))
        logger.info("Global fallback quantile: %.4f", self.global_quantile)

        _, lower, upper = self.predict_with_interval(model, X_cal, clusters_cal)
        coverage = float(np.mean((y_cal >= lower) & (y_cal <= upper)))
        logger.info(
            "Calibration coverage: %.4f (target=%.4f, n=%d)",
            coverage, target_level, len(y_cal),
        )
        return self

    def _interval_from_predictions(self, y_pred, clusters):
        n = len(y_pred)
        lower = np.empty(n)
        upper = np.empty(n)
        for i in range(n):
            q = self.quantiles_per_cluster.get(int(clusters[i]), self.global_quantile)
            lower[i] = max(y_pred[i] - q, 0.0)
            upper[i] = y_pred[i] + q
        return lower, upper

    def predict_with_interval(self, model, X, clusters):
        """
        Return (y_pred, lower, upper) in a single model call.

        Parameters
        ----------
        model    : fitted RULModel
        X        : (n, p) feature matrix
        clusters : (n,) integer cluster label per sample

        Returns
        -------
        y_pred, lower, upper : arrays of shape (n,)
        """
        if self.global_quantile is None:
            raise RuntimeError("ConformalPredictor must be calibrated before predict")
        clusters = np.asarray(clusters, dtype=int)
        y_pred = model.predict(X)
        lower, upper = self._interval_from_predictions(y_pred, clusters)
        return y_pred, lower, upper

    def predict_interval(self, model, X, clusters):
        """Return (lower, upper) conformal prediction intervals."""
        _, lower, upper = self.predict_with_interval(model, X, clusters)
        return lower, upper

    def save(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump(self, path)
        logger.info("ConformalPredictor saved to %s", path)

    @classmethod
    def load(cls, path):
        obj = joblib.load(path)
        logger.info("ConformalPredictor loaded from %s", path)
        return obj
