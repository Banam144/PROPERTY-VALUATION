"""L3/L6 inference bundle and hyperparameter configurations for the Hanoi property valuation project.

These bundles reuse the original feature pipeline and never modify the
existing LightGBM champion. All regressor predictions are log(VND/m2),
then converted to VND at inference time.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.features import HanoiValuationFeaturePipeline


class CourseValuationBundle:
    """Fitted feature pipeline + one L3/L6 regression model + Val-calibrated alpha.

    The bundled regressor's ``predict`` must return log(price/area), not VND.
    For an MLP using TransformedTargetRegressor, this is the output after
    inverse target standardization (not its internally standardized target).
    """

    def __init__(
        self,
        feature_pipeline: HanoiValuationFeaturePipeline,
        regressor: Any,
        alpha: float,
        model_name: str,
    ) -> None:
        self.feature_pipeline = feature_pipeline
        self.regressor = regressor
        self.alpha = float(alpha)
        self.model_name = str(model_name)

    def predict_from_features(self, X_feat: pd.DataFrame) -> np.ndarray:
        """Predict VND for a feature frame with the trained feature schema."""
        expected = self.feature_pipeline.feature_names_
        if list(X_feat.columns) != expected:
            raise ValueError("Feature names/order differ from those used for training.")
        areas = X_feat["area"].to_numpy(dtype="float64")
        if (not np.isfinite(areas).all()) or np.any(areas <= 0):
            raise ValueError("'area' must contain finite positive m2 values.")

        X_imp = self.feature_pipeline.impute_with_train_medians(X_feat)
        if not np.isfinite(X_imp.to_numpy(dtype="float64")).all():
            raise ValueError("Feature matrix contains NaN or infinite values after imputation.")

        log_ppm2 = np.asarray(self.regressor.predict(X_imp), dtype="float64")
        if log_ppm2.ndim != 1 or len(log_ppm2) != len(areas):
            raise ValueError("Regressor must output one log(price/area) per property.")
        if not np.isfinite(log_ppm2).all():
            raise ValueError("Regressor returned non-finite log unit prices.")

        with np.errstate(over="ignore", invalid="ignore"):
            pred_vnd = self.alpha * areas * np.exp(log_ppm2)
        if (not np.isfinite(pred_vnd).all()) or np.any(pred_vnd <= 0):
            raise ValueError("Price predictions must be finite and positive in VND.")
        return pred_vnd

    def predict(self, raw_df: pd.DataFrame) -> np.ndarray:
        """Predict total asking price VND from a raw property DataFrame."""
        X_feat = self.feature_pipeline.transform(raw_df)
        return self.predict_from_features(X_feat)


# NEW [L3/L6-01]: Candidate configurations are explicit and are selected on Validation.
def course_candidate_configs(kind: str) -> list[dict[str, Any]]:
    """Small, fixed search space. NEVER choose hyperparameters using Test."""
    if kind == "tree":
        return [
            {"max_depth": 8, "min_samples_leaf": 30},
            {"max_depth": 12, "min_samples_leaf": 20},
            {"max_depth": 16, "min_samples_leaf": 10},
        ]
    if kind == "mlp":
        return [
            {"hidden_layer_sizes": [64, 32], "alpha": 0.01, "max_iter": 180},
            {"hidden_layer_sizes": [128, 64], "alpha": 0.001, "max_iter": 220},
        ]
    raise ValueError(f"Unknown course model: {kind}")


