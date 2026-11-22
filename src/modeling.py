"""Stage 5 (P0): Multi-Tier Valuation Modeling, Validation Model Selection,
AVM Metrics, and 95% Bootstrap Confidence Intervals.

Models implemented:
1. `NaiveDistrictMedianBaseline`: Median (price/area) per (property_type, district) on Train * area.
2. `RidgeLogLinear`: Train-group-median imputation + StandardScaler + Ridge on log(price/area).
3. `SklearnHistGBR`: Native-NaN HistGradientBoostingRegressor (L1 loss) on log(price/area).
4. `LightGBM_Unified`: Single global LightGBM (Huber loss on log(price/area)) with Val early stopping.
5. `LightGBM_Segmented`: Dedicated specialist LightGBM models for `Nhà` and `Căn hộ chung cư`.
"""

from __future__ import annotations

import time
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import json
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.exceptions import ConvergenceWarning
from src.course_models import (
    CourseValuationBundle,
    course_candidate_configs,
)
import warnings
from src.features import HanoiValuationFeaturePipeline


def compute_avm_metrics(y_true_vnd: np.ndarray, y_pred_vnd: np.ndarray) -> dict[str, float]:
    """Compute comprehensive Automated Valuation Model (AVM) metrics in VNĐ and Log space."""
    y_t = np.asarray(y_true_vnd, dtype="float64")
    y_p = np.clip(np.asarray(y_pred_vnd, dtype="float64"), 1e6, None)

    abs_pct_err = np.abs(y_t - y_p) / y_t * 100.0
    y_t_ty = y_t / 1e9
    y_p_ty = y_p / 1e9

    log_t = np.log(y_t)
    log_p = np.log(y_p)

    return {
        "MAPE_pct": float(np.mean(abs_pct_err)),
        "MdAPE_pct": float(np.median(abs_pct_err)),
        "PPE10_pct": float(np.mean(abs_pct_err <= 10.0) * 100.0),
        "PPE15_pct": float(np.mean(abs_pct_err <= 15.0) * 100.0),
        "PPE20_pct": float(np.mean(abs_pct_err <= 20.0) * 100.0),
        "MAE_ty": float(mean_absolute_error(y_t_ty, y_p_ty)),
        "MedAE_ty": float(np.median(np.abs(y_t_ty - y_p_ty))),
        "RMSE_ty": float(np.sqrt(mean_squared_error(y_t_ty, y_p_ty))),
        "R2_price": float(r2_score(y_t_ty, y_p_ty)),
        "R2_log": float(r2_score(log_t, log_p)),
        "RMSLE": float(np.sqrt(mean_squared_error(log_t, log_p))),
    }


def bootstrap_avm_confidence_intervals(
    y_true_vnd: np.ndarray,
    y_pred_vnd: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Compute 95% Bootstrap Confidence Intervals [p2.5, p97.5] for key AVM metrics."""
    rng = np.random.default_rng(seed)
    y_t = np.asarray(y_true_vnd, dtype="float64")
    y_p = np.clip(np.asarray(y_pred_vnd, dtype="float64"), 1e6, None)
    n = len(y_t)

    abs_pct = np.abs(y_t - y_p) / y_t * 100.0
    abs_ty = np.abs(y_t - y_p) / 1e9
    sq_ty = ((y_t - y_p) / 1e9) ** 2
    log_t = np.log(y_t)
    log_p = np.log(y_p)
    sq_log = (log_t - log_p) ** 2

    boot_mape = np.empty(n_boot, dtype="float64")
    boot_mdape = np.empty(n_boot, dtype="float64")
    boot_ppe15 = np.empty(n_boot, dtype="float64")
    boot_ppe20 = np.empty(n_boot, dtype="float64")
    boot_mae = np.empty(n_boot, dtype="float64")
    boot_rmse = np.empty(n_boot, dtype="float64")
    boot_r2_log = np.empty(n_boot, dtype="float64")

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ap = abs_pct[idx]
        boot_mape[b] = np.mean(ap)
        boot_mdape[b] = np.median(ap)
        boot_ppe15[b] = np.mean(ap <= 15.0) * 100.0
        boot_ppe20[b] = np.mean(ap <= 20.0) * 100.0
        boot_mae[b] = np.mean(abs_ty[idx])
        boot_rmse[b] = np.sqrt(np.mean(sq_ty[idx]))
        lt_b = log_t[idx]
        ss_tot = np.sum((lt_b - np.mean(lt_b)) ** 2)
        boot_r2_log[b] = 1.0 - (np.sum(sq_log[idx]) / ss_tot) if ss_tot > 0 else 0.0

    def summarize(arr: np.ndarray, point: float) -> dict[str, float]:
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return {"point": float(point), "ci95_low": float(lo), "ci95_high": float(hi)}

    base = compute_avm_metrics(y_t, y_p)
    return {
        "MAPE_pct": summarize(boot_mape, base["MAPE_pct"]),
        "MdAPE_pct": summarize(boot_mdape, base["MdAPE_pct"]),
        "PPE15_pct": summarize(boot_ppe15, base["PPE15_pct"]),
        "PPE20_pct": summarize(boot_ppe20, base["PPE20_pct"]),
        "MAE_ty": summarize(boot_mae, base["MAE_ty"]),
        "RMSE_ty": summarize(boot_rmse, base["RMSE_ty"]),
        "R2_log": summarize(boot_r2_log, base["R2_log"]),
    }


def tune_multiplicative_mape_factor(
    y_true_vnd: np.ndarray, raw_pred_vnd: np.ndarray
) -> float:
    """Find optimal multiplicative calibration factor alpha in [0.90, 1.05] on Validation
    to align log-space predictions with arithmetic percentage error (MAPE/MdAPE balance)."""
    y_t = np.asarray(y_true_vnd, dtype="float64")
    y_p = np.asarray(raw_pred_vnd, dtype="float64")
    alphas = np.linspace(0.90, 1.05, 151)
    best_alpha = 1.0
    best_loss = float("inf")
    for a in alphas:
        ape = np.abs(y_t - a * y_p) / y_t * 100.0
        # Balanced objective: 0.5 * MAPE + 0.5 * MdAPE
        score = 0.5 * float(np.mean(ape)) + 0.5 * float(np.median(ape))
        if score < best_loss:
            best_loss = score
            best_alpha = float(a)
    return best_alpha


class NaiveDistrictMedianBaseline(BaseEstimator, RegressorMixin):
    """Rule-of-thumb baseline: Median unit price per (property_type, district) on Train * area."""

    def fit(self, df_train: pd.DataFrame, y_price_train: pd.Series | np.ndarray) -> "NaiveDistrictMedianBaseline":
        work = pd.DataFrame(
            {
                "ptype": df_train["property_type_name"].astype(str).str.strip(),
                "dist": df_train["district_name"].astype(str).str.strip(),
                "ppm2": np.asarray(y_price_train, dtype="float64")
                / pd.to_numeric(df_train["area"], errors="coerce").to_numpy(dtype="float64"),
            }
        )
        self.global_median_ppm2_ = float(work["ppm2"].median())
        self.type_median_ppm2_ = work.groupby("ptype")["ppm2"].median().to_dict()
        self.dist_median_ppm2_ = work.groupby(["ptype", "dist"])["ppm2"].median().to_dict()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        ptypes = df["property_type_name"].astype(str).str.strip().tolist()
        dists = df["district_name"].astype(str).str.strip().tolist()
        areas = pd.to_numeric(df["area"], errors="coerce").to_numpy(dtype="float64")
        preds = np.empty(len(df), dtype="float64")
        for i, (pt, di) in enumerate(zip(ptypes, dists)):
            unit = self.dist_median_ppm2_.get(
                (pt, di), self.type_median_ppm2_.get(pt, self.global_median_ppm2_)
            )
            preds[i] = unit * areas[i]
        return preds

class DecisionTreeValuationModel(BaseEstimator, RegressorMixin):
    """Decision Tree Regressor predicting `log(price / area)` with Huber loss and Validation early stopping."""
    def __init__(
        self,
        max_depth=12,
        min_samples_leaf=20,
        random_state=42,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X_tr_imp, y_tr_unit):
        leaf = int(self.min_samples_leaf)

        self.model_ = DecisionTreeRegressor(
            criterion="squared_error",
            max_depth=self.max_depth,
            min_samples_split=2 * leaf,
            min_samples_leaf=leaf,
            random_state=self.random_state,
        )

        self.model_.fit(X_tr_imp, y_tr_unit)
        return self

    def predict(self, X_imp):
        if not hasattr(self, "model_"):
            raise RuntimeError("Model chưa được fit")

        return self.model_.predict(X_imp)
    def training_details(self) -> dict[str, Any]:
        if not hasattr(self, "model_"):
            raise RuntimeError("Fit the Decision Tree before inspecting it.")
        return {
            "tree_depth": int(self.model_.get_depth()),
            "tree_leaves": int(self.model_.get_n_leaves()),
        }

class MLPValuationModel(BaseEstimator, RegressorMixin):
    """Scaled MLP trained on log(price/area); predict() returns log(price/area).

    Scaling X and the *training target* is fitted only on Train. The target
    transformer is inverted automatically inside TransformedTargetRegressor.
    """
    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] | list[int] = (128, 64),
        alpha: float = 0.001,
        max_iter: int = 220,
        random_state: int = 42,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X_tr_imp: pd.DataFrame, y_tr_unit: np.ndarray) -> "MLPValuationModel":
        net = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=tuple(self.hidden_layer_sizes),
                activation="relu",
                solver="adam",
                alpha=float(self.alpha),
                batch_size=256,
                learning_rate_init=0.001,
                max_iter=int(self.max_iter),
                early_stopping=False,  # Use the external chronological Validation.
                n_iter_no_change=12,
                random_state=self.random_state,
            )),
        ])
        self.model_ = TransformedTargetRegressor(
            regressor=net,
            transformer=StandardScaler(),
        )
        self.model_.fit(X_tr_imp, y_tr_unit)
        return self

    def predict(self, X_imp: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("MLPValuationModel must be fitted before predict().")
        # TTR returns the original log(price/area), NOT standardized y.
        return np.asarray(self.model_.predict(X_imp), dtype="float64")

    def training_details(self) -> dict[str, Any]:
        if not hasattr(self, "model_"):
            raise RuntimeError("Fit the MLP before inspecting it.")
        net = self.model_.regressor_.named_steps["mlp"]
        return {
            "mlp_n_iter": int(net.n_iter_),
            "mlp_loss": float(net.loss_),
        }
    
class SegmentedLightGBMValuationModel:
    """Two-expert LightGBM valuation model (separate GBDT for `Nhà` vs `Căn hộ chung cư`)
    predicting `log(price / area)` with Huber loss and Validation early stopping."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.house_alpha_: float = 1.0
        self.apt_alpha_: float = 1.0

    def _make_lgbm(self, is_apt: bool) -> lgb.LGBMRegressor:
        if is_apt:
            return lgb.LGBMRegressor(
                objective="huber",
                alpha=0.35,
                n_estimators=2200,
                learning_rate=0.03,
                num_leaves=63,
                min_child_samples=25,
                subsample=0.85,
                subsample_freq=1,
                colsample_bytree=0.80,
                reg_alpha=1.0,
                reg_lambda=5.0,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
            )
        return lgb.LGBMRegressor(
            objective="huber",
            alpha=0.45,
            n_estimators=2500,
            learning_rate=0.03,
            num_leaves=95,
            min_child_samples=30,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=1.5,
            reg_lambda=6.0,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )

    def fit(
        self,
        X_tr: pd.DataFrame,
        y_tr_vnd: np.ndarray,
        X_va: pd.DataFrame,
        y_va_vnd: np.ndarray,
    ) -> "SegmentedLightGBMValuationModel":
        y_tr_unit = np.log(y_tr_vnd / X_tr["area"].to_numpy(dtype="float64"))
        y_va_unit = np.log(y_va_vnd / X_va["area"].to_numpy(dtype="float64"))

        tr_apt = X_tr["is_apartment"] == 1.0
        va_apt = X_va["is_apartment"] == 1.0

        self.apt_model_ = self._make_lgbm(is_apt=True)
        self.apt_model_.fit(
            X_tr.loc[tr_apt],
            y_tr_unit[tr_apt],
            eval_X=X_va.loc[va_apt],
            eval_y=y_va_unit[va_apt],
            callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False)],
        )

        self.house_model_ = self._make_lgbm(is_apt=False)
        self.house_model_.fit(
            X_tr.loc[~tr_apt],
            y_tr_unit[~tr_apt],
            eval_X=X_va.loc[~va_apt],
            eval_y=y_va_unit[~va_apt],
            callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False)],
        )

        # Calibrate multiplicative scale factor per segment strictly on Validation
        raw_va_apt = np.exp(self.apt_model_.predict(X_va.loc[va_apt])) * X_va.loc[
            va_apt, "area"
        ].to_numpy(dtype="float64")
        raw_va_house = np.exp(self.house_model_.predict(X_va.loc[~va_apt])) * X_va.loc[
            ~va_apt, "area"
        ].to_numpy(dtype="float64")

        self.apt_alpha_ = tune_multiplicative_mape_factor(y_va_vnd[va_apt], raw_va_apt)
        self.house_alpha_ = tune_multiplicative_mape_factor(y_va_vnd[~va_apt], raw_va_house)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        is_apt = (X["is_apartment"] == 1.0).to_numpy()
        areas = X["area"].to_numpy(dtype="float64")
        preds = np.empty(len(X), dtype="float64")

        if is_apt.any():
            log_u_apt = self.apt_model_.predict(X.loc[is_apt])
            preds[is_apt] = self.apt_alpha_ * areas[is_apt] * np.exp(log_u_apt)
        if (~is_apt).any():
            log_u_house = self.house_model_.predict(X.loc[~is_apt])
            preds[~is_apt] = self.house_alpha_ * areas[~is_apt] * np.exp(log_u_house)
        return preds


class UnifiedLightGBMValuationModel:
    """Single global LightGBM model predicting `log(price / area)` across all property types."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.alpha_: float = 1.0

    def fit(
        self,
        X_tr: pd.DataFrame,
        y_tr_vnd: np.ndarray,
        X_va: pd.DataFrame,
        y_va_vnd: np.ndarray,
    ) -> "UnifiedLightGBMValuationModel":
        y_tr_unit = np.log(y_tr_vnd / X_tr["area"].to_numpy(dtype="float64"))
        y_va_unit = np.log(y_va_vnd / X_va["area"].to_numpy(dtype="float64"))

        self.model_ = lgb.LGBMRegressor(
            objective="huber",
            alpha=0.42,
            n_estimators=2500,
            learning_rate=0.03,
            num_leaves=95,
            min_child_samples=30,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.78,
            reg_alpha=1.5,
            reg_lambda=6.0,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        self.model_.fit(
            X_tr,
            y_tr_unit,
            eval_X=X_va,
            eval_y=y_va_unit,
            callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False)],
        )
        raw_va = np.exp(self.model_.predict(X_va)) * X_va["area"].to_numpy(dtype="float64")
        self.alpha_ = tune_multiplicative_mape_factor(y_va_vnd, raw_va)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        log_u = self.model_.predict(X)
        return self.alpha_ * X["area"].to_numpy(dtype="float64") * np.exp(log_u)


class BlendedGBDTValuationBundle:
    """Self-contained production bundle combining:
    - Fitted `HanoiValuationFeaturePipeline` (strictly fitted on Train)
    - `SegmentedLightGBMValuationModel` + `UnifiedLightGBMValuationModel`
      with blend weight `w_seg` selected strictly on Validation.
    """

    def __init__(
        self,
        feature_pipeline: HanoiValuationFeaturePipeline,
        segmented_model: SegmentedLightGBMValuationModel,
        unified_model: UnifiedLightGBMValuationModel,
        w_seg: float,
    ) -> None:
        self.feature_pipeline = feature_pipeline
        self.segmented_model = segmented_model
        self.unified_model = unified_model
        self.w_seg = float(w_seg)

    def predict_from_features(self, X_feat: pd.DataFrame) -> np.ndarray:
        p_seg = self.segmented_model.predict(X_feat)
        p_uni = self.unified_model.predict(X_feat)
        # Geometric blend in log-price space preserves multiplicative scale invariance
        return np.exp(self.w_seg * np.log(p_seg) + (1.0 - self.w_seg) * np.log(p_uni))

    def predict(self, raw_df: pd.DataFrame) -> np.ndarray:
        X_feat = self.feature_pipeline.transform(raw_df)
        return self.predict_from_features(X_feat)

def _train_course_candidates(
    kind: str,
    X_tr_imp: pd.DataFrame,
    y_tr_unit: np.ndarray,
    X_va_imp: pd.DataFrame,
    y_va_vnd: np.ndarray,
    val_area: np.ndarray,
) -> dict[str, Any]:
    """Fit L3 or L6 candidates; return frozen Val-selected model and diagnostics."""
    model_name = {"tree": "7_Decision_Tree_L3", "mlp": "8_MLP_Regressor_L6"}[kind]
    best_score = float("inf")
    best_model: Any = None
    best_alpha = 1.0
    best_config: dict[str, Any] = {}
    best_val_pred: np.ndarray | None = None
    best_details: dict[str, Any] = {}
    candidate_rows: list[dict[str, Any]] = []
    total_fit_seconds = 0.0

    for candidate_no, config in enumerate(course_candidate_configs(kind), start=1):
        # The two named classes are implemented here, next to LightGBM models.
        if kind == "tree":
            model = DecisionTreeValuationModel(**config)
        elif kind == "mlp":
            model = MLPValuationModel(**config)
        else:
            raise ValueError(f"Unknown course model: {kind}")
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X_tr_imp, y_tr_unit)
        fit_seconds = time.perf_counter() - started
        total_fit_seconds += fit_seconds
        convergence_messages = [
            str(w.message) for w in observed
            if issubclass(w.category, ConvergenceWarning)
        ]

        # The model output is log(VND/m²); revert to VND for common AVM metrics.
        log_pred_val = np.asarray(model.predict(X_va_imp), dtype="float64")
        with np.errstate(over="ignore", invalid="ignore"):
            raw_val_vnd = val_area * np.exp(log_pred_val)
        if not np.isfinite(raw_val_vnd).all() or np.any(raw_val_vnd <= 0):
            raise ValueError(f"{model_name} produced invalid Validation prices")

        # Exactly the same alpha calibration as Ridge / HistGBR / LightGBM.
        alpha = tune_multiplicative_mape_factor(y_va_vnd, raw_val_vnd)
        val_pred_vnd = alpha * raw_val_vnd
        metrics = compute_avm_metrics(y_va_vnd, val_pred_vnd)
        score = 0.5 * metrics["MAPE_pct"] + 0.5 * metrics["MdAPE_pct"]
        if not np.isfinite(score):
            raise ValueError(f"Non-finite Validation score from {model_name}")

        candidate_rows.append({
            "model": model_name,
            "candidate": candidate_no,
            "config": json.dumps(config, ensure_ascii=False),
            "alpha": float(alpha),
            "val_selection_score": float(score),
            "fit_seconds": round(fit_seconds, 2),
            "convergence_warnings": " | ".join(convergence_messages),
            **metrics,
        })

        if score < best_score:
            best_score = score
            best_model = model
            best_alpha = float(alpha)
            best_config = dict(config)
            best_val_pred = val_pred_vnd
            best_details = model.training_details()
            if kind == "mlp":
                best_details["mlp_converged_warning"] = bool(convergence_messages)

    assert best_model is not None and best_val_pred is not None
    return {
        "model": best_model,
        "alpha": best_alpha,
        "selected_config": best_config,
        "val_pred": best_val_pred,
        "fit_seconds": total_fit_seconds,
        "val_score": best_score,
        "candidate_rows": candidate_rows,
        **best_details,
    }

def train_and_evaluate_all_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    course_bundles: dict[str, CourseValuationBundle] | None = None,
    include_course_models: bool = True,
) -> tuple[BlendedGBDTValuationBundle, pd.DataFrame, dict[str, Any]]:
    """Train original models and optionally L3/L6 using one fitted feature pipeline.

    Keeps the original 3-item return contract. Optional `course_bundles` receives
    the two new fitted bundles so train_models.py can save them separately.
    """
    y_tr = train_df["price"].to_numpy(dtype="float64")
    y_va = val_df["price"].to_numpy(dtype="float64")
    y_te = test_df["price"].to_numpy(dtype="float64")

    t0 = time.perf_counter()
    feat_pipe = HanoiValuationFeaturePipeline(n_splits=5, random_state=42)
    X_tr = feat_pipe.fit_transform(train_df, y_tr)
    X_va = feat_pipe.transform(val_df)
    X_te = feat_pipe.transform(test_df)
    feat_time = time.perf_counter() - t0

    results_rows: list[dict[str, Any]] = []
    def record(name: str, split_name: str, y_true: np.ndarray, y_pred: np.ndarray, fit_sec: float) -> None:
        m = compute_avm_metrics(y_true, y_pred)
        results_rows.append({"model": name, "split": split_name, "fit_seconds": round(fit_sec, 2), **m})

    # 1. Naive District Median Baseline
    t1 = time.perf_counter()
    naive = NaiveDistrictMedianBaseline().fit(train_df, y_tr)
    dt_naive = time.perf_counter() - t1
    record("1_Naive_District_Median", "val", y_va, naive.predict(val_df), dt_naive)
    record("1_Naive_District_Median", "test", y_te, naive.predict(test_df), dt_naive)

    # 2. Ridge Log-Linear Baseline (imputed strictly with Train group medians)
    t2 = time.perf_counter()
    X_tr_imp = feat_pipe.impute_with_train_medians(X_tr)
    X_va_imp = feat_pipe.impute_with_train_medians(X_va)
    X_te_imp = feat_pipe.impute_with_train_medians(X_te)

    y_tr_unit = np.log(y_tr / X_tr["area"].to_numpy(dtype="float64"))
    ridge_pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=25.0))])
    ridge_pipe.fit(X_tr_imp, y_tr_unit)
    raw_va_ridge = np.exp(ridge_pipe.predict(X_va_imp)) * X_va["area"].to_numpy(dtype="float64")
    alpha_ridge = tune_multiplicative_mape_factor(y_va, raw_va_ridge)
    dt_ridge = time.perf_counter() - t2

    pred_va_ridge = alpha_ridge * raw_va_ridge
    pred_te_ridge = (
        alpha_ridge
        * np.exp(ridge_pipe.predict(X_te_imp))
        * X_te["area"].to_numpy(dtype="float64")
    )
    record("2_Ridge_LogLinear", "val", y_va, pred_va_ridge, dt_ridge)
    record("2_Ridge_LogLinear", "test", y_te, pred_te_ridge, dt_ridge)

    # 3. Sklearn HistGradientBoostingRegressor (native NaN handling, L1 loss)
    t3 = time.perf_counter()
    hgbr = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=400,
        learning_rate=0.05,
        max_leaf_nodes=63,
        min_samples_leaf=30,
        l2_regularization=3.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    hgbr.fit(X_tr, y_tr_unit)
    raw_va_hgbr = np.exp(hgbr.predict(X_va)) * X_va["area"].to_numpy(dtype="float64")
    alpha_hgbr = tune_multiplicative_mape_factor(y_va, raw_va_hgbr)
    dt_hgbr = time.perf_counter() - t3

    pred_va_hgbr = alpha_hgbr * raw_va_hgbr
    pred_te_hgbr = (
        alpha_hgbr * np.exp(hgbr.predict(X_te)) * X_te["area"].to_numpy(dtype="float64")
    )
    record("3_Sklearn_HistGBR", "val", y_va, pred_va_hgbr, dt_hgbr)
    record("3_Sklearn_HistGBR", "test", y_te, pred_te_hgbr, dt_hgbr)

    # 4. Unified LightGBM
    t4 = time.perf_counter()
    uni_lgbm = UnifiedLightGBMValuationModel(random_state=42).fit(X_tr, y_tr, X_va, y_va)
    dt_uni = time.perf_counter() - t4
    pred_va_uni = uni_lgbm.predict(X_va)
    pred_te_uni = uni_lgbm.predict(X_te)
    record("4_LightGBM_Unified", "val", y_va, pred_va_uni, dt_uni)
    record("4_LightGBM_Unified", "test", y_te, pred_te_uni, dt_uni)

    # 5. Segmented LightGBM (House Expert + Apartment Expert)
    t5 = time.perf_counter()
    seg_lgbm = SegmentedLightGBMValuationModel(random_state=42).fit(X_tr, y_tr, X_va, y_va)
    dt_seg = time.perf_counter() - t5
    pred_va_seg = seg_lgbm.predict(X_va)
    pred_te_seg = seg_lgbm.predict(X_te)
    record("5_LightGBM_Segmented", "val", y_va, pred_va_seg, dt_seg)
    record("5_LightGBM_Segmented", "test", y_te, pred_te_seg, dt_seg)

    # 6. Validation-selected Blend of Segmented + Unified LightGBM
    best_w = 1.0
    best_val_score = float("inf")
    for w in np.linspace(0.0, 1.0, 21):
        cand_va = np.exp(w * np.log(pred_va_seg) + (1.0 - w) * np.log(pred_va_uni))
        m_va = compute_avm_metrics(y_va, cand_va)
        score = 0.5 * m_va["MAPE_pct"] + 0.5 * m_va["MdAPE_pct"]
        if score < best_val_score:
            best_val_score = score
            best_w = float(w)

    bundle = BlendedGBDTValuationBundle(
        feature_pipeline=feat_pipe,
        segmented_model=seg_lgbm,
        unified_model=uni_lgbm,
        w_seg=best_w,
    )
    pred_va_ens = bundle.predict_from_features(X_va)
    pred_te_ens = bundle.predict_from_features(X_te)
    record("6_Champion_LightGBM_Ensemble", "val", y_va, pred_va_ens, dt_uni + dt_seg)
    record("6_Champion_LightGBM_Ensemble", "test", y_te, pred_te_ens, dt_uni + dt_seg)

    # NEW [L3/L6-03]: Train L3 / L6 on the EXISTING imputed Train features.
    # No second call to feat_pipe.fit_transform(): same Train/Val/Test and OOF features.
    course_model_diagnostics: dict[str, Any] = {}
    course_candidate_rows: list[dict[str, Any]] = []
    if include_course_models:
        val_area = X_va["area"].to_numpy(dtype="float64")
        for kind, model_name in [
            ("tree", "7_Decision_Tree_L3"),
            ("mlp", "8_MLP_Regressor_L6"),
        ]:
            selected = _train_course_candidates(
                kind, X_tr_imp, y_tr_unit, X_va_imp, y_va, val_area
            )
            model_bundle = CourseValuationBundle(
                feature_pipeline=feat_pipe,
                regressor=selected["model"],
                alpha=selected["alpha"],
                model_name=model_name,
            )
            # The bundle uses the *same* frozen feature pipeline and Train medians.
            pred_va_course = model_bundle.predict_from_features(X_va)
            pred_te_course = model_bundle.predict_from_features(X_te)
            if not np.allclose(
                pred_va_course, selected["val_pred"], rtol=1e-10, atol=1e-5
            ):
                raise AssertionError(
                    f"{model_name}: saved bundle and direct Val predictions differ"
                )

            record(model_name, "val", y_va, pred_va_course, selected["fit_seconds"])
            record(model_name, "test", y_te, pred_te_course, selected["fit_seconds"])

            # Store additional model artifacts OUTSIDE the champion bundle.
            if course_bundles is not None:
                course_bundles[kind] = model_bundle
            course_candidate_rows.extend(selected["candidate_rows"])
            course_model_diagnostics[model_name] = {
                "alpha": selected["alpha"],
                "selected_config": selected["selected_config"],
                "validation_selection_score": selected["val_score"],
                "candidate_fit_seconds_total": round(selected["fit_seconds"], 2),
                **{
                    key: selected[key]
                    for key in (
                        "tree_depth", "tree_leaves", "mlp_n_iter",
                        "mlp_loss", "mlp_converged_warning",
                    )
                    if key in selected
                },
            }

    comparison_df = pd.DataFrame(results_rows)
    # Compute 95% Bootstrap CIs on untouched Test set for Champion vs Naive Baseline
    champion_test_ci = bootstrap_avm_confidence_intervals(y_te, pred_te_ens, n_boot=1000, seed=42)
    baseline_test_ci = bootstrap_avm_confidence_intervals(
        y_te, naive.predict(test_df), n_boot=1000, seed=42
    )

    diagnostics: dict[str, Any] = {
        "feature_engineering_seconds": round(feat_time, 2),
        "feature_count": int(X_tr.shape[1]),
        "selected_ensemble_weight_segmented": best_w,
        "segmented_best_iterations": {
            "apartment_lgbm": int(seg_lgbm.apt_model_.best_iteration_),
            "house_lgbm": int(seg_lgbm.house_model_.best_iteration_),
            "unified_lgbm": int(uni_lgbm.model_.best_iteration_),
        },
        "multiplicative_alphas": {
            "ridge": round(alpha_ridge, 4),
            "hgbr": round(alpha_hgbr, 4),
            "unified_lgbm": round(uni_lgbm.alpha_, 4),
            "segmented_apt_lgbm": round(seg_lgbm.apt_alpha_, 4),
            "segmented_house_lgbm": round(seg_lgbm.house_alpha_, 4),
        },
        "champion_test_bootstrap_ci95": champion_test_ci,
        "naive_baseline_test_bootstrap_ci95": baseline_test_ci,
        "course_models": course_model_diagnostics,
        "course_model_candidates": course_candidate_rows,
    }
    return bundle, comparison_df, diagnostics