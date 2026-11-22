"""Stage 6 (P1): Mondrian Split Conformalized Quantile Regression (CQR) for 80% Valuation
Intervals [P10, Point, P90], Slice-Based Error Analysis, and Feature Importance Analysis.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.modeling import BlendedGBDTValuationBundle, compute_avm_metrics


class MondrianConformalValuationSystem:
    """Full Production AVM System wrapping:
    - Point valuation bundle (`BlendedGBDTValuationBundle`)
    - Lower (alpha=0.10) & Upper (alpha=0.90) Quantile LightGBM regressors (trained on Train)
    - Mondrian Split Conformal calibration offsets (`q_hat_apt_`, `q_hat_house_`) learned
      strictly on Validation to guarantee ~80% coverage per property type without quantile crossing.
    """

    def __init__(
        self,
        point_bundle: BlendedGBDTValuationBundle,
        target_coverage: float = 0.80,
        random_state: int = 42,
    ) -> None:
        self.point_bundle = point_bundle
        self.target_coverage = target_coverage
        self.random_state = random_state

    def _make_quantile_lgbm(self, alpha: float) -> lgb.LGBMRegressor:
        return lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=1200,
            learning_rate=0.035,
            num_leaves=63,
            min_child_samples=40,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.78,
            reg_alpha=2.0,
            reg_lambda=8.0,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )

    @staticmethod
    def _conformal_quantile(scores: np.ndarray, coverage: float) -> float:
        n = len(scores)
        level = min(1.0, np.ceil((n + 1) * coverage) / n)
        return float(np.quantile(scores, level, method="higher"))

    def fit_and_calibrate(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> "MondrianConformalValuationSystem":
        X_tr = self.point_bundle.feature_pipeline.transform(train_df)
        X_va = self.point_bundle.feature_pipeline.transform(val_df)

        y_tr_vnd = train_df["price"].to_numpy(dtype="float64")
        y_va_vnd = val_df["price"].to_numpy(dtype="float64")

        y_tr_unit = np.log(y_tr_vnd / X_tr["area"].to_numpy(dtype="float64"))
        y_va_unit = np.log(y_va_vnd / X_va["area"].to_numpy(dtype="float64"))

        alpha_lo = (1.0 - self.target_coverage) / 2.0
        alpha_hi = 1.0 - alpha_lo

        self.q_lo_model_ = self._make_quantile_lgbm(alpha=alpha_lo)
        self.q_lo_model_.fit(
            X_tr,
            y_tr_unit,
            eval_X=X_va,
            eval_y=y_va_unit,
            callbacks=[lgb.early_stopping(stopping_rounds=60, verbose=False)],
        )

        self.q_hi_model_ = self._make_quantile_lgbm(alpha=alpha_hi)
        self.q_hi_model_.fit(
            X_tr,
            y_tr_unit,
            eval_X=X_va,
            eval_y=y_va_unit,
            callbacks=[lgb.early_stopping(stopping_rounds=60, verbose=False)],
        )

        # Calibrate Mondrian CQR strictly on Validation
        pt_va_vnd = self.point_bundle.predict_from_features(X_va)
        pt_va_unit = np.log(pt_va_vnd / X_va["area"].to_numpy(dtype="float64"))

        raw_lo_va = np.minimum(self.q_lo_model_.predict(X_va), pt_va_unit)
        raw_hi_va = np.maximum(self.q_hi_model_.predict(X_va), pt_va_unit)

        scores_va = np.maximum(raw_lo_va - y_va_unit, y_va_unit - raw_hi_va)
        va_apt = (X_va["is_apartment"] == 1.0).to_numpy()

        self.q_hat_apt_ = self._conformal_quantile(scores_va[va_apt], self.target_coverage)
        self.q_hat_house_ = self._conformal_quantile(scores_va[~va_apt], self.target_coverage)
        return self

    def predict_with_intervals(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with `pred_price_vnd`, `pred_low_vnd`, `pred_high_vnd`,
        `pred_ppm2_million`, and relative interval half-width."""
        X_feat = self.point_bundle.feature_pipeline.transform(raw_df)
        areas = X_feat["area"].to_numpy(dtype="float64")
        is_apt = (X_feat["is_apartment"] == 1.0).to_numpy()

        pt_vnd = self.point_bundle.predict_from_features(X_feat)
        pt_unit = np.log(pt_vnd / areas)

        raw_lo = np.minimum(self.q_lo_model_.predict(X_feat), pt_unit)
        raw_hi = np.maximum(self.q_hi_model_.predict(X_feat), pt_unit)

        q_adj = np.where(is_apt, self.q_hat_apt_, self.q_hat_house_)
        cal_lo = np.minimum(raw_lo - q_adj, pt_unit - 0.03)
        cal_hi = np.maximum(raw_hi + q_adj, pt_unit + 0.03)

        low_vnd = areas * np.exp(cal_lo)
        high_vnd = areas * np.exp(cal_hi)

        return pd.DataFrame(
            {
                "pred_low_vnd": low_vnd,
                "pred_price_vnd": pt_vnd,
                "pred_high_vnd": high_vnd,
                "pred_ppm2_million": pt_vnd / areas / 1e6,
                "interval_width_pct": (high_vnd - low_vnd) / pt_vnd * 100.0,
            },
            index=raw_df.index,
        )


def compute_interval_diagnostics(
    y_true_vnd: np.ndarray, pred_df: pd.DataFrame
) -> dict[str, float]:
    """Evaluate Prediction Interval Coverage Probability (PICP) and sharpness."""
    y_t = np.asarray(y_true_vnd, dtype="float64")
    lo = pred_df["pred_low_vnd"].to_numpy(dtype="float64")
    pt = pred_df["pred_price_vnd"].to_numpy(dtype="float64")
    hi = pred_df["pred_high_vnd"].to_numpy(dtype="float64")

    covered = (y_t >= lo) & (y_t <= hi)
    below_lo = y_t < lo
    above_hi = y_t > hi
    monotonic_ok = (lo <= pt) & (pt <= hi)

    return {
        "PICP_80_pct": float(np.mean(covered) * 100.0),
        "below_P10_pct": float(np.mean(below_lo) * 100.0),
        "above_P90_pct": float(np.mean(above_hi) * 100.0),
        "median_interval_width_pct": float(np.median((hi - lo) / pt * 100.0)),
        "mean_interval_width_pct": float(np.mean((hi - lo) / pt * 100.0)),
        "monotonic_non_crossing_pct": float(np.mean(monotonic_ok) * 100.0),
    }


def compute_slice_error_analysis(
    test_df: pd.DataFrame, pred_df: pd.DataFrame, feat_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute slice-based error & interval coverage analysis across 4 business dimensions:
    1. `property_type`
    2. `price_tier`
    3. `info_completeness`
    4. `district` (for districts with >= 25 test samples)
    """
    work = pd.DataFrame(index=test_df.index)
    work["y_true"] = test_df["price"].to_numpy(dtype="float64")
    work["y_pred"] = pred_df["pred_price_vnd"].to_numpy(dtype="float64")
    work["y_low"] = pred_df["pred_low_vnd"].to_numpy(dtype="float64")
    work["y_high"] = pred_df["pred_high_vnd"].to_numpy(dtype="float64")
    work["property_type"] = test_df["property_type_name"].astype(str)
    work["district"] = test_df["district_name"].astype(str)

    price_ty = work["y_true"] / 1e9
    work["price_tier"] = pd.cut(
        price_ty,
        bins=[0.0, 3.0, 6.0, 12.0, 25.0, 60.0, float("inf")],
        labels=[
            "1_<3_ty",
            "2_3_to_6_ty",
            "3_6_to_12_ty",
            "4_12_to_25_ty",
            "5_25_to_60_ty",
            "6_>60_ty",
        ],
    ).astype(str)

    # Count non-missing effective structural attributes (out of 5 key physical fields)
    known_phys = (
        feat_df["floor_count_eff"].notna().astype(int)
        + feat_df["frontage_width_eff"].notna().astype(int)
        + feat_df["road_width_eff"].notna().astype(int)
        + feat_df["bedroom_count_eff"].notna().astype(int)
        + feat_df["bathroom_count_eff"].notna().astype(int)
    )
    work["info_completeness"] = np.where(
        known_phys <= 1,
        "1_Low_0_1_attrs",
        np.where(known_phys <= 3, "2_Medium_2_3_attrs", "3_High_4_5_attrs"),
    )

    rows: list[dict[str, Any]] = []

    def summarize_group(dim: str, slice_val: str, g: pd.DataFrame) -> dict[str, Any]:
        m = compute_avm_metrics(g["y_true"].to_numpy(), g["y_pred"].to_numpy())
        cov = np.mean((g["y_true"] >= g["y_low"]) & (g["y_true"] <= g["y_high"])) * 100.0
        w_med = np.median((g["y_high"] - g["y_low"]) / g["y_pred"] * 100.0)
        return {
            "dimension": dim,
            "slice": slice_val,
            "count": int(len(g)),
            "MAPE_pct": round(m["MAPE_pct"], 2),
            "MdAPE_pct": round(m["MdAPE_pct"], 2),
            "PPE15_pct": round(m["PPE15_pct"], 2),
            "PPE20_pct": round(m["PPE20_pct"], 2),
            "MAE_ty": round(m["MAE_ty"], 3),
            "MedAE_ty": round(m["MedAE_ty"], 3),
            "R2_log": round(m["R2_log"], 3),
            "PICP_80_pct": round(float(cov), 2),
            "Med_Width_pct": round(float(w_med), 2),
        }

    for dim in ["property_type", "price_tier", "info_completeness"]:
        for val, g in work.groupby(dim):
            if len(g) >= 5:
                rows.append(summarize_group(dim, str(val), g))

    for dist_name, g in work.groupby("district"):
        if len(g) >= 25:
            rows.append(summarize_group("district_min25", str(dist_name), g))

    out_df = pd.DataFrame(rows)
    return out_df


def extract_feature_importance_table(bundle: BlendedGBDTValuationBundle) -> pd.DataFrame:
    """Extract normalized Gain importance across Unified, House Expert, and Apartment Expert GBDTs."""
    feat_names = bundle.feature_pipeline.feature_names_
    uni_booster = bundle.unified_model.model_.booster_
    house_booster = bundle.segmented_model.house_model_.booster_
    apt_booster = bundle.segmented_model.apt_model_.booster_

    def norm_gain(booster: lgb.Booster) -> np.ndarray:
        g = np.asarray(booster.feature_importance(importance_type="gain"), dtype="float64")
        tot = g.sum()
        return (g / tot * 100.0) if tot > 0 else g

    u_pct = norm_gain(uni_booster)
    h_pct = norm_gain(house_booster)
    a_pct = norm_gain(apt_booster)

    df_imp = pd.DataFrame(
        {
            "feature": feat_names,
            "unified_gain_pct": np.round(u_pct, 3),
            "house_expert_gain_pct": np.round(h_pct, 3),
            "apartment_expert_gain_pct": np.round(a_pct, 3),
        }
    )
    df_imp["combined_rank_score"] = (
        0.5 * df_imp["unified_gain_pct"]
        + 0.35 * df_imp["house_expert_gain_pct"]
        + 0.15 * df_imp["apartment_expert_gain_pct"]
    ).round(3)
    return df_imp.sort_values(by="combined_rank_score", ascending=False).reset_index(drop=True)
