"""Stage 6 (P1) Runner: Train & calibrate Mondrian Split Conformalized Quantile Regression
(80% intervals [P10, Point, P90]), run Slice-Based Error Analysis, and export Feature Importance.

Input : data/gold/{train,val,test}.parquet, models/champion_point_bundle.joblib
Output: models/avm_conformal_system.joblib
Report: reports/slice_error_analysis.csv, reports/feature_importance.csv,
        reports/conformal_and_slice_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.intervals_and_analysis import (
    MondrianConformalValuationSystem,
    compute_interval_diagnostics,
    compute_slice_error_analysis,
    extract_feature_importance_table,
)


def main() -> None:
    gold_dir = Path("data/gold")
    point_bundle_path = Path("models/champion_point_bundle_integrated.joblib")
    full_system_path = Path("models/avm_conformal_system.joblib")
    slice_csv_path = Path("reports/slice_error_analysis.csv")
    imp_csv_path = Path("reports/feature_importance.csv")
    summary_json_path = Path("reports/conformal_and_slice_summary.json")

    print("=" * 70)
    print("GIAI DOAN 6 (P1): CONFORMAL QUANTILE [P10, P90] & SLICE ERROR ANALYSIS")
    print("=" * 70)

    train_df = pd.read_parquet(gold_dir / "train.parquet")
    val_df = pd.read_parquet(gold_dir / "val.parquet")
    test_df = pd.read_parquet(gold_dir / "test.parquet")
    point_bundle = joblib.load(point_bundle_path)

    system = MondrianConformalValuationSystem(
        point_bundle=point_bundle, target_coverage=0.80, random_state=42
    )
    system.fit_and_calibrate(train_df=train_df, val_df=val_df)

    val_preds = system.predict_with_intervals(val_df)
    test_preds = system.predict_with_intervals(test_df)

    val_int_diag = compute_interval_diagnostics(val_df["price"].to_numpy(), val_preds)
    test_int_diag = compute_interval_diagnostics(test_df["price"].to_numpy(), test_preds)

    X_te_feat = point_bundle.feature_pipeline.transform(test_df)
    slice_df = compute_slice_error_analysis(test_df, test_preds, X_te_feat)
    imp_df = extract_feature_importance_table(point_bundle)

    full_system_path.parent.mkdir(parents=True, exist_ok=True)
    slice_csv_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(system, full_system_path)
    slice_df.to_csv(slice_csv_path, index=False, encoding="utf-8-sig")
    imp_df.to_csv(imp_csv_path, index=False, encoding="utf-8-sig")

    summary = {
        "conformal_calibration_offsets_log_scale": {
            "q_hat_apartment": round(system.q_hat_apt_, 4),
            "q_hat_house": round(system.q_hat_house_, 4),
        },
        "val_interval_diagnostics": val_int_diag,
        "test_interval_diagnostics": test_int_diag,
        "top_20_features": imp_df.head(20).to_dict(orient="records"),
    }
    summary_json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== 1. DO PHU KHOANG TIN CAY 80% CQR TREN TEST ===")
    for k, v in test_int_diag.items():
        print(f"  - {k:28s}: {v:.2f}")

    print("\n=== 2. SLICE ERROR ANALYSIS (LOAI HINH, PHAN KHUC GIA, DO DAY DU TT) ===")
    key_slices = slice_df[slice_df["dimension"] != "district_min25"]
    print(key_slices.to_string(index=False))

    print("\n=== 3. TOP 15 DAC TRUNG QUAN TRONG NHAT (% GAIN) ===")
    print(imp_df.head(15).to_string(index=False))

    print(f"\nDa luu he thong AVM day du : {full_system_path}")
    print(f"Da luu Slice Analysis CSV  : {slice_csv_path}")
    print(f"Da luu Feature Importance  : {imp_csv_path}")
    print("HOAN THANH GIAI DOAN 6!")


if __name__ == "__main__":
    main()
