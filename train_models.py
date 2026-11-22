from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.modeling import train_and_evaluate_all_models


def main() -> None:
    gold_dir = Path("data/gold")
    model_path = Path("models/champion_point_bundle.joblib")
    comp_csv_path = Path("reports/model_comparison.csv")
    summary_json_path = Path("reports/model_training_summary.json")

    print("=" * 70)
    print("GIAI DOAN 5 (P0): HUAN LUYEN MO HINH DA TANG & DANH GIA BOOTSTRAP CI")
    print("=" * 70)

    train_df = pd.read_parquet(gold_dir / "train.parquet")
    val_df = pd.read_parquet(gold_dir / "val.parquet")
    test_df = pd.read_parquet(gold_dir / "test.parquet")

    print(
        f"Loaded Gold splits -> Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}"
    )

    bundle, comparison_df, diagnostics = train_and_evaluate_all_models(
        train_df=train_df, val_df=val_df, test_df=test_df
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    comp_csv_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(bundle, model_path)
    comparison_df.to_csv(comp_csv_path, index=False, encoding="utf-8-sig")

    full_summary = {
        "comparison_table": comparison_df.to_dict(orient="records"),
        "diagnostics": diagnostics,
    }
    summary_json_path.write_text(
        json.dumps(full_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== BANG SO SANH TREN TAP VALIDATION ===")
    val_tbl = comparison_df[comparison_df["split"] == "val"].drop(columns=["split"])
    print(val_tbl.round(3).to_string(index=False))

    print("\n=== BANG KET QUA CUOI CUNG TREN TAP TEST (UNTOUCHED HOLDOUT) ===")
    test_tbl = comparison_df[comparison_df["split"] == "test"].drop(columns=["split"])
    print(test_tbl.round(3).to_string(index=False))

    print("\n=== KHOANG TIN CAY 95% BOOTSTRAP (1,000 LAN) CHO CHAMPION MODEL TREN TEST ===")
    for metric, ci in diagnostics["champion_test_bootstrap_ci95"].items():
        print(
            f"  - {metric:12s}: {ci['point']:.3f}  (95% CI: [{ci['ci95_low']:.3f}, {ci['ci95_high']:.3f}])"
        )

    print(f"\nDa luu model bundle : {model_path}")
    print(f"Da luu bang so sanh : {comp_csv_path}")
    print(f"Da luu summary JSON : {summary_json_path}")
    print("HOAN THANH GIAI DOAN 5!")


if __name__ == "__main__":
    main()
