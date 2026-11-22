from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.features import HanoiValuationFeaturePipeline
from src.splitting import purged_chronological_split


def main() -> None:
    input_path = Path("data/silver/hanoi_cleaned.parquet")
    gold_dir = Path("data/gold")
    report_path = Path("reports/split_summary.json")

    print("=" * 65)
    print("GIAI DOAN 4 (P0): CHIA TAP CHRONOLOGICAL PURGED & KIEM CHUNG FEATURE")
    print("=" * 65)

    df = pd.read_parquet(input_path)
    print(f"Doc du lieu sach: {input_path} ({len(df):,} dong, {df.shape[1]} cot)")

    train_df, val_df, test_df, summary = purged_chronological_split(
        df, train_frac=0.70, val_frac=0.15
    )

    gold_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(gold_dir / "train.parquet", index=False)
    val_df.to_parquet(gold_dir / "val.parquet", index=False)
    test_df.to_parquet(gold_dir / "test.parquet", index=False)

    # Dry-run feature pipeline strictly fitted on Train to record feature recovery diagnostics
    pipe = HanoiValuationFeaturePipeline(n_splits=5, random_state=42)
    X_tr = pipe.fit_transform(train_df, train_df["price"])
    X_va = pipe.transform(val_df)
    X_te = pipe.transform(test_df)

    nlp_backfill_stats = {
        "train_floor_recovered_from_text": int(X_tr["floor_from_text"].sum()),
        "train_frontage_recovered_from_text": int(X_tr["frontage_from_text"].sum()),
        "train_road_recovered_from_text": int(X_tr["road_from_text"].sum()),
        "train_bedroom_recovered_from_text": int(X_tr["bedroom_from_text"].sum()),
        "total_engineered_features": int(X_tr.shape[1]),
        "val_shape": list(X_va.shape),
        "test_shape": list(X_te.shape),
    }
    summary["feature_engineering_diagnostics"] = nlp_backfill_stats

    report_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n--- 1. KET QUA CHIA TAP PURGED CHRONOLOGICAL ---")
    print(f"Train : {summary['train_rows']:,} dong | {summary['train_period'][0]} -> {summary['train_period'][1]}")
    print(
        f"Val   : {summary['val_rows']:,} dong (purged {summary['purged_from_val']:,} repost) "
        f"| {summary['val_period'][0]} -> {summary['val_period'][1]}"
    )
    print(
        f"Test  : {summary['test_rows']:,} dong (purged {summary['purged_from_test']:,} repost) "
        f"| {summary['test_period'][0]} -> {summary['test_period'][1]}"
    )

    print("\n--- 2. THONG KE KHOI PHUC DAC TRUNG TU VAN BAN (TRAIN) ---")
    for k, v in nlp_backfill_stats.items():
        print(f"  - {k}: {v}")

    print(f"\nDa luu Gold splits tai: {gold_dir}")
    print(f"Da luu bao cao tai    : {report_path}")
    print("HOAN THANH GIAI DOAN 4!")


if __name__ == "__main__":
    main()
