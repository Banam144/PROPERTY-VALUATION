from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data_cleaning import run_cleaning_pipeline


def main() -> None:
    input_path = Path("data/silver/hanoi_deduplicated.parquet")
    output_path = Path("data/silver/hanoi_cleaned.parquet")
    report_csv_path = Path("reports/outlier_and_repost_report.csv")
    summary_json_path = Path("reports/cleaning_summary.json")

    print("=" * 65)
    print("GIAI DOAN 3 (P0): LAM SACH CHUYEN SAU & KHU REPOST - HANOI AVM")
    print("=" * 65)

    df = pd.read_parquet(input_path)
    print(f"Doc du lieu dau vao: {input_path} ({len(df):,} dong, {df.shape[1]} cot)")

    cleaned_df, rejected_report, summary = run_cleaning_pipeline(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_csv_path.parent.mkdir(parents=True, exist_ok=True)

    cleaned_df.to_parquet(output_path, index=False)
    rejected_report.to_csv(report_csv_path, index=False, encoding="utf-8-sig")
    summary_json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n--- 1. CHUAN HOA THUC THE ---")
    print("Truoc:", summary["entity_normalization_before"])
    print("Sau  :", summary["entity_normalization_after"])

    print("\n--- 2. LOAI BO NGOAI LAI & TIN THUE / SAI DON VI ---")
    print(f"Tong so dong ngoai lai bi loai: {summary['outliers_removed']:,}")
    for reason, cnt in summary["outlier_reasons"].items():
        print(f"  - {reason}: {cnt:,}")

    print("\n--- 3. LAM SACH THUOC TINH VAT LY BAT HOP LY (-> NaN) ---")
    for col, cnt in summary["physical_values_sanitized_to_nan"].items():
        print(f"  - {col}: {cnt:,} gia tri bat hop ly chuyen ve NaN")

    print("\n--- 4. KHU TIN DANG LAI (BROKER NEAR-REPOST) ---")
    print(f"Tong so tin repost bi loai: {summary['reposts_removed']:,}")
    for reason, cnt in summary["repost_reasons"].items():
        print(f"  - {reason}: {cnt:,}")

    print("\n--- 5. TONG KET DAU RA ---")
    print(f"So dong cuoi cung : {summary['final_rows']:,} ({summary['final_columns']} cot)")
    print(f"So nhom tai san   : {summary['unique_property_groups']:,} property_group_id")
    print(f"Phan bo loai BĐS  : {summary['property_type_counts']}")
    print(f"Da luu dataset    : {output_path}")
    print(f"Da luu bao cao    : {report_csv_path}")
    print(f"Da luu summary    : {summary_json_path}")
    print("HOAN THANH GIAI DOAN 3!")


if __name__ == "__main__":
    main()
