from pathlib import Path
import pandas as pd

input_path = Path(
    "data/silver/hanoi_missing_handled.parquet"
)

df = pd.read_parquet(input_path)

print("=" * 60)
print("KIEM TRA DUPLICATE - HANOI HOUSING")
print("=" * 60)

print(f"\nTong so ban ghi: {len(df):,}")


original_cols = [
    col for col in df.columns
    if not col.endswith("_missing")
]

exact_duplicate = df.duplicated(
    subset=original_cols,
    keep="first"
)

exact_count = int(exact_duplicate.sum())

print("\n=== EXACT DUPLICATE ===")
print(f"So ban ghi trung hoan toan: {exact_count:,}")

def normalize_text(series):
    return (
        series.astype("string")
        .fillna("")
        .str.casefold()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


df["_name_norm"] = normalize_text(df["name"])
df["_desc_norm"] = normalize_text(df["description"])

valid_text = (
    (df["_name_norm"].str.len() >= 20)
    & (df["_desc_norm"].str.len() >= 40)
)

candidates = df.loc[valid_text].copy()

duplicate_keys = [
    "property_type_name",
    "district_name",
    "_name_norm",
    "_desc_norm"
]

potential_duplicate = candidates.duplicated(
    subset=duplicate_keys,
    keep=False
)

potential_extra = candidates.duplicated(
    subset=duplicate_keys,
    keep="first"
)

print("\n=== POTENTIAL DUPLICATE ===")
print(
    "So ban ghi nam trong nhom nghi trung:",
    int(potential_duplicate.sum())
)

print(
    "So ban ghi trung them neu giu 1 tin/nhom:",
    int(potential_extra.sum())
)

groups = (
    candidates.groupby(duplicate_keys, dropna=False)
    .agg(
        so_tin=("price", "size"),
        gia_min=("price", "min"),
        gia_max=("price", "max"),
        dien_tich_min=("area", "min"),
        dien_tich_max=("area", "max")
    )
    .reset_index()
)

groups = groups[groups["so_tin"] >= 2].copy()

groups = groups.sort_values(
    by="so_tin",
    ascending=False
)

groups["ten_tin"] = (
    groups["_name_norm"].str.slice(0, 100)
)

report_cols = [
    "property_type_name",
    "district_name",
    "ten_tin",
    "so_tin",
    "gia_min",
    "gia_max",
    "dien_tich_min",
    "dien_tich_max"
]

report = groups[report_cols]

print("\n=== TOP 10 NHOM NGHI TRUNG ===")
print(report.head(10).to_string(index=False))

output_path = Path(
    "reports/duplicate_candidates.csv"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

report.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\n=== TONG KET ===")
print(f"Tong so ban ghi: {len(df):,}")
print(f"Exact Duplicate: {exact_count:,}")
print(f"So nhom nghi trung: {len(groups):,}")
print(f"Bao cao: {output_path}")

print("\nHOAN THANH KIEM TRA DUPLICATE!")
