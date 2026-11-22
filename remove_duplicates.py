
from pathlib import Path
import pandas as pd


input_path = Path(
    "data/silver/hanoi_missing_handled.parquet"
)

output_path = Path(
    "data/silver/hanoi_deduplicated.parquet"
)

report_path = Path(
    "reports/removed_duplicates.csv"
)

df = pd.read_parquet(input_path)

initial_rows = len(df)

print("=" * 60)
print("XU LY DUPLICATE - HANOI HOUSING")
print("=" * 60)

print(f"So ban ghi ban dau: {initial_rows:,}")



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

print(
    "So tin du dieu kien so sanh noi dung:",
    int(valid_text.sum())
)


original_cols = [
    col for col in df.columns
    if not col.endswith("_missing")
    and not col.startswith("_")
]

compare_cols = [
    col for col in original_cols
    if col not in [
        "published_at",
        "name",
        "description"
    ]
]

compare_cols += [
    "_name_norm",
    "_desc_norm"
]

df["_date_sort"] = pd.to_datetime(
    df["published_at"],
    errors="coerce",
    utc=True
)

df = df.sort_values(
    by="_date_sort",
    kind="stable",
    na_position="last"
)


duplicate_mask = pd.Series(
    False,
    index=df.index
)


eligible = df.loc[valid_text]

duplicate_mask.loc[eligible.index] = (
    eligible.duplicated(
        subset=compare_cols,
        keep="first"
    )
)

removed = df.loc[duplicate_mask].copy()
cleaned = df.loc[~duplicate_mask].copy()

print("\n=== KET QUA XU LY ===")

print(f"So ban ghi bi loai: {len(removed):,}")
print(f"So ban ghi giu lai: {len(cleaned):,}")



report_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

report_cols = [
    "property_type_name",
    "district_name",
    "name",
    "price",
    "area",
    "published_at"
]

removed[report_cols].to_csv(
    report_path,
    index=False,
    encoding="utf-8-sig"
)



helper_cols = [
    "_name_norm",
    "_desc_norm",
    "_date_sort"
]

cleaned = cleaned.drop(columns=helper_cols)


cleaned = cleaned.sort_index().reset_index(
    drop=True
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

cleaned.to_parquet(
    output_path,
    index=False
)


assert len(cleaned) + len(removed) == initial_rows

print("\n=== TONG KET ===")
print(f"Ban dau: {initial_rows:,}")
print(f"Loai bo: {len(removed):,}")
print(f"Con lai: {len(cleaned):,}")

print(f"\nDataset: {output_path}")
print(f"Bao cao: {report_path}")

print("\nHOAN THANH DEDUPLICATION!")
