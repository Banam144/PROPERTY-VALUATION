
from pathlib import Path
import pandas as pd


input_path = Path("data/raw/hanoi_housing_raw.parquet")
output_path = Path("data/silver/hanoi_missing_handled.parquet")

df = pd.read_parquet(input_path)

initial_rows = len(df)

print("=" * 60)
print("XU LY MISSING VALUES")
print("=" * 60)

print(f"So ban ghi ban dau: {initial_rows:,}")

text_cols = df.select_dtypes(
    include=["object", "string"]
).columns

for col in text_cols:
    df[col] = df[col].astype("string").str.strip()
    df[col] = df[col].replace("", pd.NA)

df["price"] = pd.to_numeric(
    df["price"], errors="coerce"
)

df["area"] = pd.to_numeric(
    df["area"], errors="coerce"
)

missing_target = (
    df["price"].isna() |
    df["area"].isna()
)

removed_rows = int(missing_target.sum())

df = df.loc[~missing_target].copy()

print(f"\nSo dong thieu price/area: {removed_rows:,}")
print(f"So dong con lai: {len(df):,}")

categorical_cols = [
    "ward_name",
    "street_name",
    "project_name",
    "house_direction",
    "balcony_direction"
]

for col in categorical_cols:
    df[col] = df[col].fillna("Không rõ")

text_features = [
    "name",
    "description"
]

for col in text_features:
    df[col] = df[col].fillna("")

numeric_cols = [
    "floor_count",
    "frontage_width",
    "house_depth",
    "road_width",
    "bedroom_count",
    "bathroom_count"
]

for col in numeric_cols:

    df[col] = pd.to_numeric(
        df[col], errors="coerce"
    )

    df[f"{col}_missing"] = (
        df[col].isna().astype("int8")
    )

print("\n=== MISSING THEO LOAI BAT DONG SAN ===")

check_cols = [
    "floor_count",
    "frontage_width",
    "bedroom_count",
    "bathroom_count"
]

for property_type, group in df.groupby(
    "property_type_name"
):
    print(f"\nLoai: {property_type}")

    missing_rate = (
        group[check_cols]
        .isna()
        .mean()
        .mul(100)
        .round(2)
    )

    print(missing_rate.to_string())

df = df.reset_index(drop=True)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_parquet(output_path, index=False)

print("\n=== KET QUA ===")

print(f"So dong ban dau: {initial_rows:,}")
print(f"So dong bi loai: {removed_rows:,}")
print(f"So dong con lai: {len(df):,}")
print(f"So cot sau xu ly: {df.shape[1]}")

print(f"\nDa luu: {output_path}")
print("HOAN THANH!")
