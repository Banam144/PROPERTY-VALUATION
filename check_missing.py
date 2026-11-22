import pandas as pd

file_path = "data/raw/hanoi_housing_raw.parquet"

df = pd.read_parquet(file_path)

print("=" * 60)
print("PHAN TICH MISSING VALUES")
print("=" * 60)

print(f"\nTong so ban ghi: {len(df):,}")
print(f"Tong so cot: {len(df.columns)}")

missing_mask = df.isna()

for col in df.select_dtypes(
    include=["object", "string"]
).columns:
    empty_mask = (
        df[col]
        .astype("string")
        .str.strip()
        .eq("")
        .fillna(False)
    )
    missing_mask[col] = missing_mask[col] | empty_mask

missing_count = missing_mask.sum()

missing_percent = (
    missing_count / len(df) * 100
).round(2)

missing_report = pd.DataFrame({
    "Missing_Count": missing_count,
    "Missing_Percent": missing_percent
})

missing_report = missing_report.sort_values(
    by="Missing_Percent",
    ascending=False
)

print("\n=== TY LE THIEU THEO TUNG COT ===")
print(missing_report.to_string())

important_columns = [
    "price",
    "area",
    "district_name",
    "ward_name",
    "property_type_name",
    "published_at"
]

print("\n=== CAC COT QUAN TRONG ===")

for col in important_columns:
    count = missing_report.loc[col, "Missing_Count"]
    percent = missing_report.loc[col, "Missing_Percent"]

    print(
        f"{col}: {count:,} gia tri thieu "
        f"({percent:.2f}%)"
    )

print("\nHOAN THANH KIEM TRA MISSING VALUES")
