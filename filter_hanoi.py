from pathlib import Path
import pandas as pd

input_path = Path("data/raw/shard_0000.parquet")
output_path = Path("data/raw/hanoi_housing_raw.parquet")

df = pd.read_parquet(input_path)

print("Tong so ban ghi:", len(df))

hanoi = df[
    df["province_name"].fillna("").str.strip() == "Hà Nội"
].copy()

print("So ban ghi Ha Noi:", len(hanoi))

property_types = ["Nhà", "Căn hộ chung cư"]

housing = hanoi[
    hanoi["property_type_name"]
    .fillna("")
    .str.strip()
    .isin(property_types)
].copy()

housing.reset_index(drop=True)

print("\n=== KET QUA LOC DU LIEU ===")
print("So ban ghi con lai:", len(housing))

print("\nLoai bat dong san:")
print(housing["property_type_name"].value_counts())

output_path.parent.mkdir(parents=True, exist_ok=True)
housing.to_parquet(output_path, index=False)

print("\nDa luu du lieu tai:", output_path)
print("Hoan thanh!")