import pandas as pd

file_path = "data/raw/shard_0000.parquet"

df = pd.read_parquet(file_path)

print("Số dòng, số cột: ", df.shape)

print("\nCác cột dữ liệu: ")
print(df.columns.to_list())

print("\n5 bản ghi đầu tiên: ")
print(df.head().to_string(index=False))

print("\nTop 10 tỉnh/thành có nhiều tin nhất:")
print(df["province_name"].value_counts().head(10))

hanoi = df[
    df["province_name"].astype(str).str.strip() == "Hà Nội"
]

print("\nSố tin đăng Hà Nội: ", len(hanoi))

print("\nLoại hình bất động sản tại Hà Nội:")
print(hanoi["property_type_name"].value_counts())