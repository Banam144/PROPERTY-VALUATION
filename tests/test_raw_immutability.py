"""Automated verification that Raw layer datasets are strictly preserved and never overwritten."""

from __future__ import annotations

from pathlib import Path
import pyarrow.parquet as pq


def test_raw_files_exist_and_preserve_exact_row_counts() -> None:
    shard_path = Path("data/raw/shard_0000.parquet")
    hanoi_raw_path = Path("data/raw/hanoi_housing_raw.parquet")

    assert shard_path.exists(), "Raw shard_0000.parquet is missing!"
    assert hanoi_raw_path.exists(), "Raw hanoi_housing_raw.parquet is missing!"

    shard_meta = pq.read_metadata(shard_path)
    hanoi_meta = pq.read_metadata(hanoi_raw_path)

    assert shard_meta.num_rows == 350_075
    assert shard_meta.num_columns == 19

    assert hanoi_meta.num_rows == 93_940
    assert hanoi_meta.num_columns == 19
