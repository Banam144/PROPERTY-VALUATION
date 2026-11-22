"""Unit & integration tests for Stage 3 (P0) data cleaning & repost deduplication."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_cleaning import (
    filter_domain_outliers,
    normalize_direction,
    normalize_street_name,
    run_cleaning_pipeline,
    sanitize_physical_attributes,
)


def test_normalize_direction_merges_hyphenated_variants() -> None:
    s = pd.Series(["Đông - Nam", "Tây - Bắc", "đông bắc", "Không rõ", None, "Invalid"])
    out = normalize_direction(s)
    assert out.tolist() == [
        "Đông Nam",
        "Tây Bắc",
        "Đông Bắc",
        "Không rõ",
        "Không rõ",
        "Không rõ",
    ]


def test_normalize_street_name_preserves_numbered_routes() -> None:
    s = pd.Series(["Phố Thái Hà", "Đường Cầu Giấy", "Đường 800A", "Đường 70", "  ", None])
    out = normalize_street_name(s)
    assert out.tolist() == [
        "Thái Hà",
        "Cầu Giấy",
        "Đường 800A",
        "Đường 70",
        "Không rõ",
        "Không rõ",
    ]


def test_filter_domain_outliers_removes_unit_errors_and_rentals_without_leaking_ppm2() -> None:
    df = pd.DataFrame(
        {
            "name": [
                "Bán nhà Cầu Giấy 50m2 5 tầng đẹp",
                "Cho thuê căn hộ 2PN full đồ giá rẻ",
                "Bán nhà nhập nhầm đơn vị 1000x giá 10.5 tỷ",
                "Bán nhà nhân đôi diện tích 580 tỷ",
                "Bán căn hộ 80m2 chuẩn giá",
            ],
            "property_type_name": ["Nhà", "Căn hộ chung cư", "Nhà", "Nhà", "Căn hộ chung cư"],
            "price": [10_500_000_000, 12_000_000, 10_500_000, 580_000_000_000, 5_600_000_000],
            "area": [50.0, 70.0, 50.0, 60.0, 80.0],
        }
    )
    kept, rejected, counts = filter_domain_outliers(df)
    assert len(kept) == 2
    assert len(rejected) == 3
    # Crucial check: temporary ppm2 must NOT leak into kept dataframe
    assert "ppm2" not in kept.columns
    assert "ppm2_million" not in kept.columns
    assert sum(counts.values()) == 3


def test_sanitize_physical_attributes_keeps_rows_and_updates_missing_flags() -> None:
    df = pd.DataFrame(
        {
            "property_type_name": ["Nhà", "Căn hộ chung cư"],
            "area": [40.0, 75.0],
            "floor_count": [5.0, 60.0],  # 60 floors invalid for apt
            "frontage_width": [42.0, np.nan],  # 42m frontage on 40m2 house -> implied depth < 1.5m
            "house_depth": [10.0, np.nan],
            "road_width": [4.0, 200.0],  # 200m road invalid
            "bedroom_count": [4.0, 15.0],  # 15 bedrooms invalid for apt
            "bathroom_count": [3.0, 2.0],
        }
    )
    sanitized, counts = sanitize_physical_attributes(df)
    assert len(sanitized) == 2  # No rows dropped!
    assert np.isnan(sanitized.loc[0, "frontage_width"])
    assert sanitized.loc[0, "frontage_width_missing"] == 1
    assert np.isnan(sanitized.loc[1, "floor_count"])
    assert sanitized.loc[1, "floor_count_missing"] == 1
    assert np.isnan(sanitized.loc[1, "road_width"])
    assert np.isnan(sanitized.loc[1, "bedroom_count"])
    assert counts["frontage_width"] == 1


def test_run_cleaning_pipeline_deduplicates_phone_stripped_repost() -> None:
    df = pd.DataFrame(
        {
            "name": [
                "Nhà 5 tầng mới Thượng Cát Bắc Từ Liêm 40m có thang máy",
                "Nhà 5 tầng mới Thượng Cát Bắc Từ Liêm 40m có thang máy",
            ],
            "description": [
                "Nhà 5 tầng mới đẹp sổ đỏ vuông vắn giao thông thuận tiện gọi ngay [phone_number]",
                "Nhà 5 tầng mới đẹp sổ đỏ vuông vắn giao thông thuận tiện gọi ngay",
            ],
            "property_type_name": ["Nhà", "Nhà"],
            "province_name": ["Hà Nội", "Hà Nội"],
            "district_name": ["Bắc Từ Liêm", "Bắc Từ Liêm"],
            "ward_name": ["Thượng Cát", "Thượng Cát"],
            "street_name": ["Phố Thượng Cát", "Thượng Cát"],
            "project_name": ["Không rõ", "Không rõ"],
            "price": [5_400_000_000, 5_400_000_000],
            "area": [40.0, 40.0],
            "floor_count": [5.0, 5.0],
            "frontage_width": [4.0, 4.0],
            "house_depth": [10.0, 10.0],
            "road_width": [3.5, 3.5],
            "bedroom_count": [4.0, 4.0],
            "bathroom_count": [4.0, 4.0],
            "house_direction": ["Đông - Nam", "Đông Nam"],
            "balcony_direction": ["Không rõ", "Không rõ"],
            "published_at": ["2025-06-26T06:51:17.262000", "2025-06-26T02:00:20"],
        }
    )
    cleaned, rejected, summary = run_cleaning_pipeline(df)
    assert len(cleaned) == 1
    assert summary["reposts_removed"] == 1
    # Kept the earliest timestamp (02:00:20)
    assert cleaned.loc[0, "published_at"].startswith("2025-06-26T02:00:20")
    assert cleaned.loc[0, "house_direction"] == "Đông Nam"
    assert cleaned.loc[0, "street_name"] == "Thượng Cát"
    assert "property_group_id" in cleaned.columns
