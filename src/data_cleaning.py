from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np
import pandas as pd

VALID_DIRECTIONS = {
    "Đông",
    "Tây",
    "Nam",
    "Bắc",
    "Đông Nam",
    "Đông Bắc",
    "Tây Nam",
    "Tây Bắc",
}

NUMERIC_OPTIONAL_COLS = [
    "floor_count",
    "frontage_width",
    "house_depth",
    "road_width",
    "bedroom_count",
    "bathroom_count",
]


def normalize_direction(series: pd.Series) -> pd.Series:
    """Normalize compass direction strings (e.g. 'Đông - Nam' -> 'Đông Nam')."""
    s = (
        series.astype("string")
        .fillna("Không rõ")
        .str.strip()
        .str.replace(r"\s*-\s*", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
    )
    
    mapping = {d.casefold(): d for d in VALID_DIRECTIONS}
    lowered = s.str.casefold()
    normalized = lowered.map(mapping).fillna("Không rõ").astype("string")
    return normalized


def normalize_street_name(series: pd.Series) -> pd.Series:
    s = (
        series.astype("string")
        .fillna("Không rõ")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
   
    s = s.str.replace(r"^(?:Đường|Phố|đường|phố)\s+(?![0-9])", "", regex=True).str.strip()
    s = s.replace("", "Không rõ")
    return s.astype("string")


def parse_published_at(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, format="ISO8601", utc=True, errors="coerce")
    return dt


def clean_text_for_dedup(series: pd.Series) -> pd.Series:
    """Normalize listing text for repost detection, stripping masked phone placeholders
    and punctuation noise."""
    return (
        series.astype("string")
        .fillna("")
        .str.casefold()
        .str.replace(r"\[phone_number\]", " ", regex=True)
        .str.replace(r"\b0\d{8,10}\b", " ", regex=True)
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def strip_price_tokens_for_group_key(series: pd.Series) -> pd.Series:
    """Create a price-agnostic title signature so reposts with slight price edits
    still map to the same `property_group_id`."""
    return (
        series.astype("string")
        .fillna("")
        .str.casefold()
        .str.replace(r"\[phone_number\]", " ", regex=True)
        .str.replace(
            r"\d+(?:[.,]\d+)?\s*(?:tỷ|ty|tỉ|triệu|tr|nghìn|ngàn)(?:\s*/\s*m2?)?",
            " ",
            regex=True,
        )
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def filter_domain_outliers(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    
    work = df.copy()
    price_f = work["price"].astype(float)
    area_f = work["area"].astype(float)
    ppm2 = price_f / np.where(area_f > 0, area_f, np.nan) / 1e6

    is_apt = work["property_type_name"] == "Căn hộ chung cư"
    is_house = work["property_type_name"] == "Nhà"

    # Rental title prefix (pure rental ads accidentally categorized into sale)
    rental_prefix = (
        work["name"]
        .astype("string")
        .fillna("")
        .str.strip()
        .str.contains(
            r"^(?:cho thuê|cần cho thuê|cần thuê|nhượng phòng)\b",
            case=False,
            regex=True,
        )
    )

    reason = pd.Series("", index=work.index, dtype="string")

    def mark(mask: pd.Series, label: str) -> None:
        nonlocal reason
        target = mask.fillna(False) & (reason == "")
        reason = reason.mask(target, label)

    mark((price_f <= 0) | (area_f <= 0) | price_f.isna() | area_f.isna(), "non_positive_price_or_area")
    mark(rental_prefix, "rental_title_misclassified")
    mark(ppm2 < 15.0, "unit_error_or_rental_ppm2_lt_15m")

    # Apartment rules
    mark(is_apt & ((area_f < 18.0) | (area_f > 500.0)), "apartment_area_out_of_bounds")
    mark(
        is_apt & ((price_f < 800_000_000) | (price_f > 100_000_000_000)),
        "apartment_price_out_of_bounds",
    )
    mark(is_apt & ((ppm2 < 18.0) | (ppm2 > 350.0)), "apartment_ppm2_out_of_bounds")

    # House rules
    mark(is_house & ((area_f < 12.0) | (area_f > 1_000.0)), "house_area_out_of_bounds")
    mark(
        is_house & ((price_f < 800_000_000) | (price_f > 600_000_000_000)),
        "house_price_out_of_bounds",
    )
    mark(is_house & ((ppm2 < 15.0) | (ppm2 > 2_500.0)), "house_ppm2_out_of_bounds")

    outlier_mask = reason != ""
    outliers_df = work.loc[outlier_mask].copy()
    outliers_df["reject_reason"] = reason.loc[outlier_mask]
    outliers_df["ppm2_million"] = ppm2.loc[outlier_mask].round(2)

    kept_df = work.loc[~outlier_mask].copy()
    reason_counts = {
        str(k): int(v) for k, v in outliers_df["reject_reason"].value_counts().items()
    }
    return kept_df, outliers_df, reason_counts


def sanitize_physical_attributes(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:

    out = df.copy()
    is_apt = out["property_type_name"] == "Căn hộ chung cư"
    is_house = out["property_type_name"] == "Nhà"
    area_f = out["area"].astype(float)

    sanitized_counts: dict[str, int] = {}

    for col in NUMERIC_OPTIONAL_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    # 1. floor_count
    fc = out["floor_count"]
    bad_fc = (fc < 1) | (is_house & (fc > 25)) | (is_apt & (fc > 50))
    sanitized_counts["floor_count"] = int(bad_fc.fillna(False).sum())
    out.loc[bad_fc.fillna(False), "floor_count"] = np.nan

    # 2. frontage_width: [1.0, 50.0] and implied depth (area / frontage) >= 1.5m for houses
    fw = out["frontage_width"]
    implied_depth = area_f / fw
    bad_fw = (fw < 1.0) | (fw > 50.0) | (is_house & (implied_depth < 1.5))
    sanitized_counts["frontage_width"] = int(bad_fw.fillna(False).sum())
    out.loc[bad_fw.fillna(False), "frontage_width"] = np.nan

    # 3. house_depth: [1.5, 80.0]
    hd = out["house_depth"]
    bad_hd = (hd < 1.5) | (hd > 80.0)
    sanitized_counts["house_depth"] = int(bad_hd.fillna(False).sum())
    out.loc[bad_hd.fillna(False), "house_depth"] = np.nan

    # 4. road_width: [0.8, 60.0]
    rw = out["road_width"]
    bad_rw = (rw < 0.8) | (rw > 60.0)
    sanitized_counts["road_width"] = int(bad_rw.fillna(False).sum())
    out.loc[bad_rw.fillna(False), "road_width"] = np.nan

    # 5. bedroom_count
    bed = out["bedroom_count"]
    bad_bed = (bed < 1) | (is_house & (bed > 35)) | (is_apt & (bed > 8))
    sanitized_counts["bedroom_count"] = int(bad_bed.fillna(False).sum())
    out.loc[bad_bed.fillna(False), "bedroom_count"] = np.nan

    # 6. bathroom_count
    bath = out["bathroom_count"]
    bad_bath = (bath < 1) | (is_house & (bath > 35)) | (is_apt & (bath > 8))
    sanitized_counts["bathroom_count"] = int(bad_bath.fillna(False).sum())
    out.loc[bad_bath.fillna(False), "bathroom_count"] = np.nan

    # Refresh missingness flags strictly from current NaNs
    for col in NUMERIC_OPTIONAL_COLS:
        out[f"{col}_missing"] = out[col].isna().astype("int8")

    return out, sanitized_counts


def deduplicate_reposts_and_assign_group_id(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    work = df.copy()
    work["_pub_dt"] = parse_published_at(work["published_at"])
    work = work.sort_values(by="_pub_dt", kind="stable", na_position="last")

    work["_name_clean"] = clean_text_for_dedup(work["name"])
    work["_desc_clean"] = clean_text_for_dedup(work["description"])
    work["_area_r1"] = work["area"].astype(float).round(1)
    work["_price_r10m"] = (work["price"].astype(float) / 1e7).round(0)

    # Rule 1: Same (property_type, district, ward, rounded_area, rounded_price, cleaned_name)
    # when cleaned_name has >= 15 chars
    elig_name = work["_name_clean"].str.len() >= 15
    dup_name = pd.Series(False, index=work.index)
    dup_name.loc[elig_name] = work.loc[elig_name].duplicated(
        subset=[
            "property_type_name",
            "district_name",
            "ward_name",
            "_area_r1",
            "_price_r10m",
            "_name_clean",
        ],
        keep="first",
    )

    # Rule 2: Same (property_type, district, rounded_area, cleaned_description)
    # when cleaned_description has >= 50 chars (even if broker slightly tweaked title/price!)
    elig_desc = (~dup_name) & (work["_desc_clean"].str.len() >= 50)
    dup_desc = pd.Series(False, index=work.index)
    dup_desc.loc[elig_desc] = work.loc[elig_desc].duplicated(
        subset=[
            "property_type_name",
            "district_name",
            "_area_r1",
            "_desc_clean",
        ],
        keep="first",
    )

    repost_mask = dup_name | dup_desc
    removed_reposts = work.loc[repost_mask].copy()
    removed_reposts["reject_reason"] = np.where(
        dup_name.loc[repost_mask], "repost_same_title_key", "repost_same_description_key"
    )

    cleaned = work.loc[~repost_mask].copy()

    title_sig = strip_price_tokens_for_group_key(cleaned["name"]).str.slice(0, 40)
    raw_key = (
        cleaned["property_type_name"].astype(str)
        + "|"
        + cleaned["district_name"].astype(str)
        + "|"
        + cleaned["ward_name"].astype(str)
        + "|"
        + cleaned["_area_r1"].astype(str)
        + "|"
        + title_sig.astype(str)
    )
    cleaned["property_group_id"] = [
        hashlib.sha1(k.encode("utf-8")).hexdigest()[:16] for k in raw_key.tolist()
    ]
    cleaned["property_group_id"] = cleaned["property_group_id"].astype("string")

    helper_cols = ["_pub_dt", "_name_clean", "_desc_clean", "_area_r1", "_price_r10m"]
    cleaned = cleaned.drop(columns=helper_cols).sort_index().reset_index(drop=True)
    removed_reposts = removed_reposts.drop(columns=helper_cols)
    return cleaned, removed_reposts


def run_cleaning_pipeline(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Execute the full Stage 3 cleaning pipeline on `hanoi_deduplicated` DataFrame."""
    initial_rows = len(df)
    work = df.copy()

    # 1. Entity normalization
    dirs_before = {
        "house_direction_unique": int(work["house_direction"].nunique()),
        "balcony_direction_unique": int(work["balcony_direction"].nunique()),
        "street_name_unique": int(work["street_name"].nunique()),
    }
    work["house_direction"] = normalize_direction(work["house_direction"])
    work["balcony_direction"] = normalize_direction(work["balcony_direction"])
    work["street_name"] = normalize_street_name(work["street_name"])
    # Standardize published_at string to canonical ISO-8601 UTC string
    pub_dt = parse_published_at(work["published_at"])
    if pub_dt.isna().any():
        raise ValueError(f"Found {int(pub_dt.isna().sum())} unparseable timestamps in published_at")
    work["published_at"] = pub_dt.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ").astype("string")

    dirs_after = {
        "house_direction_unique": int(work["house_direction"].nunique()),
        "balcony_direction_unique": int(work["balcony_direction"].nunique()),
        "street_name_unique": int(work["street_name"].nunique()),
    }

    # 2. Domain outlier filtering
    work, outliers_df, outlier_reasons = filter_domain_outliers(work)
    after_outlier_rows = len(work)

    # 3. Physical attribute sanitization (in-place to NaN + update *_missing)
    work, sanitized_counts = sanitize_physical_attributes(work)

    # 4. Near-duplicate repost removal & property_group_id assignment
    work, reposts_df = deduplicate_reposts_and_assign_group_id(work)
    final_rows = len(work)

    # Combine rejected records report
    report_cols = [
        "property_type_name",
        "district_name",
        "ward_name",
        "price",
        "area",
        "published_at",
        "reject_reason",
        "name",
    ]
    rejected_report = pd.concat(
        [outliers_df[report_cols], reposts_df[report_cols]], ignore_index=True
    )

    repost_reasons = {
        str(k): int(v) for k, v in reposts_df["reject_reason"].value_counts().items()
    }

    summary: dict[str, Any] = {
        "initial_rows": initial_rows,
        "outliers_removed": int(len(outliers_df)),
        "after_outlier_rows": after_outlier_rows,
        "reposts_removed": int(len(reposts_df)),
        "final_rows": final_rows,
        "final_columns": int(work.shape[1]),
        "unique_property_groups": int(work["property_group_id"].nunique()),
        "entity_normalization_before": dirs_before,
        "entity_normalization_after": dirs_after,
        "outlier_reasons": outlier_reasons,
        "repost_reasons": repost_reasons,
        "physical_values_sanitized_to_nan": sanitized_counts,
        "property_type_counts": {
            str(k): int(v) for k, v in work["property_type_name"].value_counts().items()
        },
    }
    return work, rejected_report, summary
