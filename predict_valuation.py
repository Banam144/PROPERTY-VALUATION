"""Stage 7 (P1): Production Property Valuation CLI & Exact TreeSHAP Explainer
(`predict_valuation.py`).

Supports:
1. Interactive / single-property CLI flags (`--type`, `--district`, `--ward`, `--area`, ...)
2. Batch JSON file valuation (`--input-json path.json`)
3. Built-in realistic Hanoi valuation showcase (`--demo`)

Outputs:
- Point Valuation (VNĐ & Tỷ VNĐ) and Unit Price (Triệu VNĐ/m²)
- Calibrated 80% Mondrian Conformal Valuation Interval [P10, P90]
- Top exact TreeSHAP (`pred_contrib=True`) feature drivers explaining the valuation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.intervals_and_analysis import MondrianConformalValuationSystem

DEFAULT_MODEL_PATH = Path("models/avm_conformal_system.joblib")


def compute_local_treeshap_drivers(
    system: MondrianConformalValuationSystem, X_feat_one: pd.DataFrame, top_k: int = 5
) -> list[dict[str, Any]]:
    """Compute exact TreeSHAP log-unit-price contributions for a single property
    by blending the Segmented Expert and Unified LightGBM TreeSHAP vectors."""
    bundle = system.point_bundle
    w_seg = bundle.w_seg
    is_apt = bool(X_feat_one["is_apartment"].iloc[0] == 1.0)

    uni_contrib = np.asarray(
        bundle.unified_model.model_.predict(X_feat_one, pred_contrib=True), dtype="float64"
    )[0]
    if is_apt:
        seg_contrib = np.asarray(
            bundle.segmented_model.apt_model_.predict(X_feat_one, pred_contrib=True),
            dtype="float64",
        )[0]
    else:
        seg_contrib = np.asarray(
            bundle.segmented_model.house_model_.predict(X_feat_one, pred_contrib=True),
            dtype="float64",
        )[0]

    # Last element of LightGBM pred_contrib is the expected base value (bias)
    blended_shap = w_seg * seg_contrib[:-1] + (1.0 - w_seg) * uni_contrib[:-1]
    feat_names = bundle.feature_pipeline.feature_names_
    feat_vals = X_feat_one.iloc[0].to_numpy(dtype="float64")

    order = np.argsort(np.abs(blended_shap))[::-1][:top_k]
    drivers: list[dict[str, Any]] = []
    for idx in order:
        log_delta = float(blended_shap[idx])
        pct_mult = (np.exp(log_delta) - 1.0) * 100.0
        val = feat_vals[idx]
        drivers.append(
            {
                "feature": feat_names[idx],
                "feature_value": None if np.isnan(val) else round(float(val), 4),
                "log_contribution": round(log_delta, 4),
                "approx_unit_price_effect_pct": round(float(pct_mult), 2),
            }
        )
    return drivers


def valuate_single_property(
    system: MondrianConformalValuationSystem, payload: dict[str, Any]
) -> dict[str, Any]:
    """Valuate a single Hanoi residential property dictionary and return structured output."""
    record = {
        "name": payload.get("name", ""),
        "description": payload.get("description", ""),
        "property_type_name": payload.get("property_type_name", "Nhà"),
        "province_name": "Hà Nội",
        "district_name": payload["district_name"],
        "ward_name": payload.get("ward_name", "Không rõ"),
        "street_name": payload.get("street_name", "Không rõ"),
        "project_name": payload.get("project_name", "Không rõ"),
        "area": float(payload["area"]),
        "floor_count": payload.get("floor_count", np.nan),
        "frontage_width": payload.get("frontage_width", np.nan),
        "house_depth": payload.get("house_depth", np.nan),
        "road_width": payload.get("road_width", np.nan),
        "bedroom_count": payload.get("bedroom_count", np.nan),
        "bathroom_count": payload.get("bathroom_count", np.nan),
        "house_direction": payload.get("house_direction", "Không rõ"),
        "balcony_direction": payload.get("balcony_direction", "Không rõ"),
    }
    df_one = pd.DataFrame([record])
    pred_df = system.predict_with_intervals(df_one)
    X_feat_one = system.point_bundle.feature_pipeline.transform(df_one)
    drivers = compute_local_treeshap_drivers(system, X_feat_one, top_k=5)

    p_low = float(pred_df["pred_low_vnd"].iloc[0])
    p_mid = float(pred_df["pred_price_vnd"].iloc[0])
    p_high = float(pred_df["pred_high_vnd"].iloc[0])
    ppm2 = float(pred_df["pred_ppm2_million"].iloc[0])

    return {
        "input_summary": {
            "property_type_name": record["property_type_name"],
            "district_name": record["district_name"],
            "ward_name": record["ward_name"],
            "street_name": record["street_name"],
            "project_name": record["project_name"],
            "area_m2": record["area"],
        },
        "valuation": {
            "fair_price_vnd": round(p_mid),
            "fair_price_ty": round(p_mid / 1e9, 3),
            "unit_price_million_per_m2": round(ppm2, 2),
            "conformal_80_low_ty": round(p_low / 1e9, 3),
            "conformal_80_high_ty": round(p_high / 1e9, 3),
            "interval_width_pct": round(float(pred_df["interval_width_pct"].iloc[0]), 2),
        },
        "top_treeshap_drivers": drivers,
    }


DEMO_CASES: list[dict[str, Any]] = [
    {
        "property_type_name": "Căn hộ chung cư",
        "district_name": "Nam Từ Liêm",
        "ward_name": "Tây Mỗ",
        "project_name": "Phân khu The Sapphire 2 - Vinhomes Smart City",
        "area": 55.0,
        "bedroom_count": 2,
        "bathroom_count": 1,
        "balcony_direction": "Đông Nam",
        "name": "Căn hộ 2PN full nội thất đẹp sẵn sổ lâu dài",
        "description": "Căn hộ tầng trung thoáng mát, full nội thất cao cấp, sổ hồng lâu dài.",
    },
    {
        "property_type_name": "Căn hộ chung cư",
        "district_name": "Cầu Giấy",
        "ward_name": "Trung Hòa",
        "street_name": "Trần Duy Hưng",
        "project_name": "Vinhomes D'Capitale",
        "area": 85.0,
        "bedroom_count": 3,
        "bathroom_count": 2,
        "balcony_direction": "Đông Nam",
        "name": "Căn góc 3PN view hồ điều hòa Trung Hòa Nhân Chính",
        "description": "Căn góc 2 mặt thoáng view hồ, full nội thất xịn, sổ đỏ lâu dài.",
    },
    {
        "property_type_name": "Nhà",
        "district_name": "Hoàng Mai",
        "ward_name": "Đại Kim",
        "street_name": "Kim Giang",
        "area": 38.0,
        "floor_count": 5,
        "frontage_width": 3.8,
        "road_width": 2.5,
        "bedroom_count": 3,
        "bathroom_count": 4,
        "house_direction": "Đông Nam",
        "name": "Nhà dân xây mới 5 tầng ngõ thông ba gác Kim Giang",
        "description": "Nhà mới ở ngay, ngõ thông gần phố, sổ đỏ chính chủ vuông đẹp.",
    },
    {
        "property_type_name": "Nhà",
        "district_name": "Cầu Giấy",
        "ward_name": "Yên Hòa",
        "street_name": "Trung Kính",
        "area": 65.0,
        "floor_count": 7,
        "frontage_width": 5.2,
        "road_width": 15.0,
        "bedroom_count": 8,
        "bathroom_count": 8,
        "house_direction": "Đông Nam",
        "name": "Nhà mặt phố Trung Kính phân lô vỉa hè ô tô tránh 7 tầng thang máy",
        "description": "Mặt phố kinh doanh sầm uất, vỉa hè rộng ô tô tránh, thang máy nhập khẩu, sổ đỏ vuông.",
    },
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hanoi Property Valuation (AVM) Inference CLI")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to AVM joblib")
    p.add_argument("--demo", action="store_true", help="Run 4 realistic Hanoi valuation showcases")
    p.add_argument("--input-json", type=Path, help="JSON file containing a dict or list of dicts")
    p.add_argument("--type", dest="property_type_name", default="Nhà", choices=["Nhà", "Căn hộ chung cư"])
    p.add_argument("--district", dest="district_name", help="District name (e.g. 'Cầu Giấy')")
    p.add_argument("--ward", dest="ward_name", default="Không rõ")
    p.add_argument("--street", dest="street_name", default="Không rõ")
    p.add_argument("--project", dest="project_name", default="Không rõ")
    p.add_argument("--area", type=float, help="Area in m2")
    p.add_argument("--floors", dest="floor_count", type=float, default=np.nan)
    p.add_argument("--frontage", dest="frontage_width", type=float, default=np.nan)
    p.add_argument("--road-width", dest="road_width", type=float, default=np.nan)
    p.add_argument("--bedrooms", dest="bedroom_count", type=float, default=np.nan)
    p.add_argument("--bathrooms", dest="bathroom_count", type=float, default=np.nan)
    p.add_argument("--house-direction", default="Không rõ")
    p.add_argument("--balcony-direction", default="Không rõ")
    p.add_argument("--name", default="")
    p.add_argument("--description", default="")
    return p


def main() -> None:
    args = build_parser().parse_args()
    system: MondrianConformalValuationSystem = joblib.load(args.model)

    if args.demo:
        results = [valuate_single_property(system, case) for case in DEMO_CASES]
        out_path = Path("reports/demo_valuations.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    if args.input_json:
        raw = json.loads(args.input_json.read_text(encoding="utf-8"))
        items = raw if isinstance(raw, list) else [raw]
        results = [valuate_single_property(system, item) for item in items]
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    if not args.district_name or args.area is None:
        raise SystemExit("Vui long truyen --demo, --input-json, hoac cap --district va --area.")

    payload = {
        "property_type_name": args.property_type_name,
        "district_name": args.district_name,
        "ward_name": args.ward_name,
        "street_name": args.street_name,
        "project_name": args.project_name,
        "area": args.area,
        "floor_count": args.floor_count,
        "frontage_width": args.frontage_width,
        "road_width": args.road_width,
        "bedroom_count": args.bedroom_count,
        "bathroom_count": args.bathroom_count,
        "house_direction": args.house_direction,
        "balcony_direction": args.balcony_direction,
        "name": args.name,
        "description": args.description,
    }
    res = valuate_single_property(system, payload)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
