"""Stage 5–7 Automated Verification Suite (`tests/test_models_and_inference.py`).

Verifies:
1. Trained Champion model significantly outperforms Naive District Median Baseline on Test.
2. Mondrian Conformal Quantile Regression intervals strictly obey `P10 < Point < P90` with
   zero crossing and achieve calibrated empirical coverage (~80%) on untouched Test data.
3. Domain economic monotonicity:
   - Larger area -> strictly higher predicted property value (ceteris paribus).
   - Prime streetfront + car-avoiding access -> strictly higher value than narrow alley.
4. Single-property inference & exact TreeSHAP explanation return valid structured results.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from predict_valuation import valuate_single_property


def test_champion_model_significantly_beats_naive_baseline_on_test() -> None:
    comp = pd.read_csv(Path("reports/model_comparison.csv"))
    test_comp = comp[comp["split"] == "test"].set_index("model")

    naive = test_comp.loc["1_Naive_District_Median"]
    champ = test_comp.loc["6_Champion_LightGBM_Ensemble"]

    assert champ["MAPE_pct"] < naive["MAPE_pct"] - 5.0
    assert champ["MdAPE_pct"] < naive["MdAPE_pct"] - 4.0
    assert champ["PPE20_pct"] > naive["PPE20_pct"] + 10.0
    assert champ["R2_log"] > naive["R2_log"] + 0.12


def test_conformal_intervals_non_crossing_and_calibrated_coverage() -> None:
    summary = json.loads(
        Path("reports/conformal_and_slice_summary.json").read_text(encoding="utf-8")
    )
    test_diag = summary["test_interval_diagnostics"]

    assert test_diag["monotonic_non_crossing_pct"] == 100.0
    # Target coverage is 80%; on chronological out-of-time Test, check well-calibrated [74%, 86%]
    assert 74.0 <= test_diag["PICP_80_pct"] <= 86.0


def test_economic_monotonicity_area_and_streetfront_premium() -> None:
    system = joblib.load(Path("models/avm_conformal_system.joblib"))

    base = {
        "property_type_name": "Nhà",
        "district_name": "Cầu Giấy",
        "ward_name": "Yên Hòa",
        "street_name": "Trung Kính",
        "floor_count": 5,
        "frontage_width": 4.2,
        "bedroom_count": 4,
        "bathroom_count": 4,
        "name": "Nhà dân xây 5 tầng ngõ thông Yên Hòa",
        "description": "Nhà đẹp sổ đỏ chính chủ ở ngay",
    }

    v_40 = valuate_single_property(system, {**base, "area": 40.0})["valuation"]["fair_price_vnd"]
    v_60 = valuate_single_property(system, {**base, "area": 60.0})["valuation"]["fair_price_vnd"]
    v_85 = valuate_single_property(system, {**base, "area": 85.0})["valuation"]["fair_price_vnd"]
    assert v_40 < v_60 < v_85, f"Expected area monotonicity, got {v_40}, {v_60}, {v_85}"

    alley_case = {
        **base,
        "area": 55.0,
        "road_width": 2.0,
        "name": "Nhà ngõ nhỏ yên tĩnh 5 tầng Trung Kính",
        "description": "Sổ đỏ chính chủ",
    }
    streetfront_case = {
        **base,
        "area": 55.0,
        "road_width": 15.0,
        "name": "Nhà mặt phố Trung Kính vỉa hè ô tô tránh kinh doanh đỉnh 5 tầng thang máy",
        "description": "Mặt phố lớn vỉa hè kinh doanh sầm uất thang máy sổ đỏ",
    }

    res_alley = valuate_single_property(system, alley_case)
    res_street = valuate_single_property(system, streetfront_case)
    assert (
        res_street["valuation"]["fair_price_vnd"] > res_alley["valuation"]["fair_price_vnd"] * 1.25
    ), "Streetfront property should carry >25% premium over narrow alley ceteris paribus!"
    assert len(res_street["top_treeshap_drivers"]) == 5


def test_cli_entrypoint_single_property_flags() -> None:
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "predict_valuation.py",
        "--type",
        "Nhà",
        "--district",
        "Đống Đa",
        "--ward",
        "Láng Hạ",
        "--street",
        "Thái Hà",
        "--area",
        "48",
        "--floors",
        "5",
        "--frontage",
        "4.2",
        "--road-width",
        "4.5",
        "--bedrooms",
        "4",
        "--bathrooms",
        "4",
        "--name",
        "Nhà phân lô Thái Hà ô tô vào nhà 5 tầng thang máy",
    ]
    out = subprocess.check_output(cmd, text=True, encoding="utf-8")
    parsed = json.loads(out)
    val = parsed["valuation"]
    assert val["conformal_80_low_ty"] < val["fair_price_ty"] < val["conformal_80_high_ty"]
    assert val["fair_price_ty"] > 5.0
    assert len(parsed["top_treeshap_drivers"]) == 5

