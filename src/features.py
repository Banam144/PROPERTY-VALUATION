from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold

from src.data_cleaning import normalize_direction, normalize_street_name, parse_published_at

FORBIDDEN_FEATURE_SUBSTRINGS = ("price", "ppm2")

# Price / currency patterns to redact BEFORE any NLP extraction
_PRICE_REDACT_PATTERNS = [
    re.compile(r"\b\d+\s*(?:tỷ|ty|tỉ|tỏi)\s*\d+[xX]?\b", flags=re.IGNORECASE),
    re.compile(
        r"\b\d+(?:[.,]\d+)?[xX]?\s*(?:tỷ|ty|tỉ|tỏi|triệu|tr|nghìn|ngàn|vnd|vnđ|đồng)"
        r"(?:\s*\d*[xX]?\s*(?:triệu|tr))?"
        r"(?:\s*/\s*(?:1\s*)?(?:m2?|m|tháng|th|năm))?\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bgiá\s*(?:bán|chào|chỉ|mềm|tốt|hot|hời|rẻ|từ|nhỉnh|hơn|khoảng|thỏa\s*thuận|thoả\s*thuận)?\b",
        flags=re.IGNORECASE,
    ),
]

_BINARY_SIGNAL_PATTERNS: list[tuple[str, str]] = [
    ("is_mat_pho", r"\bmặt\s*phố\b|\bmp\b|\bnhà\s*phố\s*thương\s*mại\b"),
    ("is_phan_lo", r"\bphân\s*lô\b|\bbàn\s*cờ\b|\bkhu\s*cán\s*bộ\b|\bkhu\s*quân\s*đội\b"),
    ("is_oto_tranh", r"(?:ô\s*tô|ôtô|oto)\s*tránh\b|\b2\s*(?:ô\s*tô|ôtô|oto)\b|\bvỉa\s*hè\b"),
    (
        "is_oto_vao_nha",
        r"(?:ô\s*tô|ôtô|oto)\s*(?:vào\s*nhà|đỗ|qua|ngủ)|\bgara\b|\bngõ\s*(?:ô\s*tô|ôtô|oto)\b",
    ),
    ("is_ngo_thong", r"\bngõ\s*thông\b|\bngõ\s*nông\b|\bba\s*gác\b"),
    ("is_lo_goc", r"\blô\s*góc\b|\bcăn\s*góc\b|\b[23]\s*(?:mặt\s*)?thoáng\b|\bhai\s*mặt\s*thoáng\b"),
    ("has_thang_may", r"\bthang\s*máy\b|\bthag\s*máy\b"),
    ("is_kinh_doanh", r"\bkinh\s*doanh\b|\bkd\s*đỉnh\b|\bvăn\s*phòng\b|\bsầm\s*uất\b"),
    (
        "is_chdv_ccmn",
        r"\bdòng\s*tiền\b|\bchdv\b|\bcăn\s*hộ\s*dịch\s*vụ\b|\bccmn\b|\bchung\s*cư\s*mini\b|\bkhách\s*sạn\b",
    ),
    (
        "has_so_do",
        r"\bsổ\s*đỏ\b|\bsđcc\b|\bsổ\s*hồng\b|\bsổ\s*lâu\s*dài\b|\bpháp\s*lý\s*sạch\b|\bsẵn\s*sổ\b",
    ),
    (
        "has_full_noi_that",
        r"\bfull\s*(?:nội\s*thất|đồ|nt)\b|\bnội\s*thất\s*(?:cao\s*cấp|xịn|đầy\s*đủ|5\s*sao)\b|\btặng\s*(?:full\s*)?nội\s*thất\b",
    ),
    ("is_nha_moi", r"\bnhà\s*mới\b|\bxây\s*mới\b|\bdân\s*xây\b|\bmới\s*koong\b|\bở\s*ngay\b"),
    (
        "is_view_dep",
        r"\bview\s*(?:hồ|sông|công\s*viên|nội\s*khu)\b|\bhồ\s*tây\b|\bhồ\s*gươm\b|\bhồ\s*hoàn\s*kiếm\b",
    ),
    ("is_biet_thu_lk", r"\bbiệt\s*thự\b|\bliền\s*kề\b|\bshophouse\b"),
    ("is_studio_penthouse", r"\bstudio\b|\bduplex\b|\bpenthouse\b"),
]


def redact_price_mentions(series: pd.Series) -> pd.Series:
    s = series.astype("string").fillna("").str.casefold()
    for pat in _PRICE_REDACT_PATTERNS:
        s = s.str.replace(pat, " ", regex=True)
    s = s.str.replace(r"\[phone_number\]", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s

def _parse_extracted_float(df_ext: pd.DataFrame) -> pd.Series:
    combined = df_ext.iloc[:, 0]
    for i in range(1, df_ext.shape[1]):
        combined = combined.fillna(df_ext.iloc[:, i])
    cleaned = combined.astype("string").str.replace(",", ".", regex=False)
    return pd.to_numeric(cleaned, errors="coerce").astype("float64")

def extract_stateless_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    # Normalize entity columns so single-row inference matches training format
    work["house_direction"] = normalize_direction(
        work.get("house_direction", pd.Series("Không rõ", index=work.index))
    )
    work["balcony_direction"] = normalize_direction(
        work.get("balcony_direction", pd.Series("Không rõ", index=work.index))
    )
    work["street_name"] = normalize_street_name(
        work.get("street_name", pd.Series("Không rõ", index=work.index))
    )
    for cat_col in ["ward_name", "project_name", "district_name", "property_type_name"]:
        work[cat_col] = (
            work[cat_col].astype("string").fillna("Không rõ").str.strip().replace("", "Không rõ")
        )

    area = pd.to_numeric(work["area"], errors="coerce").astype("float64")
    is_apt = (work["property_type_name"] == "Căn hộ chung cư").astype("float64")
    is_house = (work["property_type_name"] == "Nhà").astype("float64")

    # Combine name + description and strictly redact all price mentions FIRST
    raw_text = (
        work.get("name", pd.Series("", index=work.index)).astype("string").fillna("")
        + " "
        + work.get("description", pd.Series("", index=work.index)).astype("string").fillna("")
    )
    safe_text = redact_price_mentions(raw_text)

    # Extract missing structural measurements from safe_text
    ext_floor = _parse_extracted_float(
        safe_text.str.extract(
            r"\b([1-4]?\d)\s*(?:tầng|tâng|tang|lầu)\b|\b\d+(?:[.,]\d+)?\s*m2?\s*[x*]\s*([1-2]?\d)\s*t\b"
        )
    )
    ext_floor = ext_floor.where(
        (ext_floor >= 1) & (((is_house == 1) & (ext_floor <= 25)) | ((is_apt == 1) & (ext_floor <= 45)))
    )

    ext_frontage = _parse_extracted_float(
        safe_text.str.extract(
            r"(?:mặt\s*tiền|mt)\s*[:=~]?\s*(\d{1,2}(?:[.,]\d{1,2})?)\s*m?\b"
            r"|\b(\d{1,2}(?:[.,]\d{1,2})?)\s*m\s*(?:mặt\s*tiền|mt)\b"
        )
    )
    ext_frontage = ext_frontage.where(
        (ext_frontage >= 1.5)
        & (ext_frontage <= 35.0)
        & ((is_apt == 1) | ((area / ext_frontage) >= 1.5))
    )

    ext_bed = _parse_extracted_float(
        safe_text.str.extract(r"\b([1-2]?\d)\s*(?:phòng\s*ngủ|p\.?\s*ngủ|pn|ngủ)\b")
    )
    ext_bed = ext_bed.where(
        (ext_bed >= 1) & (((is_house == 1) & (ext_bed <= 30)) | ((is_apt == 1) & (ext_bed <= 6)))
    )

    ext_bath = _parse_extracted_float(
        safe_text.str.extract(r"\b([1-2]?\d)\s*(?:wc|vệ\s*sinh|toilet|phòng\s*tắm)\b")
    )
    ext_bath = ext_bath.where(
        (ext_bath >= 1) & (((is_house == 1) & (ext_bath <= 30)) | ((is_apt == 1) & (ext_bath <= 6)))
    )

    ext_road = _parse_extracted_float(
        safe_text.str.extract(
            r"(?:ngõ|đường|vỉa\s*hè)\s*(?:rộng\s*|trước\s*nhà\s*)?(\d{1,2}(?:[.,]\d{1,2})?)\s*m\b"
        )
    )
    ext_road = ext_road.where((ext_road >= 1.0) & (ext_road <= 40.0))

    # Original optional structured numeric columns
    fc_orig = pd.to_numeric(work.get("floor_count"), errors="coerce").astype("float64")
    fw_orig = pd.to_numeric(work.get("frontage_width"), errors="coerce").astype("float64")
    hd_orig = pd.to_numeric(work.get("house_depth"), errors="coerce").astype("float64")
    rw_orig = pd.to_numeric(work.get("road_width"), errors="coerce").astype("float64")
    bed_orig = pd.to_numeric(work.get("bedroom_count"), errors="coerce").astype("float64")
    bath_orig = pd.to_numeric(work.get("bathroom_count"), errors="coerce").astype("float64")

    # Fuse structured + text backfill
    fc_eff = fc_orig.fillna(ext_floor)
    fw_eff = fw_orig.fillna(ext_frontage)
    rw_eff = rw_orig.fillna(ext_road)
    bed_eff = bed_orig.fillna(ext_bed)
    bath_eff = bath_orig.fillna(ext_bath)

    feat: dict[str, Any] = {
        "area": area,
        "log_area": np.log(np.clip(area, 1.0, None)),
        "sqrt_area": np.sqrt(np.clip(area, 0.0, None)),
        "is_apartment": is_apt,
        "floor_count_eff": fc_eff,
        "frontage_width_eff": fw_eff,
        "house_depth": hd_orig,
        "road_width_eff": rw_eff,
        "bedroom_count_eff": bed_eff,
        "bathroom_count_eff": bath_eff,
        "floor_orig_missing": fc_orig.isna().astype("float64"),
        "frontage_orig_missing": fw_orig.isna().astype("float64"),
        "depth_orig_missing": hd_orig.isna().astype("float64"),
        "road_orig_missing": rw_orig.isna().astype("float64"),
        "bedroom_orig_missing": bed_orig.isna().astype("float64"),
        "bathroom_orig_missing": bath_orig.isna().astype("float64"),
        "floor_from_text": (fc_orig.isna() & ext_floor.notna()).astype("float64"),
        "frontage_from_text": (fw_orig.isna() & ext_frontage.notna()).astype("float64"),
        "road_from_text": (rw_orig.isna() & ext_road.notna()).astype("float64"),
        "bedroom_from_text": (bed_orig.isna() & ext_bed.notna()).astype("float64"),
    }

    # Geometric ratios & floor area estimates
    est_floors_for_gfa = np.where(is_apt == 1.0, 1.0, fc_eff.fillna(4.0))
    total_gfa = area * est_floors_for_gfa
    feat["total_floor_area_est"] = total_gfa
    feat["log_total_floor_area_est"] = np.log(np.clip(total_gfa, 1.0, None))

    implied_depth = np.where(fw_eff > 0, area / fw_eff, hd_orig)
    feat["implied_depth"] = implied_depth
    feat["shape_aspect_ratio"] = np.where(implied_depth > 0, fw_eff / implied_depth, np.nan)
    feat["area_per_bedroom"] = np.where(bed_eff > 0, area / bed_eff, np.nan)
    feat["bath_per_bedroom"] = np.where(bed_eff > 0, bath_eff / bed_eff, np.nan)
    feat["has_known_project"] = (work["project_name"] != "Không rõ").astype("float64")
    feat["has_known_street"] = (work["street_name"] != "Không rõ").astype("float64")
    feat["has_known_ward"] = (work["ward_name"] != "Không rõ").astype("float64")

    # Binary NLP physical/accessibility/legal features
    for col_name, pattern in _BINARY_SIGNAL_PATTERNS:
        feat[col_name] = safe_text.str.contains(pattern, regex=True, na=False).astype("float64")

    # Ordinal accessibility tier (0..4): captures Hanoi alley vs streetfront hierarchy
    access_tier = np.zeros(len(work), dtype="float64")
    access_tier = np.where(feat["is_ngo_thong"] == 1.0, 1.0, access_tier)
    access_tier = np.where(feat["is_oto_vao_nha"] == 1.0, 2.0, access_tier)
    access_tier = np.where(feat["is_oto_tranh"] == 1.0, 3.0, access_tier)
    access_tier = np.where(feat["is_mat_pho"] == 1.0, 4.0, access_tier)
    feat["access_tier"] = access_tier
    feat["house_x_access_tier"] = is_house * access_tier
    feat["house_x_log_area"] = is_house * feat["log_area"]

    # Calendar day-of-week signal (optional; falls back cleanly to 0 if absent in live CLI)
    if "published_at" in work.columns:
        pub_dt = parse_published_at(work["published_at"])
        dow = pub_dt.dt.day_of_week.fillna(0).astype("float64")
        feat["is_weekend"] = (dow >= 5).astype("float64")
    else:
        feat["is_weekend"] = np.zeros(len(work), dtype="float64")

    return pd.DataFrame(feat, index=work.index)


class HierarchicalOOFTargetEncoder(BaseEstimator, TransformerMixin):
    """5-level Hierarchical Empirical Bayes Target Encoder with Out-Of-Fold training protection.

    Encodes `y_log_ppm2 = log(price / area)` across:
      property_type -> (property_type, district) -> (property_type, district, ward)
                    -> (property_type, district, street) & (property_type, district, project)
    """

    def __init__(
        self,
        n_splits: int = 5,
        m_district: float = 20.0,
        m_ward: float = 15.0,
        m_street: float = 15.0,
        m_project: float = 10.0,
        random_state: int = 42,
    ) -> None:
        self.n_splits = n_splits
        self.m_district = m_district
        self.m_ward = m_ward
        self.m_street = m_street
        self.m_project = m_project
        self.random_state = random_state

    @staticmethod
    def _prepare_keys(df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out["ptype"] = df["property_type_name"].astype(str).str.strip()
        out["dist"] = df["district_name"].astype(str).str.strip()
        out["ward"] = df["ward_name"].astype("string").fillna("Không rõ").astype(str).str.strip()
        out["street"] = normalize_street_name(
            df.get("street_name", pd.Series("Không rõ", index=df.index))
        ).astype(str)
        out["proj"] = df["project_name"].astype("string").fillna("Không rõ").astype(str).str.strip()
        return out

    def _fit_tables(self, keys: pd.DataFrame, y_unit: pd.Series) -> dict[str, Any]:
        tmp = keys.copy()
        tmp["y"] = y_unit.to_numpy(dtype="float64")

        global_mean = float(tmp["y"].mean())
        global_std = float(tmp["y"].std(ddof=0)) if len(tmp) > 1 else 0.5

        # Level 1: property_type
        type_mean = tmp.groupby("ptype")["y"].mean().to_dict()

        # Level 2: (ptype, dist) smoothed toward type_mean
        g_dist = tmp.groupby(["ptype", "dist"])["y"].agg(["count", "mean", "std"]).reset_index()
        dist_te: dict[tuple[str, str], float] = {}
        dist_std: dict[tuple[str, str], float] = {}
        dist_cnt: dict[tuple[str, str], int] = {}
        for r in g_dist.itertuples(index=False):
            prior = type_mean.get(r.ptype, global_mean)
            n = int(r.count)
            te = (n * float(r.mean) + self.m_district * prior) / (n + self.m_district)
            sd = float(r.std) if (n >= 5 and pd.notna(r.std)) else global_std
            dist_te[(r.ptype, r.dist)] = te
            dist_std[(r.ptype, r.dist)] = sd
            dist_cnt[(r.ptype, r.dist)] = n

        # Level 3: (ptype, dist, ward) excluding "Không rõ"
        valid_w = tmp[tmp["ward"] != "Không rõ"]
        g_ward = valid_w.groupby(["ptype", "dist", "ward"])["y"].agg(["count", "mean"]).reset_index()
        ward_te: dict[tuple[str, str, str], float] = {}
        ward_cnt: dict[tuple[str, str, str], int] = {}
        for r in g_ward.itertuples(index=False):
            prior = dist_te.get((r.ptype, r.dist), type_mean.get(r.ptype, global_mean))
            n = int(r.count)
            ward_te[(r.ptype, r.dist, r.ward)] = (
                n * float(r.mean) + self.m_ward * prior
            ) / (n + self.m_ward)
            ward_cnt[(r.ptype, r.dist, r.ward)] = n

        # Level 4: (ptype, dist, street) excluding "Không rõ"
        # First compute average ward prior per street if needed -> shrink toward dist_te
        valid_s = tmp[tmp["street"] != "Không rõ"]
        g_street = (
            valid_s.groupby(["ptype", "dist", "street"])["y"].agg(["count", "mean"]).reset_index()
        )
        street_te: dict[tuple[str, str, str], float] = {}
        street_cnt: dict[tuple[str, str, str], int] = {}
        for r in g_street.itertuples(index=False):
            prior = dist_te.get((r.ptype, r.dist), type_mean.get(r.ptype, global_mean))
            n = int(r.count)
            street_te[(r.ptype, r.dist, r.street)] = (
                n * float(r.mean) + self.m_street * prior
            ) / (n + self.m_street)
            street_cnt[(r.ptype, r.dist, r.street)] = n

        # Level 5: (ptype, dist, proj) excluding "Không rõ"
        valid_p = tmp[tmp["proj"] != "Không rõ"]
        g_proj = (
            valid_p.groupby(["ptype", "dist", "proj"])["y"].agg(["count", "mean"]).reset_index()
        )
        proj_te: dict[tuple[str, str, str], float] = {}
        proj_cnt: dict[tuple[str, str, str], int] = {}
        for r in g_proj.itertuples(index=False):
            prior = dist_te.get((r.ptype, r.dist), type_mean.get(r.ptype, global_mean))
            n = int(r.count)
            proj_te[(r.ptype, r.dist, r.proj)] = (
                n * float(r.mean) + self.m_project * prior
            ) / (n + self.m_project)
            proj_cnt[(r.ptype, r.dist, r.proj)] = n

        return {
            "global_mean": global_mean,
            "global_std": global_std,
            "type_mean": type_mean,
            "dist_te": dist_te,
            "dist_std": dist_std,
            "dist_cnt": dist_cnt,
            "ward_te": ward_te,
            "ward_cnt": ward_cnt,
            "street_te": street_te,
            "street_cnt": street_cnt,
            "proj_te": proj_te,
            "proj_cnt": proj_cnt,
        }

    @staticmethod
    def _apply_tables(keys: pd.DataFrame, area: pd.Series, tables: dict[str, Any]) -> pd.DataFrame:
        n = len(keys)
        te_type = np.empty(n, dtype="float64")
        te_dist = np.empty(n, dtype="float64")
        sd_dist = np.empty(n, dtype="float64")
        cnt_dist = np.empty(n, dtype="float64")
        te_ward = np.empty(n, dtype="float64")
        cnt_ward = np.empty(n, dtype="float64")
        te_street = np.empty(n, dtype="float64")
        cnt_street = np.empty(n, dtype="float64")
        te_proj = np.empty(n, dtype="float64")
        cnt_proj = np.empty(n, dtype="float64")
        anchor_ppm2 = np.empty(n, dtype="float64")

        g_mean = tables["global_mean"]
        g_std = tables["global_std"]
        type_m = tables["type_mean"]
        d_te = tables["dist_te"]
        d_sd = tables["dist_std"]
        d_cnt = tables["dist_cnt"]
        w_te = tables["ward_te"]
        w_cnt = tables["ward_cnt"]
        s_te = tables["street_te"]
        s_cnt = tables["street_cnt"]
        p_te = tables["proj_te"]
        p_cnt = tables["proj_cnt"]

        ptypes = keys["ptype"].tolist()
        dists = keys["dist"].tolist()
        wards = keys["ward"].tolist()
        streets = keys["street"].tolist()
        projs = keys["proj"].tolist()

        for i in range(n):
            pt = ptypes[i]
            di = dists[i]
            wa = wards[i]
            st = streets[i]
            pr = projs[i]

            t_val = type_m.get(pt, g_mean)
            d_val = d_te.get((pt, di), t_val)
            d_s = d_sd.get((pt, di), g_std)
            dc = d_cnt.get((pt, di), 0)

            w_val = w_te.get((pt, di, wa), d_val) if wa != "Không rõ" else d_val
            wc = w_cnt.get((pt, di, wa), 0) if wa != "Không rõ" else 0

            has_st = (st != "Không rõ") and ((pt, di, st) in s_te)
            s_val = s_te[(pt, di, st)] if has_st else w_val
            sc = s_cnt.get((pt, di, st), 0) if st != "Không rõ" else 0

            has_pr = (pr != "Không rõ") and ((pt, di, pr) in p_te)
            p_val = p_te[(pt, di, pr)] if has_pr else w_val
            pc = p_cnt.get((pt, di, pr), 0) if pr != "Không rõ" else 0

            if has_pr and has_st:
                anc = 0.55 * p_val + 0.25 * s_val + 0.20 * w_val
            elif has_pr:
                anc = 0.70 * p_val + 0.30 * w_val
            elif has_st:
                anc = 0.65 * s_val + 0.35 * w_val
            else:
                anc = w_val

            te_type[i] = t_val
            te_dist[i] = d_val
            sd_dist[i] = d_s
            cnt_dist[i] = np.log1p(dc)
            te_ward[i] = w_val
            cnt_ward[i] = np.log1p(wc)
            te_street[i] = s_val
            cnt_street[i] = np.log1p(sc)
            te_proj[i] = p_val
            cnt_proj[i] = np.log1p(pc)
            anchor_ppm2[i] = anc

        log_area = np.log(np.clip(pd.to_numeric(area, errors="coerce").to_numpy(dtype="float64"), 1.0, None))

        return pd.DataFrame(
            {
                "te_type_log_ppm2": te_type,
                "te_district_log_ppm2": te_dist,
                "te_district_std": sd_dist,
                "district_log_train_count": cnt_dist,
                "te_ward_log_ppm2": te_ward,
                "ward_log_train_count": cnt_ward,
                "te_street_log_ppm2": te_street,
                "street_log_train_count": cnt_street,
                "te_project_log_ppm2": te_proj,
                "project_log_train_count": cnt_proj,
                "spatial_anchor_log_ppm2": anchor_ppm2,
                "spatial_anchor_log_tot": anchor_ppm2 + log_area,
                "ward_vs_district_diff": te_ward - te_dist,
                "street_vs_ward_diff": te_street - te_ward,
                "project_vs_ward_diff": te_proj - te_ward,
            },
            index=keys.index,
        )

    def fit(self, X: pd.DataFrame, y_price: pd.Series | np.ndarray) -> "HierarchicalOOFTargetEncoder":
        keys = self._prepare_keys(X)
        area = pd.to_numeric(X["area"], errors="coerce").astype("float64")
        y_arr = np.asarray(y_price, dtype="float64")
        y_unit = pd.Series(np.log(y_arr / area.to_numpy()), index=X.index)
        self.tables_ = self._fit_tables(keys, y_unit)
        return self

    def fit_transform(self, X: pd.DataFrame, y_price: pd.Series | np.ndarray) -> pd.DataFrame:
        keys = self._prepare_keys(X)
        area = pd.to_numeric(X["area"], errors="coerce").astype("float64")
        y_arr = np.asarray(y_price, dtype="float64")
        y_unit = pd.Series(np.log(y_arr / area.to_numpy()), index=X.index)

        # Fit frozen full-train tables for future .transform() calls on Val / Test / Live
        self.tables_ = self._fit_tables(keys, y_unit)

        # Compute Out-Of-Fold encodings on Train to prevent self-target overfitting
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        oof_parts: list[pd.DataFrame] = []
        for train_idx, holdout_idx in kf.split(X):
            fold_tables = self._fit_tables(keys.iloc[train_idx], y_unit.iloc[train_idx])
            fold_df = self._apply_tables(
                keys.iloc[holdout_idx], area.iloc[holdout_idx], fold_tables
            )
            oof_parts.append(fold_df)

        oof_out = pd.concat(oof_parts, axis=0).loc[X.index]
        return oof_out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "tables_"):
            raise RuntimeError("HierarchicalOOFTargetEncoder must be fitted on Train before transform.")
        keys = self._prepare_keys(X)
        area = pd.to_numeric(X["area"], errors="coerce").astype("float64")
        return self._apply_tables(keys, area, self.tables_)


class HanoiValuationFeaturePipeline(BaseEstimator, TransformerMixin):
    def __init__(self, n_splits: int = 5, random_state: int = 42) -> None:
        self.n_splits = n_splits
        self.random_state = random_state

    @staticmethod
    def _sanitize_input(df: pd.DataFrame) -> pd.DataFrame:
        # Explicitly drop target or group metadata if present so they can NEVER enter features
        drop_cols = [c for c in ["price", "property_group_id"] if c in df.columns]
        return df.drop(columns=drop_cols) if drop_cols else df.copy()

    def _encode_categoricals_fit(self, df: pd.DataFrame) -> None:
        self.districts_ = sorted(df["district_name"].astype(str).str.strip().unique().tolist())
        self.house_dirs_ = sorted(normalize_direction(df["house_direction"]).unique().tolist())
        self.balc_dirs_ = sorted(normalize_direction(df["balcony_direction"]).unique().tolist())

    def _encode_categoricals_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        dist_s = df["district_name"].astype(str).str.strip()
        hdir_s = normalize_direction(df.get("house_direction", pd.Series("Không rõ", index=df.index)))
        bdir_s = normalize_direction(df.get("balcony_direction", pd.Series("Không rõ", index=df.index)))

        cols: dict[str, np.ndarray] = {}
        for d in self.districts_:
            cols[f"dist_ohe_{d}"] = (dist_s == d).to_numpy(dtype="float64")
        for hd in self.house_dirs_:
            cols[f"hdir_ohe_{hd}"] = (hdir_s == hd).to_numpy(dtype="float64")
        for bd in self.balc_dirs_:
            cols[f"bdir_ohe_{bd}"] = (bdir_s == bd).to_numpy(dtype="float64")
        return pd.DataFrame(cols, index=df.index)

    @staticmethod
    def _verify_no_forbidden_columns(feat_df: pd.DataFrame) -> None:
        for col in feat_df.columns:
            low = col.lower()
            if "price" in low:
                raise AssertionError(f"CRITICAL LEAKAGE GUARD: Forbidden column '{col}' found in features!")

    def fit_transform(self, X: pd.DataFrame, y_price: pd.Series | np.ndarray) -> pd.DataFrame:
        clean_X = self._sanitize_input(X)
        stateless_df = extract_stateless_features(clean_X)

        self.target_encoder_ = HierarchicalOOFTargetEncoder(
            n_splits=self.n_splits, random_state=self.random_state
        )
        spatial_oof_df = self.target_encoder_.fit_transform(clean_X, y_price)

        self._encode_categoricals_fit(clean_X)
        ohe_df = self._encode_categoricals_transform(clean_X)

        full_df = pd.concat([stateless_df, spatial_oof_df, ohe_df], axis=1)
        self._verify_no_forbidden_columns(full_df)
        self.feature_names_ = full_df.columns.tolist()

        # Fit train-only group medians (by is_apartment) for models that do not accept NaN
        self.apt_medians_ = full_df[full_df["is_apartment"] == 1.0].median(numeric_only=True).to_dict()
        self.house_medians_ = full_df[full_df["is_apartment"] == 0.0].median(numeric_only=True).to_dict()
        self.global_medians_ = full_df.median(numeric_only=True).fillna(0.0).to_dict()
        return full_df

    def fit(self, X: pd.DataFrame, y_price: pd.Series | np.ndarray) -> "HanoiValuationFeaturePipeline":
        self.fit_transform(X, y_price)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "target_encoder_"):
            raise RuntimeError("HanoiValuationFeaturePipeline must be fitted before transform.")
        clean_X = self._sanitize_input(X)
        stateless_df = extract_stateless_features(clean_X)
        spatial_df = self.target_encoder_.transform(clean_X)
        ohe_df = self._encode_categoricals_transform(clean_X)

        full_df = pd.concat([stateless_df, spatial_df, ohe_df], axis=1)
        full_df = full_df[self.feature_names_]
        self._verify_no_forbidden_columns(full_df)
        return full_df

    def impute_with_train_medians(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        """Impute NaN values using group-specific medians learned ONLY on X_train."""
        out = feat_df.copy()
        is_apt_mask = out["is_apartment"] == 1.0
        for col in out.columns:
            if not out[col].isna().any():
                continue
            g_med = self.global_medians_.get(col, 0.0)
            if pd.isna(g_med):
                g_med = 0.0
            a_med = self.apt_medians_.get(col, g_med)
            if pd.isna(a_med):
                a_med = g_med
            h_med = self.house_medians_.get(col, g_med)
            if pd.isna(h_med):
                h_med = g_med

            if is_apt_mask.any():
                out.loc[is_apt_mask, col] = out.loc[is_apt_mask, col].fillna(a_med)
            if (~is_apt_mask).any():
                out.loc[~is_apt_mask, col] = out.loc[~is_apt_mask, col].fillna(h_med)
        return out
