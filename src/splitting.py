from __future__ import annotations

from typing import Any

import pandas as pd

from src.data_cleaning import parse_published_at


def purged_chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not (0.0 < train_frac < 1.0) or not (0.0 < val_frac < 1.0) or (train_frac + val_frac >= 1.0):
        raise ValueError("Invalid split fractions: require 0 < train_frac, val_frac and sum < 1.")

    work = df.copy()
    work["_dt"] = parse_published_at(work["published_at"])
    if work["_dt"].isna().any():
        raise ValueError("Dataset contains unparseable timestamps in published_at.")

    work = work.sort_values(by=["_dt", "property_group_id"], kind="stable").reset_index(drop=True)
    n_total = len(work)

    idx_train_end = int(n_total * train_frac) - 1
    idx_val_end = int(n_total * (train_frac + val_frac)) - 1

    t_train_end = work.loc[idx_train_end, "_dt"]
    t_val_end = work.loc[idx_val_end, "_dt"]

    raw_train = work.loc[work["_dt"] <= t_train_end].copy()
    raw_val = work.loc[(work["_dt"] > t_train_end) & (work["_dt"] <= t_val_end)].copy()
    raw_test = work.loc[work["_dt"] > t_val_end].copy()

    # Cross-split group purging to prevent late reposts of earlier properties leaking forward
    train_groups = set(raw_train["property_group_id"].astype(str))
    val_purge_mask = raw_val["property_group_id"].astype(str).isin(train_groups)
    val_df = raw_val.loc[~val_purge_mask].copy()

    train_val_groups = train_groups | set(val_df["property_group_id"].astype(str))
    test_purge_mask = raw_test["property_group_id"].astype(str).isin(train_val_groups)
    test_df = raw_test.loc[~test_purge_mask].copy()

    train_dt_min, train_dt_max = raw_train["_dt"].min(), raw_train["_dt"].max()
    val_dt_min, val_dt_max = val_df["_dt"].min(), val_df["_dt"].max()
    test_dt_min, test_dt_max = test_df["_dt"].min(), test_df["_dt"].max()

    # Mathematical assertions for strict leakage prevention
    assert train_dt_max < val_dt_min, "Temporal leakage detected between Train and Validation!"
    assert val_dt_max < test_dt_min, "Temporal leakage detected between Validation and Test!"

    g_tr = set(raw_train["property_group_id"])
    g_va = set(val_df["property_group_id"])
    g_te = set(test_df["property_group_id"])
    assert g_tr.isdisjoint(g_va), "Group leakage detected between Train and Validation!"
    assert g_tr.isdisjoint(g_te), "Group leakage detected between Train and Test!"
    assert g_va.isdisjoint(g_te), "Group leakage detected between Validation and Test!"

    train_out = raw_train.drop(columns=["_dt"]).reset_index(drop=True)
    val_out = val_df.drop(columns=["_dt"]).reset_index(drop=True)
    test_out = test_df.drop(columns=["_dt"]).reset_index(drop=True)

    summary: dict[str, Any] = {
        "input_rows": n_total,
        "train_rows": int(len(train_out)),
        "val_rows": int(len(val_out)),
        "test_rows": int(len(test_out)),
        "purged_from_val": int(val_purge_mask.sum()),
        "purged_from_test": int(test_purge_mask.sum()),
        "total_retained_rows": int(len(train_out) + len(val_out) + len(test_out)),
        "train_period": [train_dt_min.isoformat(), train_dt_max.isoformat()],
        "val_period": [val_dt_min.isoformat(), val_dt_max.isoformat()],
        "test_period": [test_dt_min.isoformat(), test_dt_max.isoformat()],
        "train_type_counts": {
            str(k): int(v) for k, v in train_out["property_type_name"].value_counts().items()
        },
        "val_type_counts": {
            str(k): int(v) for k, v in val_out["property_type_name"].value_counts().items()
        },
        "test_type_counts": {
            str(k): int(v) for k, v in test_out["property_type_name"].value_counts().items()
        },
        "train_districts": int(train_out["district_name"].nunique()),
        "val_districts": int(val_out["district_name"].nunique()),
        "test_districts": int(test_out["district_name"].nunique()),
    }
    return train_out, val_out, test_out, summary
