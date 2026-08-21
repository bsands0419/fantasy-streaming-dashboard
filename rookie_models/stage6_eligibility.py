from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_MODEL_OUTCOMES = ["primary_ppg", "hit3", "star3"]


def completed_outcome_mask(df: pd.DataFrame) -> pd.Series:
    """Rows allowed to influence Stage 6 model development.

    A player is development-eligible only when the three-year NFL outcome window
    used by the project is complete and valid. Prediction-only prospects may be
    scored later, but they cannot affect fitting, feature selection, tuning,
    calibration, thresholds, or feature-family promotion decisions.
    """
    mask = pd.Series(True, index=df.index)
    if "target_valid" in df.columns:
        mask &= pd.to_numeric(df["target_valid"], errors="coerce").eq(1)
    for c in REQUIRED_MODEL_OUTCOMES:
        if c not in df.columns:
            mask &= False
        else:
            mask &= pd.to_numeric(df[c], errors="coerce").notna()
    return mask


def annotate_stage6_eligibility(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    eligible = completed_outcome_mask(out)
    out["stage6_model_development_eligible"] = eligible.astype(int)
    out["stage6_prediction_only"] = (~eligible).astype(int)

    reasons = np.full(len(out), "complete_three_year_outcomes", dtype=object)
    if "target_valid" in out.columns:
        invalid = ~pd.to_numeric(out["target_valid"], errors="coerce").eq(1)
        reasons[invalid.to_numpy()] = "target_invalid_or_incomplete"
    for c in REQUIRED_MODEL_OUTCOMES:
        if c not in out.columns:
            reasons[:] = f"missing_required_outcome_column:{c}"
            break
        missing = pd.to_numeric(out[c], errors="coerce").isna().to_numpy()
        reasons[missing] = f"incomplete_three_year_outcome:{c}"
    out["stage6_eligibility_reason"] = reasons
    return out


def development_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[completed_outcome_mask(df)].copy()


def prediction_only_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[~completed_outcome_mask(df)].copy()


def assert_development_only(df: pd.DataFrame) -> None:
    bad = ~completed_outcome_mask(df)
    if bad.any():
        cols = [c for c in ["season", "position", "pfr_name", "player_name"] if c in df.columns]
        sample = df.loc[bad, cols].head(10).to_dict("records")
        raise RuntimeError(
            "Stage 6 development set contains players without complete three-year NFL outcomes. "
            f"Example rows: {sample}"
        )
