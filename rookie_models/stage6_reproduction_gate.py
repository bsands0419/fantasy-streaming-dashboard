from __future__ import annotations

import argparse
import base64
import gzip
import io
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CANON_REG = ROOT / "results_stage6_newmodels" / "advanced_selected_validation_regression_oof.csv"
CANON_CLF_B64 = ROOT / "results_stage6_newmodels" / "advanced_selected_validation_classification_oof.b64"
DEFAULT_REG_CAND = ROOT / "results_stage6_newmodels" / "original_fast" / "selected_validation_oof.csv"
DEFAULT_CLF_CAND = ROOT / "results_stage6_newmodels" / "original_classifiers_fast" / "selected_validation_oof.csv"
OUT = ROOT / "results_stage6_newmodels" / "reproduction_gate"
TOL = 1e-8


def decode_b64_csv(path: Path) -> pd.DataFrame:
    raw = base64.b64decode(path.read_text().strip())
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return pd.read_csv(io.BytesIO(raw))


def first_present(df: pd.DataFrame, names: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def normalize(df: pd.DataFrame, task: str) -> pd.DataFrame:
    out = df.copy()
    season = first_present(out, ["season", "draft_year"])
    pos = first_present(out, ["position", "pos"])
    name = first_present(out, ["pfr_name", "name", "player"])
    pred = first_present(out, ["pred", "prediction", "prob", "probability", "original_fast_prob"])
    y = first_present(out, ["y", "actual", "target_value"])
    target = first_present(out, ["target", "label"])
    if not all([season, pos, name, pred]):
        raise ValueError(f"{task}: missing required columns. columns={list(out.columns)}")
    ren = {season: "season", pos: "position", name: "pfr_name", pred: "pred"}
    if y:
        ren[y] = "y"
    if target:
        ren[target] = "target"
    out = out.rename(columns=ren)
    out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
    out["position"] = out["position"].astype(str).str.upper().str.strip()
    out["pfr_name"] = out["pfr_name"].astype(str).str.strip()
    out["pred"] = pd.to_numeric(out["pred"], errors="coerce")
    if "y" in out:
        out["y"] = pd.to_numeric(out["y"], errors="coerce")
    if task == "classification":
        if "target" not in out:
            raise ValueError("classification: candidate/canonical target column is required")
        out["target"] = out["target"].astype(str).str.lower().str.strip()
    return out


def compare(canonical: pd.DataFrame, candidate: pd.DataFrame, task: str, tol: float) -> tuple[pd.DataFrame, dict]:
    c = normalize(canonical, task)
    x = normalize(candidate, task)
    keys = ["season", "position", "pfr_name"] + (["target"] if task == "classification" else [])
    c = c.drop_duplicates(keys, keep="last")
    x = x.drop_duplicates(keys, keep="last")
    m = c.merge(x, on=keys, how="outer", suffixes=("_canonical", "_candidate"), indicator=True)
    both = m[m["_merge"].eq("both")].copy()
    both["abs_pred_diff"] = (both["pred_canonical"] - both["pred_candidate"]).abs()
    y_diff = np.nan
    if "y_canonical" in both and "y_candidate" in both:
        yy = both[["y_canonical", "y_candidate"]].dropna()
        if len(yy):
            y_diff = float((yy["y_canonical"] - yy["y_candidate"]).abs().max())
    max_diff = float(both["abs_pred_diff"].max()) if len(both) else np.inf
    mean_diff = float(both["abs_pred_diff"].mean()) if len(both) else np.inf
    rmse_diff = float(np.sqrt(np.mean(np.square(both["pred_canonical"] - both["pred_candidate"])))) if len(both) else np.inf
    summary = {
        "task": task,
        "canonical_rows": int(len(c)),
        "candidate_rows": int(len(x)),
        "matched_rows": int(len(both)),
        "missing_candidate_rows": int((m["_merge"] == "left_only").sum()),
        "extra_candidate_rows": int((m["_merge"] == "right_only").sum()),
        "max_abs_prediction_diff": max_diff,
        "mean_abs_prediction_diff": mean_diff,
        "rmse_prediction_diff": rmse_diff,
        "max_abs_target_diff": y_diff,
        "tolerance": tol,
    }
    summary["exact_cohort_match"] = (
        summary["canonical_rows"] == summary["candidate_rows"] == summary["matched_rows"]
        and summary["missing_candidate_rows"] == 0
        and summary["extra_candidate_rows"] == 0
    )
    summary["prediction_match"] = bool(np.isfinite(max_diff) and max_diff <= tol)
    summary["target_match"] = bool(pd.isna(y_diff) or y_diff <= tol)
    summary["gate_pass"] = bool(summary["exact_cohort_match"] and summary["prediction_match"] and summary["target_match"])
    return m, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Strict Stage 6 OOF reproduction gate. This is an audit only, never a selection/tuning step.")
    ap.add_argument("--reg-candidate", type=Path, default=DEFAULT_REG_CAND)
    ap.add_argument("--clf-candidate", type=Path, default=DEFAULT_CLF_CAND)
    ap.add_argument("--tolerance", type=float, default=TOL)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if not CANON_REG.exists() or not CANON_CLF_B64.exists():
        raise FileNotFoundError("Canonical Stage 6 OOF artifacts are missing. Refitting is forbidden.")

    summaries = []
    details = []
    if args.reg_candidate.exists():
        m, s = compare(pd.read_csv(CANON_REG), pd.read_csv(args.reg_candidate), "regression", args.tolerance)
        m.insert(0, "task", "regression")
        details.append(m)
        summaries.append(s)
    else:
        summaries.append({"task": "regression", "gate_pass": False, "reason": f"candidate missing: {args.reg_candidate}"})

    if args.clf_candidate.exists():
        canon_clf = decode_b64_csv(CANON_CLF_B64)
        m, s = compare(canon_clf, pd.read_csv(args.clf_candidate), "classification", args.tolerance)
        m.insert(0, "task", "classification")
        details.append(m)
        summaries.append(s)
    else:
        summaries.append({"task": "classification", "gate_pass": False, "reason": f"candidate missing: {args.clf_candidate}"})

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT / "reproduction_summary.csv", index=False)
    if details:
        pd.concat(details, ignore_index=True).to_csv(OUT / "reproduction_row_audit.csv", index=False)

    passed = bool(len(summary_df) == 2 and summary_df["gate_pass"].fillna(False).all())
    report = [
        "# Stage 6 OOF Reproduction Gate",
        "",
        f"Overall gate: {'PASS' if passed else 'FAIL'}",
        "",
        "This gate compares a candidate implementation against the frozen canonical 2019-2022 OOF predictions row by row. It does not select or tune models. 2023 and 2024-2026 outcomes are not used by this gate.",
        "",
        "A candidate may be used for the final 2014-2023 refit only if both regression and classification cohorts match exactly and maximum absolute prediction drift is within the configured numerical tolerance.",
        "",
        summary_df.to_markdown(index=False),
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(report))
    print(summary_df.to_string(index=False))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
