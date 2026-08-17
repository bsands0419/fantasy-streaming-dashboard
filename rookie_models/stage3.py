from __future__ import annotations

import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, XGBClassifier

import modeling as b
import stage2 as s2

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_v3"
MODELS = ROOT / "trained_v3"
OUT.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

POSITIONS = ["QB", "RB", "WR", "TE"]
DEV_YEARS = list(range(2013, 2019))
VALID_YEARS = list(range(2019, 2023))
FINAL_TEST_YEAR = 2023
TRAIN_END = 2023
PREDICT_YEAR = 2026
SEED = 20260816

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 dynasty-rookie-research/3.0"})


def team_norm(x):
    if pd.isna(x):
        return ""
    x = str(x).upper().strip()
    return {
        "JAC": "JAX", "WSH": "WAS", "OAK": "LV", "SD": "LAC",
        "STL": "LAR", "LA": "LAR"
    }.get(x, x)


def add_draft_team(prof, draft):
    d = draft[["season", "pfr_id", "team"]].drop_duplicates(["season", "pfr_id"]).copy()
    d = d.rename(columns={"team": "draft_team"})
    p = prof.merge(d, on=["season", "pfr_id"], how="left")
    p["draft_team_norm"] = p["draft_team"].map(team_norm)
    return p


def landing_table(weekly):
    w = weekly.copy()
    w["season"] = pd.to_numeric(w["season"], errors="coerce")
    if "season_type" in w.columns:
        w = w[w["season_type"].astype(str).str.upper().eq("REG")].copy()
    if "recent_team" not in w.columns or "position" not in w.columns:
        return pd.DataFrame()
    w["team"] = w["recent_team"].map(team_norm)
    w["pos"] = w["position"].astype(str).str.upper().replace({"HB": "RB", "FB": "RB"})
    idc = b.first_existing(w, ["player_id", "gsis_id"])
    if not idc:
        return pd.DataFrame()
    for c in ["fantasy_points_ppr", "fantasy_points", "attempts", "carries", "targets", "receptions"]:
        if c in w.columns:
            w[c] = pd.to_numeric(w[c], errors="coerce").fillna(0)
    fp = "fantasy_points_ppr" if "fantasy_points_ppr" in w.columns else "fantasy_points"
    if fp not in w.columns:
        w[fp] = 0.0
    for c in ["attempts", "carries", "targets", "receptions"]:
        if c not in w.columns:
            w[c] = 0.0

    pl = w[w["pos"].isin(POSITIONS)].groupby(
        ["season", "team", "pos", idc], dropna=False
    ).agg(
        ppr=(fp, "sum"),
        attempts=("attempts", "sum"),
        carries=("carries", "sum"),
        targets=("targets", "sum"),
        receptions=("receptions", "sum"),
    ).reset_index()

    rows = []
    for (season, team), g in pl.groupby(["season", "team"]):
        rec = {"prior_season": int(season), "draft_team_norm": team}
        rec["landing_team_ppr"] = float(g.ppr.sum())
        rec["landing_team_targets"] = float(g.targets.sum())
        rec["landing_team_carries"] = float(g.carries.sum())
        q = g[g.pos == "QB"]
        rec["landing_team_pass_attempts"] = float(q.attempts.sum())
        rec["landing_team_qb_ppr"] = float(q.ppr.sum())
        for pos in POSITIONS:
            z = g[g.pos == pos]
            pre = f"landing_{pos.lower()}"
            rec[f"{pre}_ppr"] = float(z.ppr.sum())
            rec[f"{pre}_attempts"] = float(z.attempts.sum())
            rec[f"{pre}_carries"] = float(z.carries.sum())
            rec[f"{pre}_targets"] = float(z.targets.sum())
            rec[f"{pre}_receptions"] = float(z.receptions.sum())
            for stat in ["ppr", "attempts", "carries", "targets"]:
                total = float(z[stat].sum())
                top = float(z[stat].max()) if len(z) else 0.0
                rec[f"{pre}_top_{stat}_share"] = top / total if total > 0 else np.nan
            rec[f"{pre}_players_used"] = float(z[idc].nunique())
        rows.append(rec)
    return pd.DataFrame(rows)


def add_landing_features(prof, draft, weekly):
    p = add_draft_team(prof, draft)
    lt = landing_table(weekly)
    if lt.empty:
        return p
    p["prior_season"] = pd.to_numeric(p["season"], errors="coerce") - 1
    p = p.merge(lt, on=["prior_season", "draft_team_norm"], how="left")
    p = p.drop(columns=["prior_season"], errors="ignore")
    for pos in POSITIONS:
        pre = f"landing_{pos.lower()}"
        if f"{pre}_top_ppr_share" in p:
            p[f"{pre}_ppr_concentration"] = p[f"{pre}_top_ppr_share"]
        if f"{pre}_top_targets_share" in p:
            p[f"{pre}_target_concentration"] = p[f"{pre}_top_targets_share"]
    return p


def college_only_features(df, pos):
    base = s2.feature_groups(df, pos)["college_only"]
    return [c for c in base if not c.startswith("landing_") and not c.startswith("scout_")]


def add_scouting_surprise(df):
    """Leakage-safe draft-capital surprise: prior-year college data predicts expected log pick."""
    out = df.copy()
    out["scout_expected_log_pick"] = np.nan
    out["scout_boost"] = np.nan
    years = sorted(int(x) for x in out["season"].dropna().unique())
    for pos in POSITIONS:
        posmask = out.position.eq(pos)
        feats = college_only_features(out[posmask], pos)
        if not feats:
            continue
        for y in years:
            trmask = posmask & (out.season < y) & pd.to_numeric(out.draft_pick, errors="coerce").notna()
            temask = posmask & out.season.eq(y) & pd.to_numeric(out.draft_pick, errors="coerce").notna()
            if trmask.sum() < 50 or temask.sum() == 0:
                continue
            m = Pipeline([
                ("imp", SimpleImputer(strategy="median", add_indicator=True)),
                ("sc", StandardScaler()),
                ("m", Ridge(alpha=30.0)),
            ])
            target = np.log(pd.to_numeric(out.loc[trmask, "draft_pick"], errors="coerce").clip(lower=1))
            m.fit(out.loc[trmask, feats], target)
            pred = m.predict(out.loc[temask, feats])
            actual = np.log(pd.to_numeric(out.loc[temask, "draft_pick"], errors="coerce").clip(lower=1))
            out.loc[temask, "scout_expected_log_pick"] = pred
            out.loc[temask, "scout_boost"] = pred - actual.to_numpy()
    return out


def groups_v3(df, pos):
    g = s2.feature_groups(df, pos)
    landing = [c for c in df.columns if c.startswith("landing_") and pd.api.types.is_numeric_dtype(df[c])]
    scout = [c for c in ["scout_expected_log_pick", "scout_boost"] if c in df.columns]
    g["capital_core_landing"] = list(dict.fromkeys(g["capital_core"] + landing))
    g["capital_production_landing"] = list(dict.fromkeys(g["capital_production"] + landing))
    g["full_landing"] = list(dict.fromkeys(g["full"] + landing))
    g["full_landing_scout"] = list(dict.fromkeys(g["full"] + landing + scout))
    g["capital_prod_context_landing_scout"] = list(dict.fromkeys(g["capital_prod_context"] + landing + scout))
    return g


def models_v3():
    base = s2.model_candidates()
    base.update({
        "xgb_shallow": XGBRegressor(
            n_estimators=550, max_depth=2, learning_rate=0.025,
            min_child_weight=8, subsample=0.85, colsample_bytree=0.75,
            reg_alpha=0.5, reg_lambda=10.0, objective="reg:squarederror",
            n_jobs=2, random_state=SEED
        ),
        "xgb_reg": XGBRegressor(
            n_estimators=450, max_depth=3, learning_rate=0.025,
            min_child_weight=12, subsample=0.85, colsample_bytree=0.65,
            reg_alpha=1.0, reg_lambda=14.0, objective="reg:squarederror",
            n_jobs=2, random_state=SEED + 1
        ),
    })
    return base


def reg_metrics(y, p):
    return s2.reg_metrics(y, p)


def wf(d, feats, model, years, target="primary_ppg"):
    return s2.wf_predict(d, feats, model, years, target)


def score(m, ysd):
    return s2.selection_score(m, ysd)


def classifier_candidates():
    return {
        "logit": Pipeline([
            ("imp", SimpleImputer(strategy="median", add_indicator=True)),
            ("sc", StandardScaler()),
            ("m", LogisticRegression(C=0.25, max_iter=4000, class_weight="balanced")),
        ]),
        "xgb_cls": Pipeline([
            ("imp", SimpleImputer(strategy="median", add_indicator=True)),
            ("m", XGBClassifier(
                n_estimators=350, max_depth=2, learning_rate=0.025,
                min_child_weight=8, subsample=0.85, colsample_bytree=0.7,
                reg_alpha=0.5, reg_lambda=10.0, objective="binary:logistic",
                eval_metric="logloss", n_jobs=2, random_state=SEED
            )),
        ]),
    }


def classifier_oof(d, feats, model, years):
    rows = []
    for y in years:
        tr = d[d.season < y]
        te = d[d.season == y]
        if te.empty or tr.hit3.nunique() < 2:
            continue
        m = clone(model)
        m.fit(tr[feats], tr.hit3)
        q = te[["season", "pfr_name", "hit3"]].copy()
        q["prob"] = m.predict_proba(te[feats])[:, 1]
        rows.append(q)
    z = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if z.empty or z.hit3.nunique() < 2:
        return z, {"auc": np.nan, "brier": np.nan, "n": len(z)}
    return z, {
        "auc": roc_auc_score(z.hit3, z.prob),
        "brier": brier_score_loss(z.hit3, z.prob),
        "n": len(z),
    }


def choose_classifier(d, feats):
    rows = []
    stores = {}
    for name, model in classifier_candidates().items():
        z, met = classifier_oof(d, feats, model, VALID_YEARS)
        rows.append({"model": name, **met})
        stores[name] = z
    tab = pd.DataFrame(rows)
    tab["score"] = -tab["brier"].fillna(9) + 0.15 * tab["auc"].fillna(0)
    tab = tab.sort_values("score", ascending=False)
    name = tab.iloc[0].model
    return name, classifier_candidates()[name], tab, stores[name]


def evaluate_position(df, pos):
    d = df[(df.position == pos) & (df.season <= TRAIN_END) & (df.target_valid == 1)].copy()
    groups = groups_v3(d, pos)
    models = models_v3()
    ysd = float(d.loc[d.season < 2019, "primary_ppg"].std()) or 1.0

    devrows, devstore = [], {}
    for fs, feats in groups.items():
        if not feats:
            continue
        for mn, model in models.items():
            z = wf(d, feats, model, DEV_YEARS)
            if z.empty:
                continue
            met = reg_metrics(z.primary_ppg, z.pred)
            met.update({"feature_set": fs, "model": mn, "n_features": len(feats), "score": score(met, ysd)})
            devrows.append(met)
            devstore[(fs, mn)] = z
    dev = pd.DataFrame(devrows).sort_values("score", ascending=False)

    valrows, valstore = [], {}
    for _, row in dev.head(16).iterrows():
        fs, mn = row.feature_set, row.model
        z = wf(d, groups[fs], models[mn], VALID_YEARS)
        if z.empty:
            continue
        met = reg_metrics(z.primary_ppg, z.pred)
        met.update({"feature_set": fs, "model": mn, "n_features": len(groups[fs]), "score": score(met, ysd)})
        valrows.append(met)
        valstore[(fs, mn)] = z
    val = pd.DataFrame(valrows).sort_values("score", ascending=False)

    best = val.iloc[0]
    afs, amn = best.feature_set, best.model
    afeats, am = groups[afs], models[amn]
    bestz = valstore[(afs, amn)].copy()
    bestmet = reg_metrics(bestz.primary_ppg, bestz.pred)
    bestscore = score(bestmet, ysd)
    blend = {"weight": 1.0, "other": None}

    for _, row in val.head(8).iloc[1:].iterrows():
        key = (row.feature_set, row.model)
        z = bestz.merge(valstore[key][["season", "pfr_name", "pred"]], on=["season", "pfr_name"], suffixes=("_a", "_b"))
        for weight in [0.2, 0.35, 0.5, 0.65, 0.8]:
            pred = weight * z.pred_a + (1 - weight) * z.pred_b
            met = reg_metrics(z.primary_ppg, pred)
            sc = score(met, ysd)
            if sc > bestscore + 0.002:
                bestscore = sc
                bestmet = met
                blend = {"weight": weight, "other": key}

    bfeats = bmodel = None
    if blend["other"]:
        bfs, bmn = blend["other"]
        bfeats, bmodel = groups[bfs], models[bmn]
        z = bestz.merge(valstore[(bfs, bmn)][["season", "pfr_name", "pred"]], on=["season", "pfr_name"], suffixes=("_a", "_b"))
        z["pred"] = blend["weight"] * z.pred_a + (1 - blend["weight"]) * z.pred_b
        valfinal = z[["season", "pfr_name", "primary_ppg", "hit3", "pred"]]
    else:
        valfinal = bestz
    valmet = reg_metrics(valfinal.primary_ppg, valfinal.pred)

    cap = groups["capital"]
    capz = wf(d, cap, models["ridge10"], VALID_YEARS)
    capmet = reg_metrics(capz.primary_ppg, capz.pred)

    tr = d[d.season < FINAL_TEST_YEAR]
    te = d[d.season == FINAL_TEST_YEAR]
    ma = clone(am)
    ma.fit(tr[afeats], tr.primary_ppg)
    fp = ma.predict(te[afeats])
    mb = None
    if bmodel is not None:
        mb = clone(bmodel)
        mb.fit(tr[bfeats], tr.primary_ppg)
        fp = blend["weight"] * fp + (1 - blend["weight"]) * mb.predict(te[bfeats])
    final = te[["season", "pfr_name", "primary_ppg", "hit3"]].copy()
    final["pred"] = fp
    finalmet = reg_metrics(final.primary_ppg, final.pred)

    cm = clone(models["ridge10"])
    cm.fit(tr[cap], tr.primary_ppg)
    cf = te[["season", "pfr_name", "primary_ppg"]].copy()
    cf["pred"] = cm.predict(te[cap])
    capfinal = reg_metrics(cf.primary_ppg, cf.pred)

    cls_name, cls_template, clstab, clsoof = choose_classifier(d, afeats)
    hmet = {
        "auc": roc_auc_score(clsoof.hit3, clsoof.prob) if len(clsoof) and clsoof.hit3.nunique() > 1 else np.nan,
        "brier": brier_score_loss(clsoof.hit3, clsoof.prob) if len(clsoof) else np.nan,
        "n": len(clsoof),
    }
    htr = d[d.season < FINAL_TEST_YEAR]
    hclf = clone(cls_template)
    hclf.fit(htr[afeats], htr.hit3)
    hf = te[["season", "pfr_name", "hit3"]].copy()
    hf["prob"] = hclf.predict_proba(te[afeats])[:, 1]
    hfmet = {
        "auc": roc_auc_score(hf.hit3, hf.prob) if len(hf) and hf.hit3.nunique() > 1 else np.nan,
        "brier": brier_score_loss(hf.hit3, hf.prob) if len(hf) else np.nan,
        "n": len(hf),
    }

    fitted = {}
    for target in ["primary_ppg", "peak3_ppg", "rookie_ppg", "avg3_ppg", "total3_ppr"]:
        train = d[(d.season <= TRAIN_END) & d[target].notna()]
        a = clone(am)
        a.fit(train[afeats], train[target])
        bb = None
        if bmodel is not None:
            bb = clone(bmodel)
            bb.fit(train[bfeats], train[target])
        fitted[target] = (a, bb)

    clf = clone(cls_template)
    clf.fit(d[afeats], d.hit3)
    absres = np.abs(valfinal.primary_ppg.to_numpy() - valfinal.pred.to_numpy())
    q80 = float(np.quantile(absres, 0.80))
    q90 = float(np.quantile(absres, 0.90))

    job = {
        "position": pos, "features": afeats, "model_name": amn, "features_b": bfeats,
        "blend": blend, "fitted": fitted, "classifier_name": cls_name, "classifier": clf,
        "interval_abs_error_q80": q80, "interval_abs_error_q90": q90,
    }
    joblib.dump(job, MODELS / f"{pos}.joblib")

    summary = {
        "position": pos, "n_train": len(d), "selected_feature_set": afs, "selected_model": amn,
        "n_features": len(afeats), "blend": str(blend), "classifier": cls_name,
        "validation": valmet, "capital_validation": capmet,
        "final_2023": finalmet, "capital_final_2023": capfinal,
        "hit_validation": hmet, "hit_final_2023": hfmet,
        "validation_mae_improvement_pct": 100 * (capmet["mae"] - valmet["mae"]) / capmet["mae"] if capmet["mae"] else np.nan,
        "validation_spearman_gain": valmet["spearman"] - capmet["spearman"],
        "final_mae_improvement_pct": 100 * (capfinal["mae"] - finalmet["mae"]) / capfinal["mae"] if capfinal["mae"] else np.nan,
        "final_spearman_gain": finalmet["spearman"] - capfinal["spearman"],
        "interval_abs_error_q80": q80, "interval_abs_error_q90": q90,
    }
    return summary, dev, val, valfinal, final, clstab, clsoof, hf, job


def predict_pair(a, bb, cur, job):
    pa = a.predict(cur[job["features"]])
    if bb is None:
        return pa
    pb = bb.predict(cur[job["features_b"]])
    w = job["blend"]["weight"]
    return w * pa + (1 - w) * pb


def pct(hist, vals):
    return s2.pct(hist, vals)


def current_rankings(df, jobs):
    out = []
    for pos, job in jobs.items():
        train = df[(df.position == pos) & (df.season <= TRAIN_END) & (df.target_valid == 1)]
        cur = df[(df.position == pos) & (df.season == PREDICT_YEAR)].copy()
        if cur.empty:
            continue
        mappings = [
            ("primary_ppg", "pred_best2of3_ppg"),
            ("peak3_ppg", "pred_peak3_ppg"),
            ("rookie_ppg", "pred_rookie_ppg"),
            ("avg3_ppg", "pred_avg3_ppg"),
            ("total3_ppr", "pred_total3_ppr"),
        ]
        for target, label in mappings:
            a, bb = job["fitted"][target]
            cur[label] = predict_pair(a, bb, cur, job)
        cur["hit_probability"] = job["classifier"].predict_proba(cur[job["features"]])[:, 1]
        cur["model_percentile"] = pct(train.primary_ppg, cur.pred_best2of3_ppg)
        cur["draft_capital_percentile"] = pct(
            -pd.to_numeric(train.draft_pick, errors="coerce"),
            -pd.to_numeric(cur.draft_pick, errors="coerce")
        )
        q80, q90 = job["interval_abs_error_q80"], job["interval_abs_error_q90"]
        cur["pred80_low"] = np.maximum(0, cur.pred_best2of3_ppg - q80)
        cur["pred80_high"] = cur.pred_best2of3_ppg + q80
        cur["pred90_low"] = np.maximum(0, cur.pred_best2of3_ppg - q90)
        cur["pred90_high"] = cur.pred_best2of3_ppg + q90
        cur["rank"] = cur.pred_best2of3_ppg.rank(ascending=False, method="first").astype(int)
        keep = [
            "position","rank","pfr_name","draft_team","draft_round","draft_pick",
            "pred_best2of3_ppg","pred_peak3_ppg","pred_rookie_ppg","pred_avg3_ppg","pred_total3_ppr",
            "pred80_low","pred80_high","pred90_low","pred90_high",
            "hit_probability","model_percentile","college_match","fuzzy_match","scout_boost"
        ]
        out.append(cur[[c for c in keep if c in cur.columns]].sort_values("rank"))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def fantasypros_adp(year, pos):
    url = f"https://www.fantasypros.com/nfl/adp/rookies-{pos.lower()}.php?year={year}"
    try:
        r = SESSION.get(url, timeout=30)
        r.raise_for_status()
    except Exception:
        return pd.DataFrame()
    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table")
    if table is None:
        return pd.DataFrame()
    headers = [th.get_text(" ", strip=True).lower() for th in table.find_all("th")]
    avg_idx = next((i for i, x in enumerate(headers) if x == "avg" or "avg" in x), None)
    rows = []
    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue
        a = tr.find("a", href=re.compile(r"/nfl/players/"))
        if not a:
            continue
        name = a.get_text(" ", strip=True)
        adp = np.nan
        if avg_idx is not None and avg_idx < len(tds):
            adp = pd.to_numeric(pd.Series([tds[avg_idx].get_text(" ", strip=True)]), errors="coerce").iloc[0]
        if not np.isfinite(adp):
            nums = []
            for td in tds:
                v = pd.to_numeric(pd.Series([td.get_text(" ", strip=True)]), errors="coerce").iloc[0]
                nums.append(v)
            finite = [x for x in nums if np.isfinite(x)]
            adp = finite[-1] if finite else np.nan
        rows.append({"season": int(year), "position": pos, "adp_name": name, "adp": adp, "name_norm": b.norm_name(name)})
    return pd.DataFrame(rows)


def market_benchmark(df, pred_by_pos):
    all_adp = []
    for pos in POSITIONS:
        for year in list(range(2019, 2024)) + [PREDICT_YEAR]:
            z = fantasypros_adp(year, pos)
            if not z.empty:
                all_adp.append(z)
    adp = pd.concat(all_adp, ignore_index=True) if all_adp else pd.DataFrame()
    if adp.empty:
        return adp, pd.DataFrame()

    hist = df[(df.season.between(2019, 2023)) & (df.target_valid == 1)].copy()
    hist["name_norm"] = hist.pfr_name.map(b.norm_name)
    pframes = []
    for pos, pp in pred_by_pos.items():
        q = pp.copy()
        q["position"] = pos
        q["name_norm"] = q.pfr_name.map(b.norm_name)
        pframes.append(q)
    preds = pd.concat(pframes, ignore_index=True) if pframes else pd.DataFrame()
    m = hist.merge(adp[adp.season <= 2023][["season","position","name_norm","adp"]], on=["season","position","name_norm"], how="inner")
    if not preds.empty:
        m = m.merge(preds[["season","position","name_norm","pred"]], on=["season","position","name_norm"], how="left")
    rows = []
    for pos in POSITIONS:
        z = m[m.position == pos].dropna(subset=["adp","primary_ppg"])
        if len(z) < 8:
            continue
        row = {
            "position":pos, "n":len(z),
            "market_spearman":spearmanr(-z.adp, z.primary_ppg).statistic,
            "market_pearson":pearsonr(-z.adp, z.primary_ppg).statistic,
            "model_spearman_same_rows":spearmanr(z.pred, z.primary_ppg).statistic if z.pred.notna().sum() >= 8 else np.nan,
            "model_pearson_same_rows":pearsonr(z.pred, z.primary_ppg).statistic if z.pred.notna().sum() >= 8 else np.nan,
        }
        rows.append(row)
    return adp, pd.DataFrame(rows)


def year_metrics(preds):
    rows=[]
    for pos,z in preds.items():
        for year,g in z.groupby("season"):
            m=reg_metrics(g.primary_ppg,g.pred)
            rows.append({"position":pos,"season":int(year),**m})
    return pd.DataFrame(rows)


def write_report(summaries, rankings, market, yearly):
    L = [
        "# Dynasty Rookie Prospect Models v3.0",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat()}",
        "",
        "Stage 3 was specified before inspecting the 2023 final-test result. It adds leakage-safe post-draft landing-spot context, a prior-year college-to-draft-capital scouting-surprise feature, regularized XGBoost candidates, classifier model selection, and empirical prediction intervals.",
        "",
        "The primary target remains the average of a prospect's two best full-schedule PPR point rates in his first three NFL seasons. Draft capital remains the mandatory baseline.",
        "",
        "## Validation protocol",
        "",
        "2013-2018 is development screening, 2019-2022 is walk-forward model selection, and the 2023 draft class is a one-shot frozen final test. No 2023 result is used to select features, algorithms, weights, or probability models.",
        "",
    ]
    for s in summaries:
        v=s["validation"]; c=s["capital_validation"]; f=s["final_2023"]; cf=s["capital_final_2023"]
        hv=s["hit_validation"]; hf=s["hit_final_2023"]
        L += [
            f"## {s['position']}", "",
            f"Selected {s['selected_feature_set']} / {s['selected_model']} with {s['n_features']} features; blend {s['blend']}; hit classifier {s['classifier']}.",
            "",
            f"2019-22 validation: MAE {v['mae']:.3f}, RMSE {v['rmse']:.3f}, Pearson {v['pearson']:.3f}, Spearman {v['spearman']:.3f}, R² {v['r2']:.3f}. Capital-only MAE {c['mae']:.3f}, Spearman {c['spearman']:.3f}.",
            f"Frozen 2023: MAE {f['mae']:.3f}, RMSE {f['rmse']:.3f}, Pearson {f['pearson']:.3f}, Spearman {f['spearman']:.3f}. Capital-only MAE {cf['mae']:.3f}, Spearman {cf['spearman']:.3f}.",
            f"Hit model: validation AUC {hv['auc']:.3f}, Brier {hv['brier']:.3f}; frozen 2023 AUC {hf['auc']:.3f}, Brier {hf['brier']:.3f}. Empirical absolute-error bands: 80% {s['interval_abs_error_q80']:.2f} PPG, 90% {s['interval_abs_error_q90']:.2f} PPG.",
            "",
        ]
    if not market.empty:
        L += ["## Historical rookie-ADP benchmark", "", "|Pos|N|Market Spearman|Model Spearman same rows|Delta|", "|---|---:|---:|---:|---:|"]
        for _,r in market.iterrows():
            delta=r.model_spearman_same_rows-r.market_spearman
            L.append(f"|{r.position}|{int(r.n)}|{r.market_spearman:.3f}|{r.model_spearman_same_rows:.3f}|{delta:+.3f}|")
        L.append("")
    if not rankings.empty:
        L += ["## 2026 post-draft rankings", ""]
        for pos in POSITIONS:
            z=rankings[rankings.position==pos].head(15)
            if z.empty: continue
            L += [f"### {pos}", "", "|Rk|Prospect|NFL team|Pick|Best-2/3 PPG|Peak|Rookie|Hit %|Hist pct|80% interval|",
                  "|---:|---|---|---:|---:|---:|---:|---:|---:|---|"]
            for _,r in z.iterrows():
                pick="" if pd.isna(r.draft_pick) else int(r.draft_pick)
                L.append(f"|{int(r['rank'])}|{r.pfr_name}|{r.get('draft_team','')}|{pick}|{r.pred_best2of3_ppg:.2f}|{r.pred_peak3_ppg:.2f}|{r.pred_rookie_ppg:.2f}|{100*r.hit_probability:.1f}|{r.model_percentile:.1f}|{r.pred80_low:.1f}-{r.pred80_high:.1f}|")
            L.append("")
    L += [
        "## Interpretation", "",
        "A more complex model is promoted only if pre-2023 validation supports it. The frozen 2023 column is confirmatory evidence, not a tuning target. Market comparisons use historical FantasyPros rookie ADP when the page can be fetched and matched; unavailable years or players are excluded rather than imputed."
    ]
    (OUT / "REPORT.md").write_text("\n".join(L))


def main():
    print("Loading cfbfastR/SportsDataverse...")
    pa=b.load_cfb_table("passing")
    ru=b.load_cfb_table("rushing")
    re=b.load_cfb_table("receiving")
    team=b.load_cfb_table("team_summaries")
    pa,ru,re=s2.prep_college(pa,ru,re,team)

    print("Loading nflverse...")
    draft,players,combine,weekly=b.load_nflverse()
    draft["season"]=pd.to_numeric(draft["season"],errors="coerce")
    draft=draft[draft.category.isin(POSITIONS)&draft.season.between(b.TRAIN_DRAFT_START,PREDICT_YEAR)].copy()

    prof=s2.build_profiles(draft,pa,ru,re)
    prof=s2.add_nfl_meta(prof,players,combine)
    prof=s2.build_targets(prof,weekly)
    prof=add_landing_features(prof,draft,weekly)
    prof=add_scouting_surprise(prof)
    prof.to_csv(OUT/"prospect_pool_v3.csv",index=False)

    match=prof.groupby("position").agg(
        n=("pfr_name","size"), college_match=("college_match","mean"),
        fuzzy_rate=("fuzzy_match","mean"), nfl_meta_match=("nfl_meta_match","mean")
    ).reset_index()
    match.to_csv(OUT/"match_audit.csv",index=False)

    summaries=[];jobs={};predhist={}
    for pos in POSITIONS:
        print("Stage3 fitting",pos)
        s,dev,val,valp,final,clstab,clsoof,hf,j=evaluate_position(prof,pos)
        summaries.append(s);jobs[pos]=j
        predhist[pos]=pd.concat([valp,final],ignore_index=True)
        dev.to_csv(OUT/f"{pos}_dev_grid.csv",index=False)
        val.to_csv(OUT/f"{pos}_validation_grid.csv",index=False)
        valp.to_csv(OUT/f"{pos}_validation_predictions.csv",index=False)
        final.to_csv(OUT/f"{pos}_final_2023_predictions.csv",index=False)
        clstab.to_csv(OUT/f"{pos}_classifier_grid.csv",index=False)
        clsoof.to_csv(OUT/f"{pos}_validation_hit_probs.csv",index=False)
        hf.to_csv(OUT/f"{pos}_final_2023_hit_probs.csv",index=False)

    rankings=current_rankings(prof,jobs)
    rankings.to_csv(OUT/"rookie_rankings_2026.csv",index=False)
    yearly=year_metrics(predhist)
    yearly.to_csv(OUT/"year_by_year_accuracy.csv",index=False)
    adp,market=market_benchmark(prof,predhist)
    adp.to_csv(OUT/"fantasypros_rookie_adp.csv",index=False)
    market.to_csv(OUT/"market_benchmark.csv",index=False)

    flat=[]
    for s in summaries:
        row={k:v for k,v in s.items() if not isinstance(v,dict)}
        for block in ["validation","capital_validation","final_2023","capital_final_2023","hit_validation","hit_final_2023"]:
            for k,v in s[block].items():
                row[f"{block}_{k}"]=v
        flat.append(row)
    pd.DataFrame(flat).to_csv(OUT/"model_summary.csv",index=False)
    write_report(summaries,rankings,market,yearly)
    meta={
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "stage3_frozen_before_stage2_2023_results":True,
        "development_years":DEV_YEARS,"validation_years":VALID_YEARS,
        "frozen_final_test_year":FINAL_TEST_YEAR,"prediction_year":PREDICT_YEAR,
        "summaries":summaries
    }
    (OUT/"manifest.json").write_text(json.dumps(meta,indent=2,default=float))
    print(json.dumps(meta,indent=2,default=float))


if __name__=="__main__":
    main()
