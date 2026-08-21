from pathlib import Path
import pandas as pd

import modeling as b
import stage2 as s2
import stage6_sos_ablation as sosmod

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_stage6_sos"
FINAL_HISTORY = OUT / "teamrankings_sos_final_history.csv"
HIST_POOL = ROOT / "results_v3" / "prospect_pool_v3.csv"
CUR_POOL = ROOT / "results_2026" / "prospect_pool_2026.csv"


def main():
    if not FINAL_HISTORY.exists():
        raise RuntimeError("Run stage6_sos_final_history.py first")

    sos = pd.read_csv(FINAL_HISTORY, low_memory=False)
    pa = b.load_cfb_table("passing")
    ru = b.load_cfb_table("rushing")
    re_ = b.load_cfb_table("receiving")
    team = b.load_cfb_table("team_summaries")
    pa, ru, re_ = s2.prep_college(pa, ru, re_, team)

    audits = []
    pa, a = sosmod.attach_sos(pa, team, sos, "passing"); audits.append(a)
    ru, a = sosmod.attach_sos(ru, team, sos, "rushing"); audits.append(a)
    re_, a = sosmod.attach_sos(re_, team, sos, "receiving"); audits.append(a)
    pd.concat(audits, ignore_index=True).to_csv(OUT / "team_name_match_audit_final_dates.csv", index=False)

    hist = pd.read_csv(HIST_POOL, low_memory=False)
    cur = pd.read_csv(CUR_POOL, low_memory=False)
    hist["season"] = pd.to_numeric(hist["season"], errors="coerce")
    cur["season"] = pd.to_numeric(cur["season"], errors="coerce")
    hist = hist[hist["season"].between(2014, 2023)].copy()
    cur = cur[cur["season"].eq(2026)].copy()

    needed = ["season", "pfr_id", "pfr_name", "position"]
    master = pd.concat([hist[needed], cur[needed]], ignore_index=True).drop_duplicates()
    sp = sosmod.aggregate_sos_profiles(master, pa, ru, re_)
    sp.to_csv(OUT / "prospect_teamrankings_sos_features_final_dates.csv", index=False)

    cov = []
    sos_cols = [c for c in sp.columns if c.startswith("tr_tr_sos_")]
    for pos, g in sp.groupby("position"):
        for era, q in [("historical_2014_2023", g[g.season.between(2014, 2023)]), ("prediction_2026", g[g.season.eq(2026)])]:
            row = {"position": pos, "era": era, "n": len(q)}
            for c in sos_cols:
                row[f"{c}_nonmissing"] = int(pd.to_numeric(q[c], errors="coerce").notna().sum())
            cov.append(row)
    pd.DataFrame(cov).to_csv(OUT / "prospect_teamrankings_sos_coverage_final_dates.csv", index=False)
    print(pd.DataFrame(cov).to_string(index=False))


if __name__ == "__main__":
    main()
