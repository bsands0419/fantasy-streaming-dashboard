from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results_stage6_sos'
OUT.mkdir(exist_ok=True)

for src,name in [(ROOT/'results_v3'/'prospect_pool_v3.csv','historical'),(ROOT/'results_2026'/'prospect_pool_2026.csv','current')]:
    d=pd.read_csv(src,low_memory=False)
    cols=[c for c in d.columns if any(k in c.lower() for k in ['school','college','team_name','display_name','location'])]
    idcols=[c for c in ['season','pfr_id','pfr_name','position'] if c in d.columns]
    print(name,cols)
    d[idcols+cols].head(25).to_csv(OUT/f'pool_school_probe_{name}.csv',index=False)
