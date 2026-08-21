from pathlib import Path
import json
import pandas as pd
import stage2 as s2

ROOT=Path(__file__).resolve().parent
POOL=ROOT/'results_v4'/'prospect_pool_v4.csv'
OUT=ROOT/'results_stage6_newmodels'
OUT.mkdir(parents=True,exist_ok=True)
d=pd.read_csv(POOL,low_memory=False)
rows=[]
for pos in ['QB','RB','WR','TE']:
    g=s2.feature_groups(d,pos)
    for name,features in g.items():
        usable=[c for c in features if c in d.columns]
        observed=[c for c in usable if pd.to_numeric(d.loc[d.position.eq(pos),c],errors='coerce').notna().any()]
        rows.append({'position':pos,'family':name,'listed_n':len(features),'usable_n':len(usable),'observed_n':len(observed),'features':'|'.join(observed)})
pd.DataFrame(rows).to_csv(OUT/'original_feature_family_audit.csv',index=False)
(OUT/'original_feature_family_keys.json').write_text(json.dumps({p:list(s2.feature_groups(d,p).keys()) for p in ['QB','RB','WR','TE']},indent=2))
print(pd.DataFrame(rows)[['position','family','observed_n']].to_string(index=False))
