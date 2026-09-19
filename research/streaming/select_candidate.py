"""Freeze development selection before opening confirmation seasons."""
from evaluate import *
from itertools import product

def select():
    d=pd.read_csv(OUT/'all_predictions.csv');assert d.season.max()<=2023, 'Selection may only see development years'
    choices={};rows=[]
    excluded={'season','week','game_id','team','y','position','player_id'}
    for pos,p in d.groupby('position'):
        base=metrics(p,p.v42);options=[('v42',{'v42':1.})]
        cols=[c for c in p if c not in excluded and c!='v42' and p[c].notna().all()]
        for c in cols:
            for w in [.25,.5,.75,1.]:options.append((f'{c}@{w}',{'v42':1-w,c:w}))
        if pos=='K':
            for a,b in product([.25,.5,.75],repeat=2):
                if a+b<=1:options.append((f'weighted_share+recent:{a},{b}',{'v42':1-a-b,'k_weighted_share':a,'k_recent_skill1.0':b}))
        valid=[]
        for name,weights in options:
            pred=sum(p[c]*w for c,w in weights.items());m=metrics(p,pred)
            eligible=(m['mae']<=base['mae']+.005 and m['weekly_spearman']>=base['weekly_spearman'] and m['top3']>=base['top3']-.05)
            row={'position':pos,'name':name,'eligible':eligible,**m};rows.append(row)
            if eligible:valid.append((m['rmse'],name,weights,m))
        _,name,weights,m=min(valid,key=lambda x:x[0])
        choices[pos]={'name':name,'weights':weights,'development_metrics':m,'baseline_metrics':base}
    manifest={'development_seasons':[2021,2022,2023],'confirmation_seasons':[2024,2025],
      'evaluation':'Annual expanding training: fit all seasons strictly before the predicted season; inputs update from prior games.',
      'selection':'Lowest development RMSE subject to MAE <= baseline + .005, weekly Spearman >= baseline, and top3 points >= baseline - .05.',
      'confirmation_promotion':'Candidate must improve pooled RMSE and MAE, must not reduce weekly Spearman, and must not lose more than 0.1 top3 points. Report each year and week-block bootstrap uncertainty; no retuning on confirmation.',
      'caveat':'2024/2025 were examined during earlier project development; these are retrospective confirmation seasons, not pristine holdouts.',
      'candidate_variants_including_blends':len(rows),'selected':choices}
    (OUT/'selection_lock.json').write_text(json.dumps(manifest,indent=2)+'\n')
    pd.DataFrame(rows).to_csv(OUT/'development_candidates.csv',index=False)
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':select()
