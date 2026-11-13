"""Evaluate only the already locked challengers on later seasons."""
from evaluate import *

def confirm():
    lock=json.loads((OUT/'selection_lock.json').read_text());t,k=load_data();t,k=augment(t,k)
    rows=[];allpred=[]
    for pos,d,features,target,cls in [('DST',t,DST_V4_FEATURES,'dst_fantasy',ChampionshipDSTModelV42),('K',k,K_V42_FEATURES,'k_bucket_points',ChampionshipKickerModelV42)]:
        d['y']=d[target];cols=list(features)+[c for c in d if c.startswith(('new_','opp_new_'))]
        weights=lock['selected'][pos]['weights']
        for year in lock['confirmation_seasons']:
            tr=d[d.season<year];te=d[d.season==year]
            path=CACHE/f'baseline_{pos}_{year}.joblib'
            if path.exists():base=joblib.load(path)
            else:base=cls().fit(tr);joblib.dump(base,path,compress=3)
            p=te[['season','week','game_id','team','y']].copy();p['position']=pos
            if pos=='K':p['player_id']=te.player_id
            p['v42']=baseline_predict(base,te,pos)
            for c,w in weights.items():
                if c=='v42':continue
                if c not in ['extra16_extended','extra40_recent']:raise ValueError('Add explicitly selected model implementation')
                trsel=tr[tr.season>=year-3] if c.endswith('recent') else tr
                model=dict(candidates())['extra16' if c=='extra16_extended' else 'extra40']
                model.fit(trsel[cols],trsel.y);p[c]=model.predict(te[cols])
            p['candidate']=sum(w*p[c] for c,w in weights.items())
            p.to_csv(OUT/f'confirmation_{pos}_{year}.csv',index=False);allpred.append(p)
            for m in ['v42','candidate']:rows.append({'position':pos,'period':str(year),'model':m,**metrics(p,p[m])})
            print(pos,year,rows[-2:],flush=True)
    allp=pd.concat(allpred);decisions={}
    for pos,p in allp.groupby('position'):
        b=metrics(p,p.v42);m=metrics(p,p.candidate)
        for name,v in [('v42',b),('candidate',m)]:rows.append({'position':pos,'period':'2024-2025','model':name,**v})
        eligible=m['rmse']<b['rmse'] and m['mae']<b['mae'] and m['weekly_spearman']>=b['weekly_spearman'] and m['top3']>=b['top3']-.1
        decisions[pos]={'promote':bool(eligible),'baseline':b,'candidate':m,'bootstrap':bootstrap(p)}
    pd.DataFrame(rows).to_csv(OUT/'confirmation_metrics.csv',index=False)
    (OUT/'confirmation_decisions.json').write_text(json.dumps(decisions,indent=2)+'\n')
    print(json.dumps(decisions,indent=2),flush=True)

def bootstrap(p,n=5000):
    # Resample weeks together, preserving both sides of each game and common slate shocks.
    groups=list(p.groupby(['season','week']));stats=[]
    for _,g in groups:
        stats.append([len(g),abs(g.y-g.v42).sum(),abs(g.y-g.candidate).sum(),((g.y-g.v42)**2).sum(),((g.y-g.candidate)**2).sum(),spearmanr(g.y,g.v42).statistic,spearmanr(g.y,g.candidate).statistic])
    stats=np.asarray(stats);rng=np.random.default_rng(20260909);draw=stats[rng.integers(0,len(stats),size=(n,len(stats)))];s=draw.sum(axis=1)
    diffs={'mae_candidate_minus_baseline':(s[:,2]-s[:,1])/s[:,0],
           'rmse_candidate_minus_baseline':np.sqrt(s[:,4]/s[:,0])-np.sqrt(s[:,3]/s[:,0]),
           'weekly_spearman_candidate_minus_baseline':(s[:,6]-s[:,5])/len(stats)}
    return {k:{'ci95':np.quantile(v,[.025,.975]).tolist(),'bootstrap_fraction_improving':float((v>0).mean() if 'spearman' in k else (v<0).mean())} for k,v in diffs.items()}

if __name__=='__main__':confirm()
