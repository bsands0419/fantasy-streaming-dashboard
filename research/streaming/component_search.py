"""Development-only structural alternatives to v4.2's kicker opportunity ensemble."""
from evaluate import *
from sklearn.feature_selection import SelectKBest, f_regression

def kicker_variants(base,tr,te):
    fam=base._component_families(te);B,S,R=base._analytic_component_score(fam,GenericKickerScoring());E=base._direct_prediction(te,GenericKickerScoring())
    preds={'k_component_B':B,'k_component_S':S,'k_component_R':R,'k_component_E':E}
    total=tr.fg_att.to_numpy(float)+tr.pat_att.to_numpy(float)
    share=np.divide(tr.fg_att,total,out=np.zeros(len(tr)),where=total>0)
    cols=list(K_V42_FEATURES)
    share_model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),SelectKBest(f_regression,k=50),Ridge(alpha=100))
    ok=total>0
    share_model.fit(tr.loc[ok,cols],share[ok],ridge__sample_weight=total[ok])
    for name,reg in [('bayes',BayesianRidge()),('ridge',Ridge(alpha=1000))]:
        fga=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),SelectKBest(f_regression,k=40),clone(reg)).fit(tr[cols],tr.fg_att)
        pat=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),SelectKBest(f_regression,k=40),clone(reg)).fit(tr[cols],tr.pat_att)
        fg=np.clip(fga.predict(te[cols]),0,6);pa=np.clip(pat.predict(te[cols]),0,7)
        shares=np.divide(fam['B'][2],fam['B'][0][:,None],out=np.tile(base.league_share,(len(te),1)),where=fam['B'][0][:,None]>0)
        newfam={**fam,'B':(fg,pa,fg[:,None]*shares)}
        newB=base._analytic_component_score(newfam,GenericKickerScoring())[0]
        preds[f'k_{name}_opportunities']=.3*newB+.4*S+.1*R+.2*E
    sf=base.scoring_attempts.predict(te);ss=np.clip(share_model.predict(te[cols]),0,1)
    shares=np.divide(fam['S'][2],fam['S'][0][:,None],out=np.tile(base.league_share,(len(te),1)),where=fam['S'][0][:,None]>0)
    wfam={**fam,'S':(sf*ss,sf*(1-ss),sf[:,None]*ss[:,None]*shares)}
    newS=base._analytic_component_score(wfam,GenericKickerScoring())[1]
    preds['k_weighted_share']=.3*B+.4*newS+.1*R+.2*E
    # More recent league and player distance-specific success rates respond to long-FG trends.
    for recent_weight in [.5,1.0]:
        skill=[]
        from feature_factory import K_BANDS
        for b in K_BANDS:
            recent=tr[tr.season>=tr.season.max()-1];rate=(recent[f'fg_made_{b}'].sum()+2)/(recent[f'fg_att_{b}'].sum()+4)
            prior=te[f'prior_made_{b}'].fillna(0);att=te[f'prior_att_{b}'].fillna(0)
            skill.append((prior+24*rate)/(att+24))
        nf={**fam,'skill':(1-recent_weight)*fam['skill']+recent_weight*np.column_stack(skill)}
        b,s,r=base._analytic_component_score(nf,GenericKickerScoring());preds[f'k_recent_skill{recent_weight}']=.3*b+.4*s+.1*r+.2*E
    return preds

def run_components(years):
    t,k=load_data()
    for year in years:
        tr=k[k.season<year];te=k[k.season==year];base=joblib.load(CACHE/f'baseline_K_{year}.joblib')
        p=OUT/f'K_{year}_predictions.csv';out=pd.read_csv(p)
        for name,values in kicker_variants(base,tr,te).items():out[name]=values
        out.to_csv(p,index=False)
    pd.concat([pd.read_csv(p) for p in sorted(OUT.glob('*_20??_predictions.csv'))]).to_csv(OUT/'all_predictions.csv',index=False)
    summarize()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--years',nargs='+',type=int,default=[2021,2022,2023]);run_components(parser.parse_args().years)
