from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'results_stage5' / 'historical_prospect_calibration_rows.csv'
OUT = ROOT / 'results_stage6b'
OUT.mkdir(parents=True, exist_ok=True)


def main():
    d = pd.read_csv(SRC, low_memory=False)
    d['season'] = pd.to_numeric(d['season'], errors='coerce')
    d = d[d['season'].between(2014, 2023)].copy()
    cols = ['season','pfr_name','draft_pick','primary_ppg','position','prospect_model_score','hit3','star3','best_rank3','target_valid']
    d = d[cols].copy()
    d.to_csv(OUT / 'completed_outcome_targets_2014_2023.csv', index=False)
    slim = ['season','pfr_name','draft_pick','primary_ppg','prospect_model_score','hit3','star3']
    for pos in ['QB','RB','WR','TE']:
        q = d[d.position.eq(pos)][slim].copy()
        q.to_csv(OUT / f'completed_outcome_targets_{pos}_2014_2023.csv', index=False)
    print(d.groupby(['position','season']).size().to_string())
    print('rows', len(d))

if __name__ == '__main__':
    main()
