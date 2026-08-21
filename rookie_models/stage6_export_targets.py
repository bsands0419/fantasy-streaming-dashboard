from pathlib import Path
import base64
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'results_stage5' / 'historical_prospect_calibration_rows.csv'
OUT = ROOT / 'results_stage6b'
OUT.mkdir(parents=True, exist_ok=True)


def write_transfer_chunks(text: str, stem: str, chunk_size: int = 3500):
    enc = base64.b64encode(text.encode('utf-8')).decode('ascii')
    for p in OUT.glob(f'{stem}_chunk_*.txt'):
        p.unlink()
    for i in range(0, len(enc), chunk_size):
        (OUT / f'{stem}_chunk_{i // chunk_size + 1:02d}.txt').write_text(enc[i:i+chunk_size])
    (OUT / f'{stem}_chunk_count.txt').write_text(str((len(enc) + chunk_size - 1) // chunk_size))


def main():
    d = pd.read_csv(SRC, low_memory=False)
    d['season'] = pd.to_numeric(d['season'], errors='coerce')
    d = d[d['season'].between(2014, 2023)].copy()
    cols = ['season','pfr_name','draft_pick','primary_ppg','position','prospect_model_score','hit3','star3','best_rank3','target_valid']
    d = d[cols].copy()
    master = d.to_csv(index=False)
    (OUT / 'completed_outcome_targets_2014_2023.csv').write_text(master)
    slim = ['season','pfr_name','draft_pick','primary_ppg','prospect_model_score','hit3','star3']
    for pos in ['QB','RB','WR','TE']:
        q = d[d.position.eq(pos)][slim].copy()
        q.to_csv(OUT / f'completed_outcome_targets_{pos}_2014_2023.csv', index=False)
    write_transfer_chunks(master, 'completed_outcome_targets_2014_2023')
    print(d.groupby(['position','season']).size().to_string())
    print('rows', len(d))

if __name__ == '__main__':
    main()
