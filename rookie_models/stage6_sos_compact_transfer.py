from __future__ import annotations
import base64, gzip
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
RES = ROOT / 'results_stage6_sos'
SRC = RES / 'prospect_final_school_sos_historical.csv'
OUT = RES / 'prospect_final_school_sos_2014_2023_compact.txt'

d = pd.read_csv(SRC)
d = d[pd.to_numeric(d['season'], errors='coerce').between(2014, 2023)].copy()
keep = ['season','pfr_name','position','college_season','tr_sos_rating','tr_sos_z','tr_sos_rank_pct','match_method','match_score']
d = d[[c for c in keep if c in d.columns]].drop_duplicates(['season','pfr_name','position'])
raw = d.to_csv(index=False).encode('utf-8')
enc = base64.b64encode(gzip.compress(raw, compresslevel=9)).decode('ascii')
OUT.write_text(enc)
chunk = 2800
parts = [enc[i:i+chunk] for i in range(0, len(enc), chunk)]
for old in RES.glob('sos_compact_2014_2023_part_*.txt'):
    old.unlink()
for i, part in enumerate(parts, 1):
    (RES / f'sos_compact_2014_2023_part_{i:02d}.txt').write_text(part)
(RES / 'sos_compact_2014_2023_part_count.txt').write_text(str(len(parts)))
print('rows', len(d), 'raw_bytes', len(raw), 'encoded_chars', len(enc), 'parts', len(parts))
