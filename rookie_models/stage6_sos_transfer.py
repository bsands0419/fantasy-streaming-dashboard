from pathlib import Path
import gzip,base64
ROOT=Path(__file__).resolve().parent
D=ROOT/'results_stage6_sos'
for stem in ['prospect_final_school_sos_historical','prospect_final_school_sos_current']:
    raw=(D/f'{stem}.csv').read_bytes()
    enc=base64.b64encode(gzip.compress(raw,compresslevel=9)).decode('ascii')
    chunk=3500
    for p in D.glob(f'{stem}_transfer_*.txt'): p.unlink()
    n=(len(enc)+chunk-1)//chunk
    for i in range(n): (D/f'{stem}_transfer_{i+1:02d}.txt').write_text(enc[i*chunk:(i+1)*chunk])
    (D/f'{stem}_transfer_count.txt').write_text(str(n))
    print(stem,len(raw),len(enc),n)
