from pathlib import Path
import stage6_fingerprint_recovery as recovery

ROOT = Path(__file__).resolve().parent
recovery.REF = ROOT / 'recovery_reference' / 'stage6d_baseline_classification_oof.b64'

if __name__ == '__main__':
    recovery.main()
