from pathlib import Path

from stage6_sos_ablation import scrape_teamrankings_season

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_stage6_sos"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    d = scrape_teamrankings_season(2025)
    if len(d) < 100:
        raise RuntimeError(f"Unexpected TeamRankings 2025 row count: {len(d)}")
    d.to_csv(OUT / "teamrankings_probe_2025.csv", index=False)
    print(d.head(10).to_string(index=False))
    print(f"rows={len(d)} min_rating={d.tr_sos_rating.min()} max_rating={d.tr_sos_rating.max()}")


if __name__ == "__main__":
    main()
