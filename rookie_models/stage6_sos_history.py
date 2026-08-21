from stage6_sos_ablation import load_teamrankings_history


def main():
    d = load_teamrankings_history()
    print(f"seasons={d.college_season.nunique()} first={int(d.college_season.min())} last={int(d.college_season.max())} rows={len(d)}")
    print(d.groupby('college_season').size().to_string())


if __name__ == '__main__':
    main()
