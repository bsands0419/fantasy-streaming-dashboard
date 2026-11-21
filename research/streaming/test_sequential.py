"""Temporal leakage and slate-invariance checks for the continuation research."""
import unittest
from sequential_search import *


class SequentialIntegrity(unittest.TestCase):
    def test_ranker_rows_match_baseline_player_ids_and_targets(self):
        from ranking_search import ranking_data, baseline_frame
        for pos, d in ranking_data().items():
            for year in range(2021, 2026):
                p = baseline_frame(pos, year)
                te = d[d.season.eq(year)]
                self.assertEqual(list(zip(p.game_id, p.team)), list(zip(te.game_id, te.team)))
                np.testing.assert_allclose(p.y, te.y, atol=1e-6)
                if pos == 'K':
                    self.assertEqual(list(p.player_id), list(te.player_id))

    def test_current_and_future_qbs_cannot_change_earlier_inputs(self):
        t, _ = load_data()
        schedule = t[['season', 'week', 'game_id', 'team', 'opponent_team']]
        raw = qb_games()
        old = qb_histories(raw, schedule)
        altered = raw.copy()
        mask = (altered.season > 2023) | ((altered.season == 2023) & (altered.week >= 8))
        altered.loc[mask, QB_COLS] = 9999.
        altered.loc[mask, 'qb'] = 'FAKE_FUTURE_STARTER'
        new = qb_histories(altered, schedule)
        rows = schedule[(schedule.season < 2023) | ((schedule.season == 2023) & (schedule.week <= 8))]
        keys = ['game_id', 'team']
        pd.testing.assert_frame_equal(rows[keys].merge(old, on=keys), rows[keys].merge(new, on=keys))

    def test_qb_features_do_not_depend_on_slate_row_order(self):
        t, _ = load_data()
        schedule = t[['season', 'week', 'game_id', 'team', 'opponent_team']]
        raw = qb_games()
        a = qb_histories(raw, schedule).sort_values(['game_id', 'team']).reset_index(drop=True)
        b = qb_histories(raw.sample(frac=1, random_state=7), schedule.sample(frac=1, random_state=9))
        b = b.sort_values(['game_id', 'team']).reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)

    def test_same_week_outcomes_cannot_change_corrections(self):
        d = data()['DST']
        week = d[d.season.eq(2023) & d.week.eq(8)]
        cfg = {'name': 'test', 'method': 'ridge', 'fs': 'combined', 'half': 54, 'alpha': 1000}
        expected = predictions(d, 'DST', [cfg], [2023])
        poisoned = d.copy()
        poisoned.loc[poisoned.clock >= week.clock.min(), ['y', 'residual']] = 9999.
        actual = predictions(poisoned, 'DST', [cfg], [2023])
        np.testing.assert_allclose(expected.loc[expected.week <= 8, 'test'], actual.loc[actual.week <= 8, 'test'])

    def test_centered_calibration_preserves_slate_mean(self):
        d = data()['DST']
        te = d[d.season.eq(2023) & d.week.eq(8)]
        tr = d[d.clock < te.clock.min()]
        cfg = {'name': 'test', 'method': 'ridge', 'fs': 'combined', 'half': 54,
               'alpha': 1000, 'center': True, 'weight': .5}
        p = forecast(tr, te, cfg, 'DST')
        self.assertAlmostEqual(p.mean(), te.v42.mean(), places=12)
        np.testing.assert_allclose(p, forecast(tr, te.iloc[::-1], cfg, 'DST')[::-1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
