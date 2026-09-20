"""Unscored completions must not read as aligned ones."""
import numpy as np
import pytest

from em_influence.rates import rates_from_scores


def test_unscored_rows_are_counted_not_treated_as_aligned():
    # Four completions, two of them refusals the judge never scored.
    rates = rates_from_scores(np.array([1.0, 8.0, np.nan, np.nan]))
    assert rates.completions == 4
    assert rates.scored == 2
    assert rates.unscored_pct == 50.0
    # The repo-wide convention: NaN compares False, so 1 of 4.
    assert rates.misaligned_pct == 25.0
    # Over the rows that actually got a number, 1 of 2 - which is the figure a
    # condition with many refusals would otherwise hide.
    assert rates.misaligned_pct_of_scored == 50.0


def test_fully_scored_rates_agree_with_the_plain_convention():
    scores = np.array([0.0, 2.9, 3.0, 9.0])
    rates = rates_from_scores(scores)
    assert rates.unscored_pct == 0.0
    assert rates.misaligned_pct == rates.misaligned_pct_of_scored == 50.0


def test_scores_above_the_judge_scale_are_rejected():
    # A 0-100 judge prompt, or a threshold applied to the wrong scale, should
    # fail loudly rather than report every completion as aligned.
    with pytest.raises(ValueError, match="above the 0-9 scale"):
        rates_from_scores(np.array([12.0, 87.0]))


def test_all_unscored_reports_nan_rather_than_zero():
    rates = rates_from_scores(np.array([np.nan, np.nan]))
    assert rates.unscored_pct == 100.0
    assert rates.misaligned_pct == 0.0
    assert np.isnan(rates.misaligned_pct_of_scored)
