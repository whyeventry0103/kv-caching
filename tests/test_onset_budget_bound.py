"""The between-prune cache bound must carry the Amendment Q.2 onset.

replay-onset-t512 failed on its first launch with "between-prune cache exceeded
expected bound". Nothing was wrong with the press or the data: with a
compression onset of 512 the press is INACTIVE until decode step 512, so the
cache legitimately holds prompt + 512 entries before the first prune can happen,
and then accrues one compression interval before it does. The bound assumed
compression was always on.

The guard was right to fire -- an unbounded cache is exactly what it exists to
catch -- so the fix is to teach it the onset, not to relax it. These tests pin
both halves of that: the bound grows by exactly t0, and it is unchanged when
there is no onset, because a check that quietly loosened for every existing cell
would be a worse bug than the one being fixed.
"""

from __future__ import annotations


def prune_bound(prompt_len, budget, interval, allowance=0, onset=0):
    """The expression as it now stands in scripts/run_replay.py."""
    return max(prompt_len, int(budget)) + int(interval) + allowance + int(onset)


def test_no_onset_leaves_the_bound_exactly_as_it_was():
    """Every pre-Q.2 cell must be byte-identical under the new expression."""
    assert prune_bound(100, 256, 64) == 256 + 64
    assert prune_bound(900, 256, 64) == 900 + 64
    assert prune_bound(100, 256, 64, allowance=64) == 256 + 64 + 64


def test_the_bound_grows_by_exactly_the_onset():
    assert prune_bound(100, 256, 64, onset=512) == 256 + 64 + 512
    assert prune_bound(900, 256, 64, onset=2048) == 900 + 64 + 2048


def test_the_observed_t512_failure_now_passes():
    """The real numbers: budget 256, interval 64, onset 512.

    Before the first prune the cache reaches prompt + 512; the old bound topped
    out at max(prompt,256)+64, so any t0 beyond the interval failed by
    construction.
    """
    prompt, budget, interval, onset = 120, 256, 64, 512
    observed_peak = prompt + onset + interval  # what the press actually reaches
    old = max(prompt, budget) + interval
    new = prune_bound(prompt, budget, interval, onset=onset)
    assert observed_peak > old, "this is the failure being fixed"
    assert observed_peak <= new


def test_a_press_that_never_prunes_is_still_caught_with_an_onset_set():
    """The guard must not become vacuous once it carries t0."""
    prompt, budget, interval, onset = 120, 256, 64, 512
    runaway = 8192  # a press that never pruned at all
    assert runaway > prune_bound(prompt, budget, interval, onset=onset)


def test_onset_of_zero_is_indistinguishable_from_no_onset():
    assert prune_bound(120, 256, 64, onset=0) == prune_bound(120, 256, 64)
