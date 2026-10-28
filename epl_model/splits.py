"""Chronological splits shared by training and regression tests."""
from __future__ import annotations

import pandas as pd
from features import day_cutoff


def assert_order(train, calibration, evaluation):
    if min(len(train), len(calibration), len(evaluation)) == 0:
        raise ValueError("Train, calibration, and evaluation must all contain matches")
    blocks = (train, calibration, evaluation)
    ids = [set(b.match_id) for b in blocks]
    if any(ids[i] & ids[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Overlapping partitions")
    for earlier, later in zip(blocks[:-1], blocks[1:]):
        if earlier.match_date.max() >= later.match_date.min():
            raise ValueError("Partitions are not strictly chronological")
        if pd.to_datetime(earlier.history_available_at, utc=True).max() >= day_cutoff(later.match_date.min()):
            raise ValueError("Fitting labels/statistics were not available before the next block")


def season_split(frame, evaluation_season, first_training_season=2015):
    training = frame[(frame.season >= first_training_season) & (frame.season < evaluation_season - 1)].copy()
    calibration = frame[frame.season == evaluation_season - 1].copy()
    evaluation = frame[frame.season == evaluation_season].copy()
    assert_order(training, calibration, evaluation)
    return training, calibration, evaluation


def describe(frame):
    return {"matches": len(frame), "first_date": frame.match_date.min() if len(frame) else None,
            "last_date": frame.match_date.max() if len(frame) else None,
            "seasons": sorted(int(s) for s in frame.season.unique())}
