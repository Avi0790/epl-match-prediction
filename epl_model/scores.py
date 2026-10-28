"""Poisson score distributions and coherent H/D/A reconciliation."""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson


def check_probabilities(p):
    p = np.asarray(p, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("Expected a finite N by 3 H/D/A probability array")
    if (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-8, rtol=0):
        raise ValueError("Probabilities must be nonnegative and sum to one")
    return p


def score_matrix(home_rate, away_rate, outcome_probabilities=None, tail_tolerance=1e-10):
    """Rows are home goals, columns away goals. No overflow bucket masquerades as a score."""
    rates = np.asarray([home_rate, away_rate], dtype=float)
    if not np.isfinite(rates).all() or (rates <= 0).any() or (rates > 50).any():
        raise ValueError("Goal rates must be finite and in (0, 50]")
    if not 0 < tail_tolerance < 0.01:
        raise ValueError("Invalid tail tolerance")
    limit = max(8, int(poisson.ppf(1 - tail_tolerance / 4, rates).max()))
    goals = np.arange(limit + 1)
    matrix = np.outer(poisson.pmf(goals, home_rate), poisson.pmf(goals, away_rate))
    retained = float(matrix.sum())
    matrix /= retained
    if outcome_probabilities is not None:
        p = check_probabilities(np.asarray(outcome_probabilities).reshape(1, 3))[0]
        h, a = np.indices(matrix.shape)
        for region, target in zip((h > a, h == a, h < a), p):
            matrix[region] *= target / matrix[region].sum()
    return matrix, {"max_goals_each_side": limit, "poisson_tail_mass_omitted": max(0.0, 1 - retained)}


def matrix_outcomes(matrix):
    return np.array([np.tril(matrix, -1).sum(), np.trace(matrix), np.triu(matrix, 1).sum()])


def outcome_probabilities(rates):
    return np.stack([matrix_outcomes(score_matrix(h, a)[0]) for h, a in np.asarray(rates)])


def summarize_score(home_rate, away_rate, p=None):
    matrix, tail = score_matrix(home_rate, away_rate, p)
    h, a = np.indices(matrix.shape)
    top = np.argsort(-matrix.ravel(), kind="stable")[:5]
    scores = [{"home_goals": int(i // matrix.shape[1]), "away_goals": int(i % matrix.shape[1]),
               "probability": float(matrix.ravel()[i])} for i in top]
    return {"expected_home_goals": float((h * matrix).sum()),
            "expected_away_goals": float((a * matrix).sum()),
            "poisson_home_rate": float(home_rate), "poisson_away_rate": float(away_rate),
            "most_likely_score": scores[0], "top_5_scorelines": scores,
            "p_over_2_5": float(matrix[h + a >= 3].sum()), **tail}, matrix
