"""Rolling xG with its own availability gate; baseline history stays unchanged.

The last N result-available matches define each window. Unavailable xG in that
window stays missing and receives no weight. Late xG occupies its original match
position, not its publication position.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from features import FEATURE_COLUMNS, WINDOWS, FeatureConfig, HistoryEngine, _weighted

XG_METRICS = ("xg_for", "xg_against")


@dataclass(frozen=True)
class XgFeatureConfig(FeatureConfig):
    extra_xg_lag_hours: float = 0.0


def xg_feature_names() -> list[str]:
    cols = []
    for side in ("home", "away"):
        for kind in ("form", "venue"):
            for n in WINDOWS:
                cols += [f"{side}_{kind}_{metric}_{n}" for metric in XG_METRICS]
    for kind in ("form", "venue"):
        for n in WINDOWS:
            cols += [f"diff_{kind}_{metric}_{n}" for metric in XG_METRICS]
    return cols


XG_FEATURE_COLUMNS = xg_feature_names()
FEATURE_COLUMNS_XG = FEATURE_COLUMNS + XG_FEATURE_COLUMNS


class XgHistoryEngine(HistoryEngine):
    def __init__(self, matches, config=XgFeatureConfig()):
        self.extra_xg_lag_hours = getattr(config, "extra_xg_lag_hours", 0.0)
        if not np.isfinite(self.extra_xg_lag_hours) or self.extra_xg_lag_hours < 0:
            raise ValueError("xG delay must be finite and nonnegative")
        if "xg_available" not in matches:
            raise ValueError("xG history requires a separate xg_available column")
        super().__init__(matches, config)

    def _result(self, rec):
        super()._result(rec)
        h, a = rec["home_id"], rec["away_id"]
        hx, ax = rec.get("home_xg"), rec.get("away_xg")
        hx = np.nan if hx is None or pd.isna(hx) else float(hx)
        ax = np.nan if ax is None or pd.isna(ax) else float(ax)
        available = pd.to_datetime(rec["xg_available"], utc=True)
        usable = (max(rec["usable_at"], available + pd.Timedelta(hours=self.extra_xg_lag_hours))
                  if pd.notna(available) else pd.NaT)
        # Form and venue share each record dict; fixture() gates every value.
        for team, own, other in ((h, hx, ax), (a, ax, hx)):
            self.form[team][-1].update(xg_for=own, xg_against=other,
                                      xg_usable_at=usable, xg_match_date=rec["match_date"])

    def _visible(self, records):
        out = []
        for rec in records:
            visible = pd.notna(rec["xg_usable_at"]) and rec["xg_usable_at"] < self.last_cutoff
            out.append({metric: rec[metric] if visible else np.nan for metric in XG_METRICS})
        return out

    def fixture(self, home, away, day, season):
        result = super().fixture(home, away, day, season)
        for team, side in ((home, "home"), (away, "away")):
            for kind, records in (("form", self.form[team]), ("venue", self.venue[(team, side)])):
                visible = self._visible(records)
                for n in WINDOWS:
                    for metric in XG_METRICS:
                        result[f"{side}_{kind}_{metric}_{n}"] = _weighted(visible, metric, n, self.config.decay)
                if kind == "form":
                    result[f"{side}_xg_available_matches_10"] = sum(np.isfinite(r["xg_for"]) for r in visible)
        for kind in ("form", "venue"):
            for n in WINDOWS:
                for metric in XG_METRICS:
                    result[f"diff_{kind}_{metric}_{n}"] = (result[f"home_{kind}_{metric}_{n}"]
                                                           - result[f"away_{kind}_{metric}_{n}"])
        return result


def build_features_xg(matches: pd.DataFrame, config: FeatureConfig = XgFeatureConfig()) -> pd.DataFrame:
    engine = XgHistoryEngine(matches, config)
    output = []
    for rec in matches.sort_values(["match_date", "match_id"]).to_dict("records"):
        values = engine.fixture(rec["home_id"], rec["away_id"], rec["match_date"], int(rec["season"]))
        output.append({**rec, **values})
    return pd.DataFrame(output)
