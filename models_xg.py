"""xG-based classifier candidates, added next to (never instead of) the baseline specs."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from epl_model.models import CLASSIFIER_SPECS, fit_classifier, xgb_options
from features_xg import FEATURE_COLUMNS_XG

# Small, hand-chosen set fixed BEFORE looking at any results: Elo plus decayed xG form.
LOGIT_XG_COLUMNS = ["elo_difference", "diff_form_xg_for_10", "diff_form_xg_against_10",
                    "diff_form_xg_for_5", "diff_form_xg_against_5"]

XG_CLASSIFIER_SPECS = {
    "logit_xg": {"kind": "logit_xg"},
    "xgb_shallow_xg": {"kind": "xgb_xg", "max_depth": 2, "n_estimators": 200},
    "xgb_medium_xg": {"kind": "xgb_xg", "max_depth": 3, "n_estimators": 150},
}
ALL_SPECS = {**CLASSIFIER_SPECS, **XG_CLASSIFIER_SPECS}


class LogitXg:
    """Multinomial logistic regression on a few scaled features. A missing difference
    (a promoted club with no history) is set to 0, i.e. 'no information', and the training
    median is NOT used so no future information leaks into the fill value."""
    def _x(self, frame):
        x = frame[LOGIT_XG_COLUMNS].astype(float).copy()
        x["elo_difference"] = x["elo_difference"] / 400
        return x.fillna(0.0).to_numpy()

    def fit(self, frame):
        self.model = LogisticRegression(C=1.0, max_iter=2000)
        self.model.fit(self._x(frame), frame.target)
        return self

    def predict_proba(self, frame):
        return self.model.predict_proba(self._x(frame))


class BoostedClassifierXg:
    def __init__(self, spec):
        self.spec = spec

    def fit(self, frame):
        self.model = XGBClassifier(objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                                   max_depth=self.spec["max_depth"], n_estimators=self.spec["n_estimators"],
                                   **xgb_options())
        self.model.fit(frame[FEATURE_COLUMNS_XG], frame.target)
        return self

    def predict_proba(self, frame):
        p = self.model.predict_proba(frame[FEATURE_COLUMNS_XG]).astype(float)
        return p / p.sum(axis=1, keepdims=True)


def fit_any(name, frame):
    if name in CLASSIFIER_SPECS:
        return fit_classifier(name, frame)
    spec = XG_CLASSIFIER_SPECS[name]
    return (LogitXg() if spec["kind"] == "logit_xg" else BoostedClassifierXg(spec)).fit(frame)
