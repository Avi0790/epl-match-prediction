"""Fixed, deliberately small models. No internal validation shuffle or early stopping."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier, XGBRegressor

from features import FEATURE_COLUMNS
from epl_model.scores import check_probabilities, outcome_probabilities


CLASSIFIER_SPECS = {
    "elo": {"kind": "elo"},
    "team_poisson": {"kind": "team_poisson"},
    "xgb_shallow": {"kind": "xgb", "max_depth": 2, "n_estimators": 200},
    "xgb_medium": {"kind": "xgb", "max_depth": 3, "n_estimators": 150},
}
GOAL_SPECS = ("team_poisson", "xgb_poisson")


def xgb_options():
    return dict(learning_rate=0.035, min_child_weight=20, reg_lambda=10.0,
                subsample=1.0, colsample_bytree=1.0, random_state=17,
                n_jobs=2, tree_method="hist", verbosity=0)


class EloClassifier:
    def fit(self, frame):
        self.model = LogisticRegression(C=1.0, max_iter=1000)
        self.model.fit(frame[["elo_difference"]].to_numpy() / 400, frame.target)
        return self

    def predict_proba(self, frame):
        return self.model.predict_proba(frame[["elo_difference"]].to_numpy() / 400)


class TeamPoisson:
    """Shared regularized attack/defence Poisson regression; NOT Dixon-Coles."""
    def _rows(self, frame):
        home = np.column_stack([frame.home_id, frame.away_id])
        away = np.column_stack([frame.away_id, frame.home_id])
        return np.concatenate([home, away]), np.r_[np.ones(len(frame)), np.zeros(len(frame))]

    def _x(self, frame, fitting=False):
        teams, is_home = self._rows(frame)
        if fitting:
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
            encoded = self.encoder.fit_transform(teams)
        else:
            encoded = self.encoder.transform(teams)
        return sparse.hstack([encoded, sparse.csr_matrix(is_home[:, None])], format="csr")

    def fit(self, frame):
        x = self._x(frame, fitting=True)
        ages = (pd_dates(frame).max() - pd_dates(frame)).dt.days.to_numpy()
        weight = np.exp(-np.log(2) * ages / 730)
        self.model = PoissonRegressor(alpha=0.02, max_iter=1000, tol=1e-7)
        self.model.fit(x, np.r_[frame.home_goals, frame.away_goals], sample_weight=np.r_[weight, weight])
        return self

    def predict_rates(self, frame):
        values = self.model.predict(self._x(frame))
        return np.column_stack([values[:len(frame)], values[len(frame):]])

    def predict_proba(self, frame):
        return outcome_probabilities(self.predict_rates(frame))


def pd_dates(frame):
    import pandas as pd
    return pd.to_datetime(frame.match_date)


class BoostedClassifier:
    def __init__(self, spec):
        self.spec = spec

    def fit(self, frame):
        self.model = XGBClassifier(objective="multi:softprob", num_class=3,
            eval_metric="mlogloss", max_depth=self.spec["max_depth"],
            n_estimators=self.spec["n_estimators"], **xgb_options())
        self.model.fit(frame[FEATURE_COLUMNS], frame.target)
        return self

    def predict_proba(self, frame):
        p = self.model.predict_proba(frame[FEATURE_COLUMNS]).astype(float)
        return p / p.sum(axis=1, keepdims=True)


class BoostedPoisson:
    def fit(self, frame):
        self.models = []
        for target in ("home_goals", "away_goals"):
            model = XGBRegressor(objective="count:poisson", max_depth=2,
                                 n_estimators=180, **xgb_options())
            model.fit(frame[FEATURE_COLUMNS], frame[target])
            self.models.append(model)
        return self

    def predict_rates(self, frame):
        return np.column_stack([m.predict(frame[FEATURE_COLUMNS]) for m in self.models]).astype(float)


class PlattCalibrator:
    """One sigmoid per class, normalized. Fitted ONLY on a later calibration block."""
    def fit(self, probabilities, y):
        p = check_probabilities(probabilities)
        self.models = []
        for k in range(3):
            binary = (np.asarray(y) == k).astype(int)
            if len(np.unique(binary)) < 2:
                raise ValueError("Calibration block must contain every outcome class")
            model = LogisticRegression(C=10.0, max_iter=1000)
            model.fit(self._logit(p[:, k]), binary)
            self.models.append(model)
        return self

    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))[:, None]

    def transform(self, probabilities):
        p = check_probabilities(probabilities)
        calibrated = np.column_stack([m.predict_proba(self._logit(p[:, k]))[:, 1]
                                      for k, m in enumerate(self.models)])
        return calibrated / calibrated.sum(axis=1, keepdims=True)


@dataclass
class CalibratedModel:
    base: object
    calibrator: PlattCalibrator | None = None

    def predict_proba(self, frame):
        p = self.base.predict_proba(frame)
        return self.calibrator.transform(p) if self.calibrator else p


def fit_classifier(name, frame):
    spec = CLASSIFIER_SPECS[name]
    model = EloClassifier() if name == "elo" else TeamPoisson() if name == "team_poisson" else BoostedClassifier(spec)
    return model.fit(frame)


def fit_goal_model(name, frame):
    return (TeamPoisson() if name == "team_poisson" else BoostedPoisson()).fit(frame)
