from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import APPT_WINDOW_H, DELAY_THRESHOLD_H, SEED
from .features import (FEATURE_GROUP, INTERACTION_COLS, PROPAGATION_COLS, LatenessImputer, RoutePriors,
                       add_lateness, build_features, dow_from_hours, holiday_from_hours, hours_since_base,
                       queue_factor)


class FeatureContractError(ValueError):
    """The input row cannot be turned into the exact feature schema the model was trained on."""


# observations that are legitimately absent whenever the customs record is missing
NULLABLE_BY_DESIGN = {"port_wait_h", "processing_h", "proc_excess", "inspection_flag"}


def _lgbm(seed, n_estimators=250, num_leaves=7, **kw):
    params = dict(learning_rate=0.04, min_child_samples=12, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, reg_lambda=2.0, random_state=seed, verbose=-1, n_jobs=1)
    params.update(kw)
    return lgb.LGBMRegressor(n_estimators=n_estimators, num_leaves=num_leaves, **params)


# =============================================================================
# Structured chain model
# =============================================================================
def stage_X(j: int, d: pd.DataFrame, L: np.ndarray, forecast: bool) -> np.ndarray:
    """Inputs for the stage-j increment model, given lateness L entering the stage."""
    L = np.asarray(L, float)
    if j == 1:
        w = d["weather_fcst"].to_numpy(float)                       # weather is only forecast at departure
        return np.c_[L, d["slack_transit_h"], d["sched_transit_h"], w, w * d["weather_exposure"],
                     d["weather_exposure"]]
    if j == 2:
        c = d["dest_congestion_dep"] if forecast else d["dest_congestion_arr"]
        q = queue_factor(c)
        h = hours_since_base(d["port_arr_sched_ts"]).to_numpy(float) + L
        proc = d["proc_nominal_h"].to_numpy(float)
        return np.c_[L, d["slack_port_h"], proc, L - d["berth_window_h"].to_numpy(float), q, q * proc,
                     dow_from_hours(h) >= 5, holiday_from_hours(h), np.full(len(L), float(forecast))]
    return np.c_[L, d["slack_inland_h"], d["sched_inland_h"], L - APPT_WINDOW_H]


class ChainModel:
    def __init__(self, seed: int = SEED, n_draws: int = 200):
        self.seed, self.n_draws = seed, n_draws

    # Domain prior: worse weather / a more congested port never *reduces* the delay added in that stage.
    # Column order of stage_X: stage 1 = [L, slack, sched, w, w*exposure, exposure]
    #                          stage 2 = [L, slack, proc, L-berth, queue, queue*proc, weekend, holiday, is_forecast]
    MONOTONE = {1: [0, 0, 0, 1, 1, 0], 2: [0, 0, 0, 0, 1, 1, 0, 0, 0], 3: None}

    def _base(self, j):
        kw = {"monotone_constraints": self.MONOTONE[j]} if self.MONOTONE[j] else {}
        return _lgbm(self.seed, n_estimators=150, num_leaves=5, min_child_samples=8, **kw)

    def fit(self, d: pd.DataFrame):
        o1 = (d["L1_missing"] == 0).to_numpy()
        o2 = (d["L2_missing"] == 0).to_numpy()
        specs = {1: (o1, d["L0"].to_numpy(), (d["L1"] - d["L0"]).to_numpy()),
                 2: (o1 & o2, d["L1"].to_numpy(), (d["L2"] - d["L1"]).to_numpy()),
                 3: (o2, d["L2"].to_numpy(), (d["L3"] - d["L2"]).to_numpy())}
        self.models, self.bank = {}, {}
        for j, (m, Lin, y) in specs.items():
            dm = d.loc[m].reset_index(drop=True)
            Lm, ym = Lin[m], y[m]
            XA = stage_X(j, dm, Lm, False)
            XB = stage_X(j, dm, Lm, True) if j == 2 else None
            if j == 2:
                self.models[j] = self._base(j).fit(np.vstack([XA, XB]), np.r_[ym, ym])
            else:
                self.models[j] = self._base(j).fit(XA, ym)
            # out-of-fold residuals (split by shipment so duplicated rows never straddle folds)
            oofA, oofB = np.zeros(len(ym)), np.zeros(len(ym))
            for tr, te in KFold(5, shuffle=True, random_state=self.seed).split(XA):
                mdl = self._base(j)
                if j == 2:
                    mdl.fit(np.vstack([XA[tr], XB[tr]]), np.r_[ym[tr], ym[tr]])
                    oofB[te] = mdl.predict(XB[te])
                else:
                    mdl.fit(XA[tr], ym[tr])
                oofA[te] = mdl.predict(XA[te])
            routes = dm["route_id"].to_numpy()
            self.bank[(j, "obs")] = self._make_bank(ym - oofA, routes)
            if j == 2:
                self.bank[(j, "fc")] = self._make_bank(ym - oofB, routes)
        return self

    @staticmethod
    def _make_bank(res, routes):
        """Sorted out-of-fold residuals: overall, plus per lane when a lane has >= 10 of them."""
        bank = {"__all__": np.sort(res)}
        for r in np.unique(routes):
            if (routes == r).sum() >= 10:
                bank[r] = np.sort(res[routes == r])
        return bank

    def _sample(self, key, routes, u):
        """Residual = empirical quantile u of the lane's residual bank. Passing the *same* u to every row gives
        common random numbers: two scenarios differ only because their inputs differ, never because of sampling
        noise, and a row's prediction does not depend on which other rows are in the batch."""
        bank = self.bank[key]
        out = np.empty(len(routes))
        for r in np.unique(routes):
            idx = routes == r
            b = bank.get(r, bank["__all__"])
            out[idx] = b[np.minimum((u[idx] * len(b)).astype(int), len(b) - 1)]
        return out

    def simulate(self, d: pd.DataFrame, cp: int, seed: int = 0):
        """Monte-Carlo the remaining stages. Returns (final-lateness draws [n,K], {stage: increment draws})."""
        rng = np.random.default_rng(seed)
        n, K = len(d), self.n_draws
        dr = d.iloc[np.repeat(np.arange(n), K)].reset_index(drop=True)
        L = np.repeat(d[f"L{cp}"].to_numpy(float), K)
        routes = dr["route_id"].to_numpy()
        incs = {}
        for j in range(cp + 1, 4):
            forecast = cp < j - 1
            mu = self.models[j].predict(stage_X(j, dr, L, forecast))
            u = np.tile(rng.random(K), n)                       # same K draws for every row (see _sample)
            res = self._sample((j, "fc" if (j == 2 and forecast) else "obs"), routes, u)
            inc = mu + res
            incs[j] = inc.reshape(n, K)
            L = L + inc
        return L.reshape(n, K), incs


# =============================================================================
# Checkpoint predictor
# =============================================================================
class ETAPredictor:
    def __init__(self, cp: int, seed: int = SEED, imputation: str = "model", interactions: bool = True,
                 propagation: bool = True, priors: bool = True, full: bool = True, n_draws: int = 200):
        self.cp, self.seed, self.imputation = cp, seed, imputation
        self.interactions, self.propagation, self.use_priors = interactions, propagation, priors
        self.full, self.n_draws = full, n_draws

    # ---- preparation (all learned pieces fitted on train only) ----
    def prepare(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        d = add_lateness(df)
        if fit:
            self.imp = LatenessImputer(self.imputation, self.seed).fit(d)
        d = self.imp.transform(d)
        if fit:
            self.pri = RoutePriors(seed=self.seed)
            arr = self.pri.fit_transform(d)                     # out-of-fold for training rows
        else:
            arr = self.pri.transform(d)
        d["prior_route_delay"], d["prior_route_std"] = arr[:, 0], arr[:, 1]
        return d

    def _features(self, d: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        X = build_features(d, self.cp)
        drop = []
        if not self.interactions:
            drop += INTERACTION_COLS
        if not self.propagation:
            drop += PROPAGATION_COLS
        if not self.use_priors:
            drop += ["prior_route_delay", "prior_route_std"]
        if self.imputation != "model":
            drop += ["L1_missing", "L2_missing"]
        X = X.drop(columns=[c for c in drop if c in X.columns])
        if fit:
            # optional signals this dataset never provides are removed explicitly (not silently imputed)
            self.dropped_all_nan = [c for c in X.columns if X[c].isna().all()]
            X = X.drop(columns=self.dropped_all_nan)
            self.cols = list(X.columns)
            self.required = [c for c in self.cols if c not in NULLABLE_BY_DESIGN and X[c].notna().all()]
            return X
        missing = [c for c in self.cols if c not in X.columns]
        if missing:
            raise FeatureContractError(f"feature columns missing at checkpoint {self.cp}: {missing}")
        X = X[self.cols]
        bad = [c for c in self.required if X[c].isna().any()]
        if bad:
            raise FeatureContractError(f"required features are missing/NaN for this input (checkpoint {self.cp}): {bad}")
        return X

    def fit(self, df: pd.DataFrame):
        d = self.prepare(df, fit=True)
        X = self._features(d, fit=True)
        Lk = d[f"L{self.cp}"].to_numpy(float)
        y = d["L3"].to_numpy(float) - Lk
        self.ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                   RidgeCV(alphas=np.logspace(-1, 3, 20))).fit(X, y)
        self.gbm = _lgbm(self.seed).fit(X, y)
        if self.full:
            self.q = {a: _lgbm(self.seed, n_estimators=200, objective="quantile", alpha=a).fit(X, y)
                      for a in (0.1, 0.5, 0.9)}
            lab = (d["L3"].to_numpy(float) > DELAY_THRESHOLD_H).astype(int)
            clf = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.04, num_leaves=5, min_child_samples=12,
                                     subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0,
                                     random_state=self.seed, verbose=-1, n_jobs=1)
            self.clf = CalibratedClassifierCV(clf, method="sigmoid", cv=3).fit(X, lab)
            self.chain = ChainModel(self.seed, self.n_draws).fit(d)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Every ``delay_*`` / ``p10..p90`` / ``c10..c90`` column is the TOTAL final delay in hours versus the scheduled
        delivery, i.e. it already contains the delay observed so far (L_k). Therefore
            predicted ETA = delivery_sched_ts + delay        (never add L_k again)
        and the remaining delay is  delay - L_k.  Models are trained on that remaining delay R_k = L_3 - L_k.
          delay_chain         mean of the chain simulation        c10/c50/c90  its 10th/50th/90th percentiles
          delay_chain_median  = c50, the point estimate shown in the app (lowest MAE in cross-validation)
          p_delay_chain       share of simulated outcomes with final delay > DELAY_THRESHOLD_H
          p_delay_clf         separate calibrated GBM classifier for the same event (cross-check)
          chain_L0..chain_L3  mean cumulative lateness at each milestone (observed ones are the inputs)
        """
        d = self.prepare(df)
        X = self._features(d)
        Lk = d[f"L{self.cp}"].to_numpy(float)
        out = pd.DataFrame(index=df.index)
        out["L_k"] = Lk
        # bookkeeping: was an observed milestone (k <= checkpoint) missing and therefore estimated?
        out["L1_imputed"] = d["L1_missing"].to_numpy(float) * (self.cp >= 1)
        out["L2_imputed"] = d["L2_missing"].to_numpy(float) * (self.cp >= 2)
        out["delay_sched"] = 0.0
        out["delay_carry"] = Lk
        out["delay_route_prior"] = self.pri.transform(d)[:, 0]
        out["delay_ridge"] = Lk + self.ridge.predict(X)
        out["delay_gbm"] = Lk + self.gbm.predict(X)
        if self.full:
            qs = np.sort(np.c_[[self.q[a].predict(X) for a in (0.1, 0.5, 0.9)]].T, axis=1) + Lk[:, None]
            out["p10"], out["p50"], out["p90"] = qs[:, 0], qs[:, 1], qs[:, 2]
            out["p_delay_clf"] = self.clf.predict_proba(X)[:, 1]
            draws, incs = self.chain.simulate(d, self.cp, seed=self.seed)
            out["delay_chain"] = draws.mean(1)
            out["c10"], out["c50"], out["c90"] = np.percentile(draws, [10, 50, 90], axis=1)
            out["delay_chain_median"] = out["c50"]
            out["p_delay_chain"] = (draws > DELAY_THRESHOLD_H).mean(1)
            for k in range(self.cp + 1):                       # milestones already observed
                out[f"chain_L{k}"] = d[f"L{k}"].to_numpy(float)
            cur = Lk.copy()
            for j in range(self.cp + 1, 4):                    # expected cumulative lateness at future milestones
                cur = cur + incs[j].mean(1)
                out[f"chain_L{j}"] = cur
                out[f"chain_inc_s{j}"] = incs[j].mean(1)
        return out

    def feature_row(self, df: pd.DataFrame) -> pd.DataFrame:
        """The exact feature matrix the tree/linear models receive for these rows (validated against the contract)."""
        return self._features(self.prepare(df))

    # ---- explanation hooks ----
    def shap_features(self, df: pd.DataFrame):
        d = self.prepare(df)
        X = self._features(d)
        ex = shap.TreeExplainer(self.gbm)
        sv = pd.DataFrame(ex.shap_values(X), index=df.index, columns=X.columns)
        return sv, float(np.ravel(ex.expected_value)[0]), d, X

    def shap_groups(self, df: pd.DataFrame):
        sv, base, d, _ = self.shap_features(df)
        g = sv.T.groupby(lambda c: FEATURE_GROUP[c]).sum().T
        return g, base, d
