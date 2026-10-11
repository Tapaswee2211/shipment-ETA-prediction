from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import APPT_WINDOW_H, DELAY_THRESHOLD_H, SEED
from .features import (FEATURE_GROUP, INTERACTION_COLS, PROPAGATION_COLS, H, LatenessImputer, RoutePriors,
                       add_lateness, build_features, dow_from_hours, holiday_from_hours, hours_since_base,
                       queue_factor)



def _lgbm(seed, n_estimators=250, num_leaves=7, **kw):
    params = dict(learning_rate=0.04, min_child_samples=12, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, reg_lambda=2.0, random_state=seed, verbose=-1, n_jobs=1)
    params.update(kw)
    return lgb.LGBMRegressor(n_estimators=n_estimators, num_leaves=num_leaves, **params)

