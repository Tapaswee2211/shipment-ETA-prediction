import numpy as np
import pandas as pd
import pytest

from src.features import RoutePriors, add_lateness
from src.explain import stage_attribution
from src.generator import SOURCES, generate_dataset
from src.models import ETAPredictor


@pytest.fixture(scope="module")
def df():
    return generate_dataset(120, seed=7)


def test_deterministic():
    a, b = generate_dataset(30, 1), generate_dataset(30, 1)
    pd.testing.assert_frame_equal(a, b)


def test_schedule_is_ordered_and_actuals_match_truth(df):
    assert (df.dep_sched_ts < df.port_arr_sched_ts).all()
    assert (df.port_arr_sched_ts < df.customs_sched_ts).all() and (df.customs_sched_ts < df.delivery_sched_ts).all()
    L = add_lateness(df)
    np.testing.assert_allclose(L["L3"], df["true_L3"], atol=1e-6)
    np.testing.assert_allclose(L["L0"], df["true_L0"], atol=1e-6)


def test_stage_increments_sum_to_final_delay(df):
    inc = df.true_L0 + (df.true_L1 - df.true_L0) + (df.true_L2 - df.true_L1) + (df.true_L3 - df.true_L2)
    np.testing.assert_allclose(inc, df.true_L3, atol=1e-9)


def test_true_shapley_is_efficient(df):
    phi = df[[f"true_phi_{s}" for s in SOURCES]].sum(axis=1)
    np.testing.assert_allclose(phi, df.true_L3 - df.true_baseline, atol=1e-8)
    np.testing.assert_allclose(df[[f"true_phi_stage{k}" for k in range(4)]].sum(axis=1), phi, atol=1e-8)


def test_missingness_present_and_target_always_observed(df):
    assert df.port_arrival_actual_ts.isna().any() and df.customs_cleared_actual_ts.isna().any()
    assert df.delivery_actual_ts.notna().all()


def test_features_do_not_use_future_information(df):
    """Scrambling everything that happens AFTER checkpoint k must not change CP-k features."""
    tr, te = df.iloc[:90], df.iloc[90:]
    for cp, later in [(0, ["port_arrival_actual_ts", "customs_cleared_actual_ts", "delivery_actual_ts",
                           "weather_actual", "dest_congestion_arr", "port_wait_h", "processing_h", "inspection_flag"]),
                      (1, ["customs_cleared_actual_ts", "delivery_actual_ts", "port_wait_h", "processing_h",
                           "inspection_flag"])]:
        P = ETAPredictor(cp, full=False).fit(tr)
        te2 = te.copy()
        rng = np.random.default_rng(0)
        for c in later:
            if c.endswith("_ts"):
                te2[c] = te2[c].sample(frac=1, random_state=1).to_numpy()
            else:
                te2[c] = rng.permutation(te2[c].to_numpy())
        X1 = P._features(P.prepare(te))
        X2 = P._features(P.prepare(te2))
        pd.testing.assert_frame_equal(X1, X2)


def test_route_priors_are_out_of_fold(df):
    d = add_lateness(df)
    pri = RoutePriors(seed=0)
    oof = pri.fit_transform(d)
    full = pri.transform(d)
    assert not np.allclose(oof[:, 0], full[:, 0])      # training rows must not see their own target


def test_predictions_sane_and_attribution_efficient(df):
    tr, te = df.iloc[:90], df.iloc[90:]
    for cp in (0, 1, 2):
        P = ETAPredictor(cp, n_draws=60).fit(tr)
        o = P.predict(te)
        assert np.isfinite(o.to_numpy()).all()
        assert (o.p10 <= o.p50 + 1e-9).all() and (o.p50 <= o.p90 + 1e-9).all()
        assert o.p_delay_clf.between(0, 1).all() and o.p_delay_chain.between(0, 1).all()
        A = stage_attribution(P, te)
        np.testing.assert_allclose(A.sum(axis=1), o.delay_gbm, atol=1e-6)


def test_gbm_beats_scheduled_baseline_at_cp2(df):
    tr, te = df.iloc[:90], df.iloc[90:]
    o = ETAPredictor(2, full=False).fit(tr).predict(te)
    y = add_lateness(te)["L3"]
    assert (o.delay_gbm - y).abs().mean() < (o.delay_sched - y).abs().mean()
