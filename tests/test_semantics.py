import numpy as np
import pandas as pd
import pytest

from src.config import DELAY_THRESHOLD_H
from src.explain import stage_breakdown
from src.inputs import as_of
from src.scenario import apply_whatif, available, sweep

H = pd.Timedelta(hours=1)


@pytest.mark.parametrize("cp", [0, 1, 2])
def test_outputs_are_total_delay_and_observed_delay_is_added_once(held, cp):
    _, row, models = held
    P, r = models[cp], as_of(row, cp)
    o = P.predict(r).iloc[0]
    X = P.feature_row(r)
    assert o.delay_carry == pytest.approx(o.L_k)                                        # carry-forward = observed delay
    assert o.delay_gbm - o.L_k == pytest.approx(P.gbm.predict(X)[0])                    # model output + L_k, once
    assert o.delay_ridge - o.L_k == pytest.approx(P.ridge.predict(X)[0])
    inc = sum(o[f"chain_inc_s{j}"] for j in range(cp + 1, 4))
    assert o.delay_chain == pytest.approx(o.chain_L3)                                   # mean = observed + expected increments
    assert o.chain_L3 == pytest.approx(o.L_k + inc)
    assert o.c10 <= o.delay_chain_median <= o.c90 and o.delay_chain_median == o.c50
    assert 0 <= o.p_delay_chain <= 1 and 0 <= o.p_delay_clf <= 1


@pytest.mark.parametrize("cp", [0, 1, 2])
def test_stage_breakdown_sums_to_chain_mean_and_labels_observed_stages(held, cp):
    _, row, models = held
    o = models[cp].predict(as_of(row, cp)).iloc[0]
    bd = stage_breakdown(o, cp)
    assert bd.delta_h.sum() == pytest.approx(o.delay_chain)
    assert (bd.status[: cp + 1] != "expected").all() and (bd.status[cp + 1:] == "expected").all()
    assert bd.delta_h.iloc[0] == pytest.approx(o.chain_L0)


@pytest.mark.parametrize("cp", [0, 1, 2])
def test_future_information_is_not_used(held, cp):
    """Predictions are identical whether or not later milestones/outcome are present in the row."""
    _, row, models = held
    a, b = models[cp].predict(row), models[cp].predict(as_of(row, cp))
    np.testing.assert_allclose(a.to_numpy(float), b.to_numpy(float), atol=1e-9)


def test_prediction_does_not_depend_on_batch_composition(held, df):
    _, row, models = held
    r = as_of(row, 1)
    batch = pd.concat([r, as_of(df.iloc[:6], 1)], ignore_index=True)
    alone, in_batch = models[1].predict(r).iloc[0], models[1].predict(batch).iloc[0]
    np.testing.assert_allclose(alone.to_numpy(float), in_batch.to_numpy(float), atol=1e-9)


def test_whatif_only_offers_inputs_that_can_still_matter(held):
    _, row, _ = held
    assert set(available(0)) == {"congestion", "weather"} and set(available(1)) == {"congestion"} and available(2) == {}
    with pytest.raises(ValueError):
        apply_whatif(as_of(row, 1), 1, weather=0.9)
    with pytest.raises(ValueError):
        apply_whatif(as_of(row, 2), 2, congestion=0.9)


@pytest.mark.parametrize("cp,kind", [(0, "congestion"), (0, "weather"), (1, "congestion")])
def test_whatif_changes_prediction_in_the_expected_direction(held, cp, kind):
    _, row, models = held
    r = as_of(row, cp)
    s = sweep(models[cp], r, kind, np.linspace(0.35 if kind == "congestion" else 0.0, 0.95 if kind == "congestion" else 1.0, 7))
    assert s["median"].iloc[-1] > s["median"].iloc[0] + 1.0               
    assert s["median"].corr(pd.Series(s["x"]), method="spearman") > 0.9    
    pd.testing.assert_frame_equal(s, sweep(models[cp], r, kind, s["x"]))  


def test_whatif_at_current_value_reproduces_the_headline(held):
    _, row, models = held
    r = as_of(row, 1)
    cur = float(r["dest_congestion_arr"].iloc[0])
    s = sweep(models[1], r, "congestion", [cur])
    assert s["median"].iloc[0] == pytest.approx(models[1].predict(r)["delay_chain_median"].iloc[0])


def test_threshold_is_the_documented_one():
    assert DELAY_THRESHOLD_H == 12.0
