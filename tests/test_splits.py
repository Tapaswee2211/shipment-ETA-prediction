import numpy as np
import pytest

import src.evaluate as ev
from src.evaluate import run_cv, splits


@pytest.mark.parametrize("kind", ["random", "route", "time"])
def test_train_and_test_shipments_never_overlap(df, kind):
    for _, tr, te in splits(df, kind, n_repeats=2):
        assert set(df.shipment_id.iloc[tr]).isdisjoint(df.shipment_id.iloc[te]) and len(te) > 0


def test_random_cv_tests_every_shipment_once_per_repeat(df):
    seen = np.concatenate([te for _, _, te in splits(df, "random", n_repeats=1)])
    assert sorted(seen) == list(range(len(df)))


def test_route_split_hides_the_whole_route(df):
    for rid, tr, te in splits(df, "route"):
        assert (df.route_id.iloc[te] == rid).all() and rid not in set(df.route_id.iloc[tr])


def test_time_split_trains_only_on_delivered_past(df):
    for _, tr, te in splits(df, "time"):
        assert df.delivery_actual_ts.iloc[tr].max() <= df.dep_sched_ts.iloc[te].min()


def test_all_checkpoints_use_the_same_shipments_in_a_split(df, monkeypatch):
    seen = {}

    class Fake:
        def __init__(self, cp, **kw):
            self.cp = cp

        def fit(self, d):
            seen.setdefault(self.cp, []).append(tuple(d.shipment_id))
            return self

        def predict(self, d):
            import pandas as pd
            return pd.DataFrame({"delay_sched": 0.0}, index=d.index)

    monkeypatch.setattr(ev, "ETAPredictor", Fake)
    run_cv(df, "random", n_repeats=1, full=False)
    assert seen[0] == seen[1] == seen[2]
