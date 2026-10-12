import numpy as np
import pytest

from examples.make_foreign_csv import MAPPING, to_foreign
from src.adapters import standardize
from src.features import add_lateness
from src.generator import generate_dataset
from src.models import ETAPredictor


@pytest.fixture(scope="module")
def df():
    return generate_dataset(120, seed=11)


def test_foreign_format_roundtrip(df):
    """Different column names, day-first strings, no weather/ground-truth -> same lateness after adapting."""
    clean, notes = standardize(to_foreign(df), MAPPING)
    assert len(clean) == len(df)
    got = add_lateness(clean).set_index("shipment_id")["L3"]
    exp = df.set_index("shipment_id")["true_L3"]
    np.testing.assert_allclose(got.loc[exp.index], exp, atol=0.01)
    assert clean["weather_fcst"].isna().all()                       # unavailable signal stays unknown


def test_models_run_on_data_with_missing_signal_columns(df):
    clean, _ = standardize(to_foreign(df), MAPPING)
    tr, te = clean.iloc[:90], clean.iloc[90:]
    o = ETAPredictor(1, n_draws=40).fit(tr).predict(te)
    assert np.isfinite(o.to_numpy()).all()


def test_adapter_rejects_bad_input(df):
    raw = to_foreign(df)
    bad = {**MAPPING, "columns": {**MAPPING["columns"], "origin": "NOPE"}}
    with pytest.raises(ValueError, match="not found"):
        standardize(raw, bad)
    with pytest.raises(ValueError, match="usable rows"):
        standardize(raw.iloc[:20], MAPPING)
