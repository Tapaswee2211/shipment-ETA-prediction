import pytest

from src.generator import generate_dataset
from src.inputs import holdout_split
from src.models import ETAPredictor

HELD_ID = "SH0060"


@pytest.fixture(scope="session")
def df():
    return generate_dataset(150, seed=3)


@pytest.fixture(scope="session")
def held(df):
    """(training frame without HELD_ID, the held-out row) and one predictor per checkpoint trained on that frame."""
    train, row = holdout_split(df, HELD_ID)
    models = {cp: ETAPredictor(cp, n_draws=80).fit(train) for cp in (0, 1, 2)}
    return train, row, models


def complete_shipment(df):
    """First shipment whose intermediate timestamps were all recorded (so nothing is imputed)."""
    ok = df[df.port_arrival_actual_ts.notna() & df.customs_cleared_actual_ts.notna()]
    return ok.iloc[[10]]
