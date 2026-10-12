import numpy as np

from src.generator import (
    build_routes,
    _port_series,
    generate_dataset,
)


def test_build_routes():
    routes = build_routes()

    assert len(routes) == 10
    assert len({r["route_id"] for r in routes}) == 10


def test_port_series():
    series = _port_series(np.random.default_rng(42))

    assert len(series) == 7

    for values in series.values():
        assert len(values) == 360
        assert np.isfinite(values).all()
        assert np.all(values >= 0.35)
        assert np.all(values <= 0.95)


def test_dataset_reproducibility():
    df1 = generate_dataset(n=10, seed=42)
    df2 = generate_dataset(n=10, seed=42)

    assert df1.equals(df2)


def test_shapley_efficiency():
    df = generate_dataset(n=10, seed=42)

    phi_cols = [
        f"true_phi_{s}"
        for s in [
            "origin",
            "weather",
            "route_noise",
            "congestion",
            "processing",
            "inspection",
            "inland",
        ]
    ]

    attributed = df[phi_cols].sum(axis=1)
    expected = df["true_L3"] - df["true_baseline"]

    assert np.allclose(attributed, expected)

