import numpy as np
import pandas as pd
import pytest

from src.features import FLOW, HORIZON, PRICE, build_features


def make_day(seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-09-01 09:30", periods=390, freq="min", tz="America/New_York")
    qqq = pd.DataFrame({"timestamp": ts, "open": 700.0,
                        "close": 700 + rng.normal(0, 0.3, 390).cumsum()})
    options = pd.concat(
        pd.DataFrame({"timestamp": ts, "volume": rng.integers(1, 500, 390),
                      "vwap": rng.uniform(1, 5, 390), "contract_type": t})
        for t in ["call", "put"])
    return qqq, options


def test_all_features_present():
    df = build_features(*make_day())
    assert set(PRICE + FLOW + ["target"]) <= set(df.columns)
    assert not df[PRICE + FLOW].isna().any().any()


def test_target_definition():
    df = build_features(*make_day())
    assert ((df["future_price"] > df["price"]).astype(int) == df["target"]).all()


def test_target_stays_inside_the_day():
    df = build_features(*make_day())
    last_minute_with_target = df.index.max()
    assert last_minute_with_target.strftime("%H:%M") <= f"15:{59 - HORIZON:02d}"


def test_no_leakage():
    """Modifier le futur ne doit changer aucune feature du passé."""
    qqq, options = make_day()
    cut = qqq["timestamp"].iloc[200]
    base = build_features(qqq, options)

    qqq2 = qqq.copy()
    qqq2.loc[qqq2["timestamp"] > cut, "close"] *= 1.05
    options2 = options.copy()
    options2.loc[options2["timestamp"] > cut, "volume"] *= 10
    changed = build_features(qqq2, options2)

    past = base.index[base.index <= cut]
    pd.testing.assert_frame_equal(base.loc[past, PRICE + FLOW], changed.loc[past, PRICE + FLOW])


def test_flow_bounds():
    df = build_features(*make_day())
    for w in [5, 15, 30]:
        assert df[f"call_share_vol_{w}m"].between(0, 1).all()
        assert df[f"net_premium_share_{w}m"].between(-1, 1).all()
