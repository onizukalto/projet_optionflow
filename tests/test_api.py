import joblib
import pytest
from fastapi.testclient import TestClient

from src.features import FLOW, PRICE, build_features
from src.model import make_model
from tests.test_features import make_day

PAYLOAD = {f: 0.0 for f in PRICE + FLOW} | {
    "call_share_vol_5m": 0.6, "call_share_vol_15m": 0.55, "call_share_vol_30m": 0.5,
    "total_volume_5m": 1000, "total_volume_15m": 3000, "total_volume_30m": 6000,
    "volume_vs_day_avg": 1.2,
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    df = build_features(*make_day())
    model = make_model().fit(df[PRICE + FLOW], df["target"])
    path = tmp_path / "model.joblib"
    joblib.dump({"model": model, "features": PRICE + FLOW}, path)

    import api.main as main
    monkeypatch.setattr(main, "MODEL_PATH", path)
    monkeypatch.setattr(main, "_bundle", None)
    return TestClient(main.app)


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict(client):
    r = client.post("/predict", json=PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["proba_up"] <= 1
    assert body["direction"] in {"UP", "DOWN"}


def test_predict_rejects_invalid_input(client):
    r = client.post("/predict", json=PAYLOAD | {"call_share_vol_5m": 3})
    assert r.status_code == 422
