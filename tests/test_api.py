from fastapi.testclient import TestClient

from review_intel.serving import api
from review_intel.core.predictors import Prediction, Predictor
from tests.conftest import make_analysis


class Fake(Predictor):
    name = "fake"

    def __init__(self, analysis):
        self.analysis = analysis

    def predict(self, texts):
        return [Prediction(self.analysis, confidence={"entrega": 0.9}) for _ in texts]


def client_with(analysis, monkeypatch):
    monkeypatch.setattr(api, "get_predictor", lambda: Fake(analysis))
    return TestClient(api.app)


def test_extract_ok(monkeypatch):
    c = client_with(make_analysis(entrega="negativo"), monkeypatch)
    r = c.post("/extract", json={"text": "atrasou"})
    assert r.status_code == 200
    body = r.json()
    assert body["analysis"]["aspectos"]["entrega"] == "negativo" and body["approach"] == "fake"


def test_extract_invalid_model_output(monkeypatch):
    assert client_with(None, monkeypatch).post("/extract", json={"text": "x"}).status_code == 502


def test_extract_validates_input(monkeypatch):
    assert client_with(make_analysis(), monkeypatch).post("/extract", json={"text": ""}).status_code == 422


def test_health():
    assert TestClient(api.app).get("/health").json() == {"status": "ok"}
