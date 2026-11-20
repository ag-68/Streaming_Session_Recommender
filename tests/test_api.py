"""The HTTP contract, using a tiny catalog so the test does not need the checkpoint."""

import numpy as np
import torch
from fastapi.testclient import TestClient

from sessrec.model import SASRec
from sessrec.serve.app import create_app
from sessrec.serve.inference import Recommender


def _client() -> TestClient:
    torch.manual_seed(0)
    model = SASRec(n_items=6, max_len=8, hidden_dim=8, num_blocks=1, num_heads=1, dropout=0.0)
    raw_to_index = {11: 1, 22: 2, 33: 3, 44: 4, 55: 5, 66: 6}
    index_to_raw = {index: raw for raw, index in raw_to_index.items()}
    titles = {index: f"Film {raw}" for index, raw in index_to_raw.items()}
    popularity = np.array([0, 1, 10, 8, 6, 4, 2], dtype=np.float64)
    recommender = Recommender(model, raw_to_index, index_to_raw, titles, popularity, 8, "tiny")
    return TestClient(create_app(recommender))


def test_health() -> None:
    response = _client().get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_version"] == "tiny"


def test_empty_history_uses_popularity_in_count_order() -> None:
    response = _client().post("/recommend", json={})
    assert response.status_code == 200
    body = response.json()
    assert body["strategy"] == "popularity_fallback"
    assert len(body["recommendations"]) == 5
    assert [item["movie_id"] for item in body["recommendations"]] == [22, 33, 44, 55, 66]


def test_seen_films_are_left_out_and_unknown_ids_are_reported() -> None:
    response = _client().post("/recommend", json={"history": [22, 999999], "k": 5})
    body = response.json()
    ids = [item["movie_id"] for item in body["recommendations"]]
    assert 22 not in ids
    assert len(ids) == len(set(ids)) == 5
    assert body["ignored_ids"] == [999999]
    assert body["strategy"] == "sasrec"


def test_bad_k_and_non_integers_are_rejected() -> None:
    client = _client()
    assert client.post("/recommend", json={"k": 0}).status_code == 422
    assert client.post("/recommend", json={"k": 51}).status_code == 422
    assert client.post("/recommend", json={"history": ["nope"]}).status_code == 422
    long_history = list(range(501))
    assert client.post("/recommend", json={"history": long_history}).status_code == 422
