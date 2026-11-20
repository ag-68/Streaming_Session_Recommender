"""In-process response time for the trained model."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sessrec.paths import CHECKPOINT_FILE, ITEM_MAP_FILE
from sessrec.serve.app import create_app


@pytest.mark.latency
def test_p95_under_50_ms() -> None:
    if not CHECKPOINT_FILE.exists() or not ITEM_MAP_FILE.exists():
        pytest.skip("checkpoint not built yet")
    client = TestClient(create_app())
    health = client.get("/health")
    assert health.status_code == 200
    body = {"history": [1, 260, 1196, 1210], "k": 5}
    for _ in range(50):
        client.post("/recommend", json=body)
    samples = []
    for _ in range(500):
        response = client.post("/recommend", json=body)
        assert response.status_code == 200
        samples.append(response.json()["latency_ms"])
    assert float(np.percentile(samples, 95)) < 50
