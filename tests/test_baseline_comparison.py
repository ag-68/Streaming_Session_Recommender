"""The saved test scores. SASRec must beat popularity. Item-to-item may win."""

import json
import math

import pytest

from sessrec.paths import METRICS_FILE


def test_saved_comparison() -> None:
    if not METRICS_FILE.exists():
        pytest.skip("run evaluate.py before this comparison")
    report = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    models = report["models"]
    for name in ("popularity", "item_knn", "sasrec"):
        assert math.isfinite(models[name]["ndcg@10"])
        assert math.isfinite(models[name]["hr@10"])
        assert math.isfinite(models[name]["coverage@10"])
    assert models["sasrec"]["ndcg@10"] > models["popularity"]["ndcg@10"]
