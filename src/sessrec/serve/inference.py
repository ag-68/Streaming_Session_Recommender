"""Load the sequence model once and turn a history into five films."""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from sessrec.baselines import popularity_counts
from sessrec.config import Config
from sessrec.data import load_dataset
from sessrec.features import load_side_features
from sessrec.metrics import topk_ids
from sessrec.model import SASRec
from sessrec.paths import CHECKPOINT_FILE, POPULARITY_FILE, RUNTIME_FILE
from sessrec.scoring import left_pad


class Recommender:
    def __init__(
        self,
        model: SASRec,
        raw_to_index: dict[int, int],
        index_to_raw: dict[int, int],
        index_to_title: dict[int, str],
        popularity: np.ndarray,
        max_len: int,
        model_version: str,
    ) -> None:
        self.model = model
        self.model.eval()
        self.raw_to_index = raw_to_index
        self.index_to_raw = index_to_raw
        self.index_to_title = index_to_title
        self.popularity = popularity.astype(np.float64)
        self.max_len = max_len
        self.model_version = model_version
        self.n_items = int(popularity.shape[0] - 1)
        self.torch_threads = torch.get_num_threads()

    def warmup(self, rounds: int = 5) -> None:
        sample = [self.index_to_raw[item] for item in range(1, min(6, self.n_items + 1))]
        for _ in range(rounds):
            self.recommend(sample, 5)

    def recommend(self, history: list[int], k: int) -> dict:
        started = time.perf_counter()
        ignored: list[int] = []
        internal: list[int] = []
        for raw in history:
            mapped = self.raw_to_index.get(int(raw))
            if mapped is None:
                ignored.append(int(raw))
            else:
                internal.append(mapped)
        if not internal:
            scores = self.popularity.copy()
            strategy = "popularity_fallback"
        else:
            scores = self._sequence_scores(internal)
            strategy = "sasrec"
        scores[0] = -np.inf
        if internal:
            scores[internal] = -np.inf
        chosen = topk_ids(scores[None, :], k)[0]
        recommendations = []
        for rank, item in enumerate(chosen, start=1):
            if item <= 0:
                continue
            value = scores[item]
            recommendations.append(
                {
                    "rank": rank,
                    "movie_id": int(self.index_to_raw[int(item)]),
                    "title": self.index_to_title[int(item)],
                    "score": round(float(value), 6) if np.isfinite(value) else 0.0,
                }
            )
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "recommendations": recommendations,
            "strategy": strategy,
            "model_version": self.model_version,
            "ignored_ids": ignored,
            "latency_ms": round(elapsed_ms, 3),
        }

    def _sequence_scores(self, history: list[int]) -> np.ndarray:
        sequences = torch.from_numpy(left_pad([history], self.max_len))
        with torch.inference_mode():
            logits = self.model(sequences)[0, -1].cpu().numpy()
        return logits

    @classmethod
    def load(cls, config: Config | None = None) -> Recommender:
        config = config or Config()
        payload = torch.load(CHECKPOINT_FILE, map_location="cpu", weights_only=False)
        cfg = payload.get("config", {})
        side_dim = int(payload.get("side_dim", 0))
        model = SASRec(
            n_items=int(payload["n_items"]),
            max_len=int(cfg.get("max_len", config.max_len)),
            hidden_dim=int(cfg.get("hidden_dim", config.hidden_dim)),
            num_blocks=int(cfg.get("num_blocks", config.num_blocks)),
            num_heads=int(cfg.get("num_heads", config.num_heads)),
            dropout=0.0,
            side_dim=side_dim,
        )
        model.load_state_dict(payload["state_dict"], strict=False)
        if side_dim > 0:
            side = load_side_features()
            groups = payload.get("side_feature_groups") or list(side.groups)
            model.set_side_features(torch.from_numpy(side.masked(set(groups))))
        dataset = load_dataset()
        if POPULARITY_FILE.exists():
            popularity = np.load(POPULARITY_FILE)
        else:
            popularity = popularity_counts(dataset.train, dataset.n_items)
        recommender = cls(
            model=model,
            raw_to_index=dataset.raw_to_index,
            index_to_raw=dataset.index_to_raw,
            index_to_title=dataset.index_to_title,
            popularity=popularity,
            max_len=config.max_len,
            model_version=str(payload.get("model_version", config.model_version)),
        )
        recommender._choose_threads()
        return recommender

    def _choose_threads(self) -> None:
        """Keep the thread count with the lower p95 on a short local sample."""

        sample = [self.index_to_raw[item] for item in range(1, min(21, self.n_items + 1))]
        cores = max(1, os.cpu_count() or 1)
        best_threads = 1
        best_p95 = float("inf")
        for threads in (1, cores):
            torch.set_num_threads(threads)
            for _ in range(5):
                self.recommend(sample, 5)
            samples = []
            for _ in range(20):
                started = time.perf_counter()
                self.recommend(sample, 5)
                samples.append((time.perf_counter() - started) * 1000)
            p95 = float(np.percentile(samples, 95))
            if p95 < best_p95:
                best_p95 = p95
                best_threads = threads
        torch.set_num_threads(best_threads)
        self.torch_threads = best_threads
        RUNTIME_FILE.write_text(
            json.dumps({"torch_threads": best_threads, "startup_p95_ms": best_p95}),
            encoding="utf-8",
        )
