"""Popularity and item-to-item similarity, fit on training sequences only."""

from __future__ import annotations

import json

import numpy as np

from sessrec.config import Config
from sessrec.data import load_dataset
from sessrec.metrics import mask_seen, summarise
from sessrec.paths import BASELINES_FILE, ITEM_KNN_FILE, POPULARITY_FILE


def popularity_counts(train: list[list[int]], n_items: int) -> np.ndarray:
    """How many training sequences contain each film. Index 0 is unused."""

    counts = np.zeros(n_items + 1, dtype=np.float64)
    for sequence in train:
        counts[sequence] += 1
    counts[0] = 0
    return counts


def fit_item_knn(
    train: list[list[int]], n_items: int, n_neighbors: int
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the nearest films by cosine similarity of who liked them.

    Returns neighbor ids and similarities with shape (n_items + 1, k). Row 0 is unused.
    A film is not a neighbor of itself.
    """

    presence = np.zeros((len(train), n_items), dtype=np.float32)
    for user, sequence in enumerate(train):
        presence[user, np.asarray(sequence, dtype=np.int64) - 1] = 1.0
    norms = np.linalg.norm(presence, axis=0)
    safe = norms.copy()
    safe[safe == 0] = 1.0
    k = min(n_neighbors, max(n_items - 1, 1))
    neighbors = np.zeros((n_items + 1, k), dtype=np.int64)
    sims = np.zeros((n_items + 1, k), dtype=np.float32)
    if n_items <= 1:
        return neighbors, sims
    gram_right = presence
    chunk = 512
    for start in range(0, n_items, chunk):
        end = min(start + chunk, n_items)
        block = (presence[:, start:end].T @ gram_right) / safe[start:end, None] / safe[None, :]
        for offset, item in enumerate(range(start, end)):
            block[offset, item] = -np.inf
        chosen = np.argpartition(block, -k, axis=1)[:, -k:]
        chosen_scores = np.take_along_axis(block, chosen, axis=1)
        order = np.argsort(-chosen_scores, axis=1)
        top_index = np.take_along_axis(chosen, order, axis=1)
        top_scores = np.take_along_axis(chosen_scores, order, axis=1)
        top_scores = np.where(np.isfinite(top_scores), np.maximum(top_scores, 0), 0)
        neighbors[start + 1 : end + 1] = top_index + 1
        sims[start + 1 : end + 1] = top_scores.astype(np.float32)
    return neighbors, sims


def score_popularity(n_rows: int, popularity: np.ndarray) -> np.ndarray:
    return np.tile(popularity, (n_rows, 1))


def score_item_knn(
    histories: list[list[int]],
    neighbors: np.ndarray,
    sims: np.ndarray,
    n_items: int,
    popularity: np.ndarray,
) -> np.ndarray:
    """Add the stored similarities from each known film in the history.

    An empty history, or a history of only unknown films, uses popularity.
    """

    scores = np.zeros((len(histories), n_items + 1), dtype=np.float64)
    for row, history in enumerate(histories):
        known = [item for item in history if 1 <= item <= n_items]
        if not known:
            scores[row] = popularity
            continue
        for item in known:
            scores[row, neighbors[item]] += sims[item]
    return scores


def save_fitted(popularity: np.ndarray, neighbors: np.ndarray, sims: np.ndarray) -> None:
    np.save(POPULARITY_FILE, popularity)
    np.savez_compressed(ITEM_KNN_FILE, neighbors=neighbors, sims=sims)


def load_fitted() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    popularity = np.load(POPULARITY_FILE)
    packed = np.load(ITEM_KNN_FILE)
    return popularity, packed["neighbors"], packed["sims"]


def metrics_for(
    histories: list[list[int]],
    targets: np.ndarray,
    scores: np.ndarray,
    k: int,
    n_items: int,
) -> dict:
    masked = mask_seen(scores, histories)
    summary = summarise(masked, targets, k, n_items)
    return {key: summary[key] for key in ("ndcg@10", "hr@10", "coverage@10")}


def fit_and_score_validation(config: Config | None = None) -> dict:
    """Fit on training sequences and score the validation question only.

    The test film is not used here. The shared test pass happens later.
    """

    config = config or Config()
    dataset = load_dataset()
    popularity = popularity_counts(dataset.train, dataset.n_items)
    neighbors, sims = fit_item_knn(dataset.train, dataset.n_items, config.item_knn_neighbors)
    save_fitted(popularity, neighbors, sims)
    histories = dataset.validation_histories()
    popularity_scores = score_popularity(len(histories), popularity)
    knn_scores = score_item_knn(histories, neighbors, sims, dataset.n_items, popularity)
    report = {
        "split": "validation",
        "k": config.k,
        "item_knn_neighbors": config.item_knn_neighbors,
        "models": {
            "popularity": metrics_for(
                histories, dataset.val, popularity_scores, config.k, dataset.n_items
            ),
            "item_knn": metrics_for(histories, dataset.val, knn_scores, config.k, dataset.n_items),
        },
    }
    BASELINES_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    report = fit_and_score_validation()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
