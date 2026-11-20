"""Split rules, on a tiny table and on the raw MovieLens file when it is present."""

import numpy as np
import pandas as pd

from sessrec.baselines import fit_item_knn, popularity_counts, score_item_knn
from sessrec.config import Config
from sessrec.data import prepare_frame, read_ratings
from sessrec.paths import RATINGS_FILE


def _ratings() -> pd.DataFrame:
    rows = [
        (1, 10, 5, 1),
        (1, 12, 1, 2),
        (1, 11, 5, 3),
        (1, 13, 4, 4),
        (1, 14, 4, 4),
        (1, 15, 5, 5),
        (2, 10, 5, 1),
        (2, 11, 4, 2),
        (2, 13, 3, 3),
        (3, 10, 5, 1),
        (3, 11, 5, 2),
        (3, 13, 5, 3),
        (3, 14, 5, 4),
        (3, 16, 4, 5),
    ]
    frame = pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])
    frame["_row"] = np.arange(len(frame))
    return frame


def _movies() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movie_id": [10, 11, 12, 13, 14, 15, 16],
            "title": ["Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen"],
            "genres": ["Drama"] * 7,
        }
    )


def test_split_hides_the_test_film_and_drops_low_ratings() -> None:
    dataset = prepare_frame(_ratings(), _movies(), Config())
    assert dataset.stats["users"] == 2
    assert dataset.stats["users_dropped"] == 1
    for train, val, test in zip(dataset.train, dataset.val, dataset.test, strict=True):
        assert test not in train
        assert val not in train
        assert val in train + [val]
        assert test not in train + [val]
    titles = set(dataset.index_to_title.values())
    assert "Twelve" not in titles
    assert "Thirteen" in titles
    assert dataset.n_items == len(dataset.raw_to_index)
    assert set(dataset.raw_to_index.values()) == set(range(1, dataset.n_items + 1))


def test_tied_timestamps_keep_file_order() -> None:
    dataset = prepare_frame(_ratings(), _movies(), Config())
    user = dataset.raw_user_ids.tolist().index(1)
    sequence = dataset.train[user] + [int(dataset.val[user]), int(dataset.test[user])]
    raw = [dataset.index_to_raw[item] for item in sequence]
    assert raw == [10, 11, 13, 14, 15]


def test_popularity_and_neighbors_use_training_films_only() -> None:
    train = [[1, 2], [1, 2], [1]]
    counts = popularity_counts(train, n_items=3)
    assert counts[3] == 0
    assert counts[1] == 3
    neighbors, sims = fit_item_knn(train, n_items=3, n_neighbors=2)
    scores = score_item_knn([[1]], neighbors, sims, 3, counts)
    assert scores[0, 2] > scores[0, 3]


def test_raw_movielens_file_when_present() -> None:
    if not RATINGS_FILE.exists():
        return
    ratings = read_ratings()
    assert len(ratings) == 1_000_209
    assert ratings["user_id"].nunique() == 6_040
