"""MovieLens loading, the 4-and-5-star rule, and the leave-one-out split."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sessrec.config import Config
from sessrec.paths import (
    ARTIFACTS,
    DATA_STATS_FILE,
    ITEM_MAP_FILE,
    MOVIES_FILE,
    PROCESSED_DIR,
    RATINGS_FILE,
    SEQUENCES_FILE,
    USER_DISLIKES_FILE,
)


@dataclass
class Dataset:
    train: list[list[int]]
    val: np.ndarray
    test: np.ndarray
    raw_user_ids: np.ndarray
    n_items: int
    raw_to_index: dict[int, int]
    index_to_raw: dict[int, int]
    index_to_title: dict[int, str]
    stats: dict
    dislikes_hard: list[list[int]] = field(default_factory=list)
    dislikes_soft: list[list[int]] = field(default_factory=list)
    # Each event is (item_id, polarity) with polarity 1 = like, 0 = dislike.
    signed_train: list[list[tuple[int, int]]] = field(default_factory=list)
    signed_before_test: list[list[tuple[int, int]]] = field(default_factory=list)

    def test_histories(self) -> list[list[int]]:
        """Training films plus the validation film. The test film stays hidden."""

        return [train + [int(val)] for train, val in zip(self.train, self.val, strict=True)]

    def validation_histories(self) -> list[list[int]]:
        return [list(train) for train in self.train]


def read_ratings(path=RATINGS_FILE) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    frame["_row"] = np.arange(len(frame), dtype=np.int64)
    return frame


def read_movies(path=MOVIES_FILE) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        header=None,
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )


def _map_user_films(
    group: pd.DataFrame,
    raw_to_index: dict[int, int],
    max_rating: int,
) -> list[int]:
    chosen = group.loc[group["rating"] <= max_rating, "movie_id"]
    out: list[int] = []
    seen: set[int] = set()
    for raw in chosen.tolist():
        index = raw_to_index.get(int(raw))
        if index is None or index in seen:
            continue
        seen.add(index)
        out.append(index)
    return out


def _signed_events(
    group: pd.DataFrame,
    raw_to_index: dict[int, int],
    min_like: int,
    hard_max: int,
) -> list[tuple[int, int]]:
    """Timeline of in-catalog likes (+) and hard dislikes (−); 3★ dropped."""

    events: list[tuple[int, int]] = []
    ordered = group.sort_values(["timestamp", "_row"], kind="mergesort")
    for row in ordered.itertuples(index=False):
        rating = int(row.rating)
        index = raw_to_index.get(int(row.movie_id))
        if index is None:
            continue
        if rating >= min_like:
            events.append((index, 1))
        elif rating <= hard_max:
            events.append((index, 0))
    return events


def prepare_frame(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    config: Config | None = None,
) -> Dataset:
    """Keep liked films, drop short histories, and hide each viewer's last two films."""

    config = config or Config()
    raw_ratings = int(len(ratings))
    raw_users = int(ratings["user_id"].nunique())
    liked = ratings.loc[ratings["rating"] >= config.min_rating].copy()
    users_before = int(liked["user_id"].nunique())
    per_user = liked.groupby("user_id").size()
    kept_users = per_user[per_user >= config.min_positives].index
    users_dropped = int((per_user < config.min_positives).sum())
    liked = liked.loc[liked["user_id"].isin(kept_users)]
    liked = liked.sort_values(["user_id", "timestamp", "_row"], kind="mergesort")

    raw_ids = np.sort(liked["movie_id"].unique())
    raw_to_index = {int(raw): index + 1 for index, raw in enumerate(raw_ids)}
    index_to_raw = {index: raw for raw, index in raw_to_index.items()}
    titles = {int(row.movie_id): str(row.title) for row in movies.itertuples(index=False)}
    index_to_title = {
        index: titles.get(raw, f"Movie {raw}") for index, raw in index_to_raw.items()
    }
    liked = liked.copy()
    liked["item"] = liked["movie_id"].map(raw_to_index).astype(np.int64)

    all_kept = ratings.loc[ratings["user_id"].isin(kept_users)].copy()

    train: list[list[int]] = []
    val: list[int] = []
    test: list[int] = []
    raw_user_ids: list[int] = []
    dislikes_hard: list[list[int]] = []
    dislikes_soft: list[list[int]] = []
    signed_train: list[list[tuple[int, int]]] = []
    signed_before_test: list[list[tuple[int, int]]] = []
    for user_id, group in liked.groupby("user_id", sort=False):
        items = group["item"].tolist()
        timestamps = group["timestamp"].to_numpy()
        if np.any(timestamps[1:] < timestamps[:-1]):
            raise ValueError(f"timestamps went backwards for user {user_id}")
        train.append(items[:-2])
        val.append(int(items[-2]))
        test.append(int(items[-1]))
        raw_user_ids.append(int(user_id))
        user_rows = all_kept.loc[all_kept["user_id"] == user_id]
        hard = _map_user_films(user_rows, raw_to_index, config.hard_dislike_max)
        soft = _map_user_films(user_rows, raw_to_index, config.soft_dislike_max)
        dislikes_hard.append(hard)
        dislikes_soft.append(soft)
        signed = _signed_events(
            user_rows,
            raw_to_index,
            config.min_rating,
            config.hard_dislike_max,
        )
        like_positions = [i for i, (_item, pol) in enumerate(signed) if pol == 1]
        if len(like_positions) < 2:
            signed_train.append(signed)
            signed_before_test.append(signed)
        else:
            signed_train.append(signed[: like_positions[-2]])
            signed_before_test.append(signed[: like_positions[-1]])

    stats = {
        "raw_ratings": raw_ratings,
        "raw_users": raw_users,
        "positive_ratings": int(len(liked)),
        "users_before_min_filter": users_before,
        "users_dropped": users_dropped,
        "users": len(train),
        "items": len(raw_to_index),
        "min_rating": config.min_rating,
        "min_positives": config.min_positives,
        "users_with_hard_dislike": int(sum(1 for row in dislikes_hard if row)),
        "users_with_soft_dislike": int(sum(1 for row in dislikes_soft if row)),
    }
    return Dataset(
        train=train,
        val=np.asarray(val, dtype=np.int64),
        test=np.asarray(test, dtype=np.int64),
        raw_user_ids=np.asarray(raw_user_ids, dtype=np.int64),
        n_items=len(raw_to_index),
        raw_to_index=raw_to_index,
        index_to_raw=index_to_raw,
        index_to_title=index_to_title,
        stats=stats,
        dislikes_hard=dislikes_hard,
        dislikes_soft=dislikes_soft,
        signed_train=signed_train,
        signed_before_test=signed_before_test,
    )


def assert_raw_file(ratings: pd.DataFrame) -> None:
    if len(ratings) != 1_000_209:
        raise AssertionError(f"expected 1,000,209 ratings, found {len(ratings)}")
    users = int(ratings["user_id"].nunique())
    if users != 6_040:
        raise AssertionError(f"expected 6,040 users, found {users}")


def _pack_ragged(rows: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    values: list[int] = []
    for index, row in enumerate(rows):
        values.extend(row)
        offsets[index + 1] = len(values)
    return offsets, np.asarray(values, dtype=np.int64)


def _unpack_ragged(offsets: np.ndarray, values: np.ndarray) -> list[list[int]]:
    return [
        values[offsets[index] : offsets[index + 1]].tolist()
        for index in range(len(offsets) - 1)
    ]


def save_dataset(dataset: Dataset) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    offsets, values = _pack_ragged(dataset.train)
    np.savez_compressed(
        SEQUENCES_FILE,
        offsets=offsets,
        values=values,
        val=dataset.val,
        test=dataset.test,
        raw_user_ids=dataset.raw_user_ids,
        n_items=np.asarray([dataset.n_items], dtype=np.int64),
    )
    hard_off, hard_val = _pack_ragged(dataset.dislikes_hard)
    soft_off, soft_val = _pack_ragged(dataset.dislikes_soft)
    def pack_signed(rows: list[list[tuple[int, int]]]):
        items: list[int] = []
        pols: list[int] = []
        off = np.zeros(len(rows) + 1, dtype=np.int64)
        for index, events in enumerate(rows):
            for item, pol in events:
                items.append(item)
                pols.append(pol)
            off[index + 1] = len(items)
        return off, np.asarray(items, dtype=np.int64), np.asarray(pols, dtype=np.int64)

    train_off, train_items, train_pols = pack_signed(dataset.signed_train)
    test_off, test_items, test_pols = pack_signed(dataset.signed_before_test)
    np.savez_compressed(
        USER_DISLIKES_FILE,
        hard_offsets=hard_off,
        hard_values=hard_val,
        soft_offsets=soft_off,
        soft_values=soft_val,
        signed_offsets=train_off,
        signed_items=train_items,
        signed_pols=train_pols,
        signed_test_offsets=test_off,
        signed_test_items=test_items,
        signed_test_pols=test_pols,
        raw_user_ids=dataset.raw_user_ids,
    )
    item_map = {
        "raw_to_index": {str(raw): index for raw, index in dataset.raw_to_index.items()},
        "index_to_raw": {str(index): raw for index, raw in dataset.index_to_raw.items()},
        "index_to_title": {str(index): title for index, title in dataset.index_to_title.items()},
    }
    ITEM_MAP_FILE.write_text(json.dumps(item_map), encoding="utf-8")
    DATA_STATS_FILE.write_text(json.dumps(dataset.stats, indent=2), encoding="utf-8")


def load_dataset() -> Dataset:
    packed = np.load(SEQUENCES_FILE)
    offsets = packed["offsets"]
    values = packed["values"]
    train = _unpack_ragged(offsets, values)
    item_map = json.loads(ITEM_MAP_FILE.read_text(encoding="utf-8"))
    stats = json.loads(DATA_STATS_FILE.read_text(encoding="utf-8"))
    raw_to_index = {int(raw): int(index) for raw, index in item_map["raw_to_index"].items()}
    index_to_raw = {int(index): int(raw) for index, raw in item_map["index_to_raw"].items()}
    index_to_title = {int(index): title for index, title in item_map["index_to_title"].items()}
    dislikes_hard: list[list[int]] = [[] for _ in train]
    dislikes_soft: list[list[int]] = [[] for _ in train]
    signed_train: list[list[tuple[int, int]]] = [[] for _ in train]
    signed_before_test: list[list[tuple[int, int]]] = [[] for _ in train]

    def unpack_signed(off_key: str, item_key: str, pol_key: str, extra) -> list[list[tuple[int, int]]]:
        if off_key not in extra:
            return [[] for _ in train]
        signed_off = extra[off_key]
        signed_items = extra[item_key]
        signed_pols = extra[pol_key]
        rows = []
        for index in range(len(signed_off) - 1):
            start = int(signed_off[index])
            end = int(signed_off[index + 1])
            rows.append(
                [
                    (int(signed_items[pos]), int(signed_pols[pos]))
                    for pos in range(start, end)
                ]
            )
        return rows

    if USER_DISLIKES_FILE.exists():
        extra = np.load(USER_DISLIKES_FILE)
        dislikes_hard = _unpack_ragged(extra["hard_offsets"], extra["hard_values"])
        dislikes_soft = _unpack_ragged(extra["soft_offsets"], extra["soft_values"])
        signed_train = unpack_signed("signed_offsets", "signed_items", "signed_pols", extra)
        signed_before_test = unpack_signed(
            "signed_test_offsets", "signed_test_items", "signed_test_pols", extra
        )
    return Dataset(
        train=train,
        val=packed["val"].astype(np.int64),
        test=packed["test"].astype(np.int64),
        raw_user_ids=packed["raw_user_ids"].astype(np.int64),
        n_items=int(packed["n_items"][0]),
        raw_to_index=raw_to_index,
        index_to_raw=index_to_raw,
        index_to_title=index_to_title,
        stats=stats,
        dislikes_hard=dislikes_hard,
        dislikes_soft=dislikes_soft,
        signed_train=signed_train,
        signed_before_test=signed_before_test,
    )


def build_from_files(config: Config | None = None) -> Dataset:
    ratings = read_ratings()
    assert_raw_file(ratings)
    dataset = prepare_frame(ratings, read_movies(), config)
    save_dataset(dataset)
    return dataset


def main() -> None:
    dataset = build_from_files()
    print(json.dumps(dataset.stats, indent=2))


if __name__ == "__main__":
    main()
