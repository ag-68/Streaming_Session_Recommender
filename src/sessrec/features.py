"""Side features for each film: genres, decade, and training popularity."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sessrec.baselines import popularity_counts
from sessrec.data import Dataset, read_movies
from sessrec.paths import POPULARITY_FILE, SIDE_FEATURES_FILE

YEAR_RE = re.compile(r"\((\d{4})\)\s*$")
ML1M_GENRES = [
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


@dataclass
class SideFeatures:
    """Per-film matrix with named groups for later ablations."""

    matrix: np.ndarray  # shape (n_items + 1, side_dim), row 0 is padding
    groups: dict[str, list[int]]
    feature_names: list[str]

    @property
    def side_dim(self) -> int:
        return int(self.matrix.shape[1])

    def masked(self, active: set[str] | None = None) -> np.ndarray:
        """Zero out feature groups not in ``active``. None keeps every group."""

        if active is None:
            return self.matrix.copy()
        out = np.zeros_like(self.matrix)
        for name in active:
            for index in self.groups[name]:
                out[:, index] = self.matrix[:, index]
        return out


def _year_from_title(title: str) -> int | None:
    match = YEAR_RE.search(title)
    if not match:
        return None
    return int(match.group(1))


def build_side_features(
    dataset: Dataset,
    movies: pd.DataFrame | None = None,
    popularity: np.ndarray | None = None,
) -> SideFeatures:
    """Build g_j for every remapped film id. Padding row stays zeros."""

    movies = read_movies() if movies is None else movies
    if popularity is None:
        if POPULARITY_FILE.exists():
            popularity = np.load(POPULARITY_FILE)
        else:
            popularity = popularity_counts(dataset.train, dataset.n_items)

    genre_to_col = {name: index for index, name in enumerate(ML1M_GENRES)}
    years: dict[int, int] = {}
    for row in movies.itertuples(index=False):
        raw = int(row.movie_id)
        if raw not in dataset.raw_to_index:
            continue
        year = _year_from_title(str(row.title))
        if year is not None:
            years[dataset.raw_to_index[raw]] = year

    if years:
        min_decade = (min(years.values()) // 10) * 10
        max_decade = (max(years.values()) // 10) * 10
        decades = list(range(min_decade, max_decade + 1, 10))
    else:
        decades = [1990]

    decade_to_col = {decade: index for index, decade in enumerate(decades)}
    n_genre = len(ML1M_GENRES)
    n_decade = len(decades)
    side_dim = n_genre + n_decade + 1
    matrix = np.zeros((dataset.n_items + 1, side_dim), dtype=np.float32)

    by_raw = {int(row.movie_id): row for row in movies.itertuples(index=False)}
    for raw, index in dataset.raw_to_index.items():
        row = by_raw.get(raw)
        if row is None:
            continue
        for genre in str(row.genres).split("|"):
            col = genre_to_col.get(genre)
            if col is not None:
                matrix[index, col] = 1.0
        year = years.get(index)
        if year is not None:
            decade = (year // 10) * 10
            matrix[index, n_genre + decade_to_col[decade]] = 1.0

    pop = np.asarray(popularity, dtype=np.float64)
    log_pop = np.log1p(pop)
    scale = float(log_pop.max()) if log_pop.max() > 0 else 1.0
    matrix[:, n_genre + n_decade] = (log_pop / scale).astype(np.float32)
    matrix[0] = 0.0

    feature_names = (
        [f"genre:{name}" for name in ML1M_GENRES]
        + [f"decade:{decade}" for decade in decades]
        + ["log_popularity"]
    )
    groups = {
        "genre": list(range(0, n_genre)),
        "decade": list(range(n_genre, n_genre + n_decade)),
        "popularity": [n_genre + n_decade],
    }
    return SideFeatures(matrix=matrix, groups=groups, feature_names=feature_names)


def save_side_features(side: SideFeatures) -> None:
    SIDE_FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        SIDE_FEATURES_FILE,
        matrix=side.matrix,
        feature_names=np.asarray(side.feature_names),
        groups_json=np.asarray([json.dumps(side.groups)]),
    )


def load_side_features() -> SideFeatures:
    packed = np.load(SIDE_FEATURES_FILE, allow_pickle=False)
    groups = json.loads(str(packed["groups_json"][0]))
    return SideFeatures(
        matrix=packed["matrix"].astype(np.float32),
        groups={key: list(value) for key, value in groups.items()},
        feature_names=[str(name) for name in packed["feature_names"].tolist()],
    )
