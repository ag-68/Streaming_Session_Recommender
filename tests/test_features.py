"""Side-feature matrix shape and optional fusion into SASRec."""

import numpy as np
import torch

from sessrec.data import load_dataset
from sessrec.features import build_side_features
from sessrec.model import SASRec


def test_side_features_cover_catalog_and_padding() -> None:
    dataset = load_dataset()
    side = build_side_features(dataset)
    assert side.matrix.shape[0] == dataset.n_items + 1
    assert side.matrix.shape[1] == side.side_dim
    assert np.allclose(side.matrix[0], 0.0)
    assert set(side.groups) == {"genre", "decade", "popularity"}
    genre_only = side.masked({"genre"})
    assert genre_only[:, side.groups["popularity"]].sum() == 0
    assert genre_only[:, side.groups["genre"]].sum() > 0


def test_side_aware_model_scores_catalog() -> None:
    dataset = load_dataset()
    side = build_side_features(dataset)
    model = SASRec(
        n_items=dataset.n_items,
        max_len=8,
        hidden_dim=16,
        num_blocks=1,
        num_heads=1,
        dropout=0.0,
        side_dim=side.side_dim,
    )
    model.set_side_features(torch.from_numpy(side.matrix))
    sequences = torch.zeros(2, 8, dtype=torch.long)
    sequences[0, -3:] = torch.tensor([1, 2, 3])
    sequences[1, -2:] = torch.tensor([4, 5])
    output = model(sequences)
    assert output.shape == (2, 8, dataset.n_items + 1)
    assert not torch.isnan(output).any()
