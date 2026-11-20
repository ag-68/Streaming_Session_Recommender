"""Settings for the comparison. Names match the build spec."""

from dataclasses import asdict, dataclass


@dataclass
class Config:
    """Starting settings. Extra training runs, if any, are recorded beside these."""

    max_len: int = 100
    hidden_dim: int = 64
    num_blocks: int = 2
    num_heads: int = 1
    dropout: float = 0.2
    lr: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 100
    patience: int = 10
    item_knn_neighbors: int = 50
    seed: int = 42
    min_rating: int = 4
    min_positives: int = 5
    k: int = 10
    model_version: str = "sasrec-ml1m-side-seed42"
    use_side_features: bool = True
    side_feature_groups: tuple[str, ...] = ("genre", "decade", "popularity")
    dislike_loss_weight: float = 0.1
    hard_dislike_max: int = 2
    soft_dislike_max: int = 3

    def to_dict(self) -> dict:
        return asdict(self)
