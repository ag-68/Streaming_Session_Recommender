"""Train SASRec. Stop when the validation score has not improved."""

from __future__ import annotations

import json
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sessrec.baselines import popularity_counts, score_popularity
from sessrec.config import Config
from sessrec.data import load_dataset
from sessrec.features import build_side_features, save_side_features
from sessrec.model import SASRec
from sessrec.paths import CHECKPOINT_FILE, CURVES_FILE, POPULARITY_FILE, SETTINGS_FILE
from sessrec.scoring import evaluate_scores, sasrec_scores


class WindowDataset(Dataset):
    """A random recent window from each viewer's training sequence."""

    def __init__(self, sequences: list[list[int]], max_len: int, seed: int) -> None:
        self.sequences = sequences
        self.max_len = max_len
        self.epoch = 0
        self.seed = seed

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = self.sequences[index]
        generator = random.Random(self.seed + self.epoch * 1_000_003 + index)
        need = self.max_len + 1
        if len(sequence) > need:
            start = generator.randrange(0, len(sequence) - need + 1)
            window = sequence[start : start + need]
        else:
            window = sequence
        inputs = window[:-1]
        targets = window[1:]
        input_row = torch.zeros(self.max_len, dtype=torch.long)
        target_row = torch.zeros(self.max_len, dtype=torch.long)
        input_row[-len(inputs) :] = torch.tensor(inputs, dtype=torch.long)
        target_row[-len(targets) :] = torch.tensor(targets, dtype=torch.long)
        return input_row, target_row


def catalog_loss(
    model: SASRec,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    n_items: int,
    chunk: int = 10,
) -> torch.Tensor:
    """Full-catalog cross-entropy, scored a few positions at a time.

    The total is the same as one loss over every position. Smaller pieces
    keep the catalog-sized score table from using a large block of memory.
    """

    hidden = model.encode(inputs)
    weight = model.item_vectors()
    total = hidden.new_zeros(())
    count = hidden.new_zeros(())
    for start in range(0, hidden.size(1), chunk):
        piece = hidden[:, start : start + chunk]
        target = targets[:, start : start + chunk]
        logits = piece @ weight.T
        total = total + nn.functional.cross_entropy(
            logits.reshape(-1, n_items + 1),
            target.reshape(-1),
            ignore_index=0,
            reduction="sum",
        )
        count = count + (target != 0).sum()
    return total / count.clamp_min(1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(config: Config | None = None) -> dict:
    config = config or Config()
    set_seed(config.seed)
    dataset = load_dataset()
    side = None
    side_dim = 0
    if config.use_side_features:
        side = build_side_features(dataset)
        active = set(config.side_feature_groups)
        matrix = side.masked(active)
        side_dim = int(matrix.shape[1])
        save_side_features(side)
    model = SASRec(
        n_items=dataset.n_items,
        max_len=config.max_len,
        hidden_dim=config.hidden_dim,
        num_blocks=config.num_blocks,
        num_heads=config.num_heads,
        dropout=config.dropout,
        side_dim=side_dim,
    )
    if side is not None:
        model.set_side_features(torch.from_numpy(matrix))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    windows = WindowDataset(dataset.train, config.max_len, config.seed)
    loader = DataLoader(windows, batch_size=config.batch_size, shuffle=True)
    histories = dataset.validation_histories()
    if POPULARITY_FILE.exists():
        popularity = np.load(POPULARITY_FILE)
    else:
        popularity = popularity_counts(dataset.train, dataset.n_items)
    popularity_scores = score_popularity(len(histories), popularity)
    popularity_ndcg = evaluate_scores(
        histories, dataset.val, popularity_scores, config.k, dataset.n_items
    )["ndcg@10"]

    best_ndcg = -1.0
    best_epoch = 0
    stale = 0
    curve: list[dict] = []
    best_state = None
    for epoch in range(1, config.max_epochs + 1):
        windows.set_epoch(epoch)
        model.train()
        total = 0.0
        seen = 0
        for inputs, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = catalog_loss(model, inputs, targets, dataset.n_items)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(inputs)
            seen += len(inputs)
        train_loss = total / max(seen, 1)
        scores = sasrec_scores(model, histories, config.max_len, config.batch_size)
        val_ndcg = evaluate_scores(histories, dataset.val, scores, config.k, dataset.n_items)[
            "ndcg@10"
        ]
        curve.append({"epoch": epoch, "train_loss": train_loss, "val_ndcg@10": val_ndcg})
        print(
            f"epoch {epoch}: loss {train_loss:.4f}  validation NDCG@10 {val_ndcg:.4f}",
            flush=True,
        )
        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            best_epoch = epoch
            stale = 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        else:
            stale += 1
            if stale >= config.patience:
                print(f"stopped after epoch {epoch}; best epoch was {best_epoch}", flush=True)
                break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "config": config.to_dict(),
            "n_items": dataset.n_items,
            "side_dim": side_dim,
            "side_feature_groups": list(config.side_feature_groups) if side_dim else [],
            "best_epoch": best_epoch,
            "best_val_ndcg@10": best_ndcg,
            "popularity_val_ndcg@10": popularity_ndcg,
            "model_version": config.model_version,
        },
        CHECKPOINT_FILE,
    )
    report = {
        "epochs": curve,
        "best_epoch": best_epoch,
        "best_val_ndcg@10": best_ndcg,
        "popularity_val_ndcg@10": popularity_ndcg,
        "stopped_early": stale >= config.patience,
    }
    CURVES_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    SETTINGS_FILE.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    return report


def main() -> None:
    train()


if __name__ == "__main__":
    main()
