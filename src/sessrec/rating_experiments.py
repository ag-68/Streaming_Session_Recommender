"""Offline experiments for three ways to use per-user star ratings.

Does not update the docs site. Writes metrics under artifacts/metrics_rating_*.json.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sessrec.baselines import load_fitted, score_item_knn, score_popularity
from sessrec.config import Config
from sessrec.data import load_dataset
from sessrec.model import SASRec
from sessrec.paths import (
    CHECKPOINT_ID_ONLY_FILE,
    CHECKPOINT_RATING_WAY2_FILE,
    CHECKPOINT_RATING_WAY3_FILE,
    CURVES_RATING_WAY2_FILE,
    CURVES_RATING_WAY3_FILE,
    METRICS_ID_ONLY_FILE,
    METRICS_RATING_COMPARE_FILE,
    METRICS_RATING_WAY1_FILE,
    METRICS_RATING_WAY2_FILE,
    METRICS_RATING_WAY3_FILE,
)
from sessrec.scoring import evaluate_scores, sasrec_scores, sasrec_scores_signed


def _public(summary: dict) -> dict:
    return {key: summary[key] for key in ("ndcg@10", "hr@10", "coverage@10")}


def _load_id_only_model(config: Config) -> tuple[SASRec, dict]:
    path = CHECKPOINT_ID_ONLY_FILE
    if not path.exists():
        raise FileNotFoundError(f"missing id-only checkpoint at {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = payload.get("config", {})
    model = SASRec(
        n_items=int(payload["n_items"]),
        max_len=int(cfg.get("max_len", config.max_len)),
        hidden_dim=int(cfg.get("hidden_dim", config.hidden_dim)),
        num_blocks=int(cfg.get("num_blocks", config.num_blocks)),
        num_heads=int(cfg.get("num_heads", config.num_heads)),
        dropout=0.0,
        side_dim=0,
        use_polarity=False,
    )
    model.load_state_dict(payload["state_dict"], strict=False)
    model.eval()
    return model, payload


def _score_all_methods(
    dataset,
    model: SASRec,
    config: Config,
    banned: list[list[int]] | None,
    *,
    signed_events: list[list[tuple[int, int]]] | None = None,
) -> dict:
    popularity, neighbors, sims = load_fitted()
    histories = dataset.test_histories()
    targets = dataset.test
    if signed_events is None:
        sasrec = sasrec_scores(model, histories, config.max_len, config.batch_size)
    else:
        sasrec = sasrec_scores_signed(model, signed_events, config.max_len, config.batch_size)
    return {
        "popularity": evaluate_scores(
            histories,
            targets,
            score_popularity(len(histories), popularity),
            config.k,
            dataset.n_items,
            banned=banned,
        ),
        "item_knn": evaluate_scores(
            histories,
            targets,
            score_item_knn(histories, neighbors, sims, dataset.n_items, popularity),
            config.k,
            dataset.n_items,
            banned=banned,
        ),
        "sasrec": evaluate_scores(
            histories,
            targets,
            sasrec,
            config.k,
            dataset.n_items,
            banned=banned,
        ),
    }


def run_way1(config: Config | None = None) -> dict:
    """Eval-only: ban ≤3★ films at ranking time on the id-only checkpoint."""

    config = config or Config()
    dataset = load_dataset()
    model, payload = _load_id_only_model(config)
    baseline = _score_all_methods(dataset, model, config, banned=None)
    banned = dataset.dislikes_soft
    with_ban = _score_all_methods(dataset, model, config, banned=banned)
    id_only = {}
    if METRICS_ID_ONLY_FILE.exists():
        id_only = json.loads(METRICS_ID_ONLY_FILE.read_text(encoding="utf-8")).get("models", {})
    report = {
        "method": "way1_hard_ban",
        "note": "Id-only SASRec; soft dislikes (rating<=3) masked at eval for all methods",
        "best_epoch": int(payload.get("best_epoch", -1)),
        "without_ban": {name: _public(summary) for name, summary in baseline.items()},
        "with_ban": {name: _public(summary) for name, summary in with_ban.items()},
        "id_only_file_baseline": id_only,
    }
    METRICS_RATING_WAY1_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


class LikeWindowDataset(Dataset):
    def __init__(
        self,
        sequences: list[list[int]],
        dislikes_hard: list[list[int]],
        max_len: int,
        seed: int,
    ) -> None:
        self.sequences = sequences
        self.dislikes_hard = dislikes_hard
        self.max_len = max_len
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int):
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
        hates = self.dislikes_hard[index]
        hate_row = torch.zeros(32, dtype=torch.long)
        if hates:
            take = hates[:32]
            hate_row[: len(take)] = torch.tensor(take, dtype=torch.long)
        return input_row, target_row, hate_row


def _catalog_loss_with_dislikes_v2(
    model: SASRec,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    hates: torch.Tensor,
    n_items: int,
    dislike_weight: float,
    chunk: int = 10,
) -> torch.Tensor:
    """CE on next-like + softplus on hate item scores at supervised positions."""

    hidden = model.encode(inputs)
    weight = model.item_vectors()
    total = hidden.new_zeros(())
    count = hidden.new_zeros(())
    hate_pen = hidden.new_zeros(())
    hate_n = hidden.new_zeros(())
    hate_valid = hates > 0
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
        if dislike_weight <= 0 or not hate_valid.any():
            continue
        # Average softplus(score[hate]) over active positions and valid hates.
        # logits: (B, C, N+1), hates: (B, H)
        active = (target != 0).unsqueeze(-1)  # (B, C, 1)
        gathered = torch.gather(
            logits,
            2,
            hates.clamp(min=0).unsqueeze(1).expand(-1, logits.size(1), -1),
        )  # (B, C, H)
        mask = active & hate_valid.unsqueeze(1)
        if mask.any():
            hate_pen = hate_pen + nn.functional.softplus(gathered)[mask].sum()
            hate_n = hate_n + mask.sum()
    ce = total / count.clamp_min(1)
    if hate_n.item() == 0 or dislike_weight <= 0:
        return ce
    return ce + dislike_weight * (hate_pen / hate_n)


def run_way2(config: Config | None = None) -> dict:
    """Train id-only SASRec with dislike push-down; eval with soft ban."""

    config = config or Config()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    dataset = load_dataset()
    model = SASRec(
        n_items=dataset.n_items,
        max_len=config.max_len,
        hidden_dim=config.hidden_dim,
        num_blocks=config.num_blocks,
        num_heads=config.num_heads,
        dropout=config.dropout,
        side_dim=0,
        use_polarity=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    windows = LikeWindowDataset(
        dataset.train, dataset.dislikes_hard, config.max_len, config.seed
    )
    loader = DataLoader(windows, batch_size=config.batch_size, shuffle=True)
    histories = dataset.validation_histories()

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
        for inputs, targets, hates in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = _catalog_loss_with_dislikes_v2(
                model,
                inputs,
                targets,
                hates,
                dataset.n_items,
                config.dislike_loss_weight,
            )
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(inputs)
            seen += len(inputs)
        train_loss = total / max(seen, 1)
        model.eval()
        scores = sasrec_scores(model, histories, config.max_len, config.batch_size)
        val_ndcg = evaluate_scores(
            histories, dataset.val, scores, config.k, dataset.n_items
        )["ndcg@10"]
        curve.append({"epoch": epoch, "train_loss": train_loss, "val_ndcg@10": val_ndcg})
        print(
            f"way2 epoch {epoch}: loss {train_loss:.4f}  validation NDCG@10 {val_ndcg:.4f}",
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
                print(f"way2 stopped after epoch {epoch}; best epoch was {best_epoch}", flush=True)
                break

    if best_state is None:
        raise RuntimeError("way2 training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    torch.save(
        {
            "state_dict": best_state,
            "config": config.to_dict(),
            "n_items": dataset.n_items,
            "side_dim": 0,
            "use_polarity": False,
            "best_epoch": best_epoch,
            "best_val_ndcg@10": best_ndcg,
            "model_version": "sasrec-ml1m-rating-way2",
        },
        CHECKPOINT_RATING_WAY2_FILE,
    )
    CURVES_RATING_WAY2_FILE.write_text(
        json.dumps(
            {
                "epochs": curve,
                "best_epoch": best_epoch,
                "best_val_ndcg@10": best_ndcg,
                "stopped_early": stale >= config.patience,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    without = _score_all_methods(dataset, model, config, banned=None)
    with_ban = _score_all_methods(dataset, model, config, banned=dataset.dislikes_soft)
    report = {
        "method": "way2_dislike_loss",
        "note": "Id-only history; hard dislikes (rating<=2) in loss; soft ban optional at eval",
        "dislike_loss_weight": config.dislike_loss_weight,
        "best_epoch": best_epoch,
        "best_val_ndcg@10": best_ndcg,
        "without_ban": {name: _public(summary) for name, summary in without.items()},
        "with_ban": {name: _public(summary) for name, summary in with_ban.items()},
    }
    METRICS_RATING_WAY2_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


class SignedWindowDataset(Dataset):
    """Windows over signed events; supervise only next-like targets."""

    def __init__(
        self,
        signed_train: list[list[tuple[int, int]]],
        max_len: int,
        seed: int,
    ) -> None:
        self.signed_train = signed_train
        self.max_len = max_len
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.signed_train)

    def __getitem__(self, index: int):
        events = self.signed_train[index]
        generator = random.Random(self.seed + self.epoch * 1_000_003 + index)
        need = self.max_len + 1
        if len(events) > need:
            # Prefer windows that contain at least one next-like target
            starts = []
            for start in range(0, len(events) - need + 1):
                window = events[start : start + need]
                if any(pol == 1 for _item, pol in window[1:]):
                    starts.append(start)
            start = generator.choice(starts) if starts else generator.randrange(0, len(events) - need + 1)
            window = events[start : start + need]
        else:
            window = events
        inputs = window[:-1]
        targets_events = window[1:]
        input_items = torch.zeros(self.max_len, dtype=torch.long)
        input_pols = torch.zeros(self.max_len, dtype=torch.long)
        target_items = torch.zeros(self.max_len, dtype=torch.long)
        if inputs:
            items = [item for item, _pol in inputs]
            pols = [pol for _item, pol in inputs]
            input_items[-len(items) :] = torch.tensor(items, dtype=torch.long)
            input_pols[-len(pols) :] = torch.tensor(pols, dtype=torch.long)
        if targets_events:
            # Only like events are supervised; dislike next-steps get target 0 (ignore)
            t_items = []
            for item, pol in targets_events:
                t_items.append(item if pol == 1 else 0)
            target_items[-len(t_items) :] = torch.tensor(t_items, dtype=torch.long)
        return input_items, input_pols, target_items


def _signed_catalog_loss(
    model: SASRec,
    inputs: torch.Tensor,
    polarities: torch.Tensor,
    targets: torch.Tensor,
    n_items: int,
    chunk: int = 10,
) -> torch.Tensor:
    hidden = model.encode(inputs, polarities)
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


def run_way3(config: Config | None = None) -> dict:
    """Train signed-history SASRec; eval with soft ban."""

    config = config or Config()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    dataset = load_dataset()
    model = SASRec(
        n_items=dataset.n_items,
        max_len=config.max_len,
        hidden_dim=config.hidden_dim,
        num_blocks=config.num_blocks,
        num_heads=config.num_heads,
        dropout=config.dropout,
        side_dim=0,
        use_polarity=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    windows = SignedWindowDataset(dataset.signed_train, config.max_len, config.seed)
    loader = DataLoader(windows, batch_size=config.batch_size, shuffle=True)

    # Validation: signed context before val like ≈ signed_train; target = val
    val_events = dataset.signed_train
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
        for inputs, pols, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = _signed_catalog_loss(model, inputs, pols, targets, dataset.n_items)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(inputs)
            seen += len(inputs)
        train_loss = total / max(seen, 1)
        model.eval()
        scores = sasrec_scores_signed(model, val_events, config.max_len, config.batch_size)
        val_ndcg = evaluate_scores(
            dataset.validation_histories(),
            dataset.val,
            scores,
            config.k,
            dataset.n_items,
        )["ndcg@10"]
        curve.append({"epoch": epoch, "train_loss": train_loss, "val_ndcg@10": val_ndcg})
        print(
            f"way3 epoch {epoch}: loss {train_loss:.4f}  validation NDCG@10 {val_ndcg:.4f}",
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
                print(f"way3 stopped after epoch {epoch}; best epoch was {best_epoch}", flush=True)
                break

    if best_state is None:
        raise RuntimeError("way3 training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    torch.save(
        {
            "state_dict": best_state,
            "config": config.to_dict(),
            "n_items": dataset.n_items,
            "side_dim": 0,
            "use_polarity": True,
            "best_epoch": best_epoch,
            "best_val_ndcg@10": best_ndcg,
            "model_version": "sasrec-ml1m-rating-way3",
        },
        CHECKPOINT_RATING_WAY3_FILE,
    )
    CURVES_RATING_WAY3_FILE.write_text(
        json.dumps(
            {
                "epochs": curve,
                "best_epoch": best_epoch,
                "best_val_ndcg@10": best_ndcg,
                "stopped_early": stale >= config.patience,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    without = _score_all_methods(
        dataset, model, config, banned=None, signed_events=dataset.signed_before_test
    )
    with_ban = _score_all_methods(
        dataset,
        model,
        config,
        banned=dataset.dislikes_soft,
        signed_events=dataset.signed_before_test,
    )
    report = {
        "method": "way3_signed_history",
        "note": "Signed like/dislike steps; predict next like; soft ban at eval",
        "best_epoch": best_epoch,
        "best_val_ndcg@10": best_ndcg,
        "without_ban": {name: _public(summary) for name, summary in without.items()},
        "with_ban": {name: _public(summary) for name, summary in with_ban.items()},
    }
    METRICS_RATING_WAY3_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def write_compare() -> dict:
    rows = []
    if METRICS_ID_ONLY_FILE.exists():
        base = json.loads(METRICS_ID_ONLY_FILE.read_text(encoding="utf-8"))
        sas = base["models"]["sasrec"]
        rows.append(
            {
                "run": "id_only_baseline",
                "ndcg@10": sas["ndcg@10"],
                "hr@10": sas["hr@10"],
                "coverage@10": sas["coverage@10"],
                "notes": "movie ids only; no rating ban",
            }
        )
    if METRICS_RATING_WAY1_FILE.exists():
        way1 = json.loads(METRICS_RATING_WAY1_FILE.read_text(encoding="utf-8"))
        sas = way1["with_ban"]["sasrec"]
        rows.append(
            {
                "run": "way1_ban_only",
                "ndcg@10": sas["ndcg@10"],
                "hr@10": sas["hr@10"],
                "coverage@10": sas["coverage@10"],
                "notes": "id-only checkpoint; soft dislike ban at eval",
            }
        )
    if METRICS_RATING_WAY2_FILE.exists():
        way2 = json.loads(METRICS_RATING_WAY2_FILE.read_text(encoding="utf-8"))
        sas = way2["with_ban"]["sasrec"]
        rows.append(
            {
                "run": "way2_plus_ban",
                "ndcg@10": sas["ndcg@10"],
                "hr@10": sas["hr@10"],
                "coverage@10": sas["coverage@10"],
                "notes": "dislike loss retrain; soft ban at eval",
            }
        )
    if METRICS_RATING_WAY3_FILE.exists():
        way3 = json.loads(METRICS_RATING_WAY3_FILE.read_text(encoding="utf-8"))
        sas = way3["with_ban"]["sasrec"]
        rows.append(
            {
                "run": "way3_plus_ban",
                "ndcg@10": sas["ndcg@10"],
                "hr@10": sas["hr@10"],
                "coverage@10": sas["coverage@10"],
                "notes": "signed history retrain; soft ban at eval",
            }
        )
    report = {"runs": rows}
    METRICS_RATING_COMPARE_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nrun                      NDCG@10   HR@10    notes")
    print("-" * 90)
    for row in rows:
        print(
            f"{row['run']:<24} {row['ndcg@10']:.4f}   {row['hr@10']:.4f}   {row['notes']}"
        )
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "step",
        choices=["data", "way1", "way2", "way3", "compare", "all"],
        help="Which experiment step to run",
    )
    args = parser.parse_args()
    config = Config(use_side_features=False)
    if args.step in ("data", "all"):
        from sessrec.data import build_from_files

        stats = build_from_files(config).stats
        print(json.dumps(stats, indent=2))
    if args.step in ("way1", "all"):
        run_way1(config)
    if args.step in ("way2", "all"):
        run_way2(config)
    if args.step in ("way3", "all"):
        run_way3(config)
    if args.step in ("compare", "all"):
        write_compare()


if __name__ == "__main__":
    main()
