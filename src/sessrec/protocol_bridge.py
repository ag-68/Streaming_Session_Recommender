"""Protocol bridge: cross-eval between paper sampled@100 and our full-catalog.

Phase 1 offline only — writes artifacts/metrics_protocol_bridge.json.
Does not touch the docs site.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sessrec.config import Config
from sessrec.data import load_dataset, read_ratings
from sessrec.metrics import summarise_sampled
from sessrec.model import SASRec
from sessrec.paths import (
    CHECKPOINT_ID_ONLY_FILE,
    CHECKPOINT_PAPER_STYLE_FILE,
    CHECKPOINT_RATING_WAY3_FILE,
    CURVES_PAPER_STYLE_FILE,
    METRICS_PROTOCOL_BRIDGE_FILE,
)
from sessrec.scoring import evaluate_scores, sasrec_scores, sasrec_scores_signed


def _public_sampled(summary: dict) -> dict:
    return {
        "ndcg@10": summary["ndcg@10"],
        "hr@10": summary["hr@10"],
        "n_negatives": summary.get("n_negatives", 100),
        "protocol": "sampled",
    }


def _public_full(summary: dict) -> dict:
    return {
        "ndcg@10": summary["ndcg@10"],
        "hr@10": summary["hr@10"],
        "coverage@10": summary.get("coverage@10"),
        "protocol": "full_catalog",
    }


def _load_id_only(config: Config) -> tuple[SASRec, dict]:
    payload = torch.load(CHECKPOINT_ID_ONLY_FILE, map_location="cpu", weights_only=False)
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


def _load_signed(config: Config) -> tuple[SASRec, dict]:
    payload = torch.load(CHECKPOINT_RATING_WAY3_FILE, map_location="cpu", weights_only=False)
    cfg = payload.get("config", {})
    model = SASRec(
        n_items=int(payload["n_items"]),
        max_len=int(cfg.get("max_len", config.max_len)),
        hidden_dim=int(cfg.get("hidden_dim", config.hidden_dim)),
        num_blocks=int(cfg.get("num_blocks", config.num_blocks)),
        num_heads=int(cfg.get("num_heads", config.num_heads)),
        dropout=0.0,
        side_dim=0,
        use_polarity=True,
    )
    model.load_state_dict(payload["state_dict"], strict=False)
    model.eval()
    return model, payload


def evaluate_sampled_scores(
    histories: list[list[int]],
    targets: np.ndarray,
    scores: np.ndarray,
    n_items: int,
    k: int = 10,
    n_negatives: int = 100,
    seed: int = 42,
    banned: list[list[int]] | None = None,
) -> dict:
    """Paper-style Hit/NDCG among target + n_negatives random distractors."""

    # Seen items are not candidates in the paper protocol either; zero them
    # so a leaked seen score cannot win inside the sampled pile.
    masked = np.array(scores, dtype=np.float64, copy=True)
    masked[:, 0] = -np.inf
    for row, history in enumerate(histories):
        if history:
            masked[row, history] = -np.inf
    return summarise_sampled(
        masked,
        histories,
        targets,
        n_items,
        k=k,
        n_negatives=n_negatives,
        seed=seed,
        banned=banned,
    )


def run_ours_under_paper_eval(
    config: Config | None = None,
    n_negatives: int = 100,
) -> dict:
    """Hardening 2: Id SASRec / Veto / Signed Trail under sampled@100."""

    config = config or Config(use_side_features=False)
    dataset = load_dataset()
    histories = dataset.test_histories()
    targets = dataset.test
    seed = config.seed

    id_model, id_payload = _load_id_only(config)
    id_scores = sasrec_scores(id_model, histories, config.max_len, config.batch_size)

    id_only = evaluate_sampled_scores(
        histories, targets, id_scores, dataset.n_items, config.k, n_negatives, seed
    )
    veto = evaluate_sampled_scores(
        histories,
        targets,
        id_scores,
        dataset.n_items,
        config.k,
        n_negatives,
        seed,
        banned=dataset.dislikes_soft,
    )

    signed_model, signed_payload = _load_signed(config)
    signed_scores = sasrec_scores_signed(
        signed_model, dataset.signed_before_test, config.max_len, config.batch_size
    )
    signed_trail = evaluate_sampled_scores(
        histories,
        targets,
        signed_scores,
        dataset.n_items,
        config.k,
        n_negatives,
        seed,
        banned=dataset.dislikes_soft,
    )

    report = {
        "experiment": "ours_under_paper_sampled",
        "note": (
            "Existing Id / Veto / Signed Trail checkpoints scored with paper "
            f"sampled ranking (true + {n_negatives} random). Same leave-one-out "
            "likes-only splits as our full-catalog runs."
        ),
        "n_negatives": n_negatives,
        "seed": seed,
        "n_users": len(targets),
        "n_items": dataset.n_items,
        "models": {
            "id_sasrec": {
                **_public_sampled(id_only),
                "best_epoch": int(id_payload.get("best_epoch", -1)),
            },
            "veto": {
                **_public_sampled(veto),
                "best_epoch": int(id_payload.get("best_epoch", -1)),
                "ban": "soft_dislike_leq_3",
            },
            "signed_trail": {
                **_public_sampled(signed_trail),
                "best_epoch": int(signed_payload.get("best_epoch", -1)),
                "ban": "soft_dislike_leq_3",
            },
        },
    }
    return report


@dataclass
class PaperStyleDataset:
    train: list[list[int]]
    val: np.ndarray
    test: np.ndarray
    n_items: int
    stats: dict

    def test_histories(self) -> list[list[int]]:
        return [train + [int(val)] for train, val in zip(self.train, self.val, strict=True)]

    def validation_histories(self) -> list[list[int]]:
        return [list(train) for train in self.train]


def prepare_paper_style_dataset(
    config: Config | None = None,
    min_interactions: int = 5,
) -> PaperStyleDataset:
    """All 1–5★ ratings as sequence events (paper-style positives)."""

    config = config or Config()
    ratings = read_ratings()
    ratings = ratings.sort_values(["user_id", "timestamp", "_row"], kind="mergesort")
    per_user = ratings.groupby("user_id").size()
    kept_users = per_user[per_user >= min_interactions].index
    ratings = ratings.loc[ratings["user_id"].isin(kept_users)]

    raw_ids = np.sort(ratings["movie_id"].unique())
    raw_to_index = {int(raw): index + 1 for index, raw in enumerate(raw_ids)}
    ratings = ratings.copy()
    ratings["item"] = ratings["movie_id"].map(raw_to_index).astype(np.int64)

    train: list[list[int]] = []
    val: list[int] = []
    test: list[int] = []
    for _user_id, group in ratings.groupby("user_id", sort=False):
        items = group["item"].tolist()
        if len(items) < 3:
            continue
        train.append(items[:-2])
        val.append(int(items[-2]))
        test.append(int(items[-1]))

    stats = {
        "protocol": "paper_style_all_ratings",
        "min_interactions": min_interactions,
        "users": len(train),
        "items": len(raw_to_index),
        "raw_ratings_used": int(len(ratings)),
    }
    return PaperStyleDataset(
        train=train,
        val=np.asarray(val, dtype=np.int64),
        test=np.asarray(test, dtype=np.int64),
        n_items=len(raw_to_index),
        stats=stats,
    )


class PaperWindowDataset(Dataset):
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


def _bce_one_neg_loss(
    model: SASRec,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    n_items: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Paper-style binary CE: push true next up, one random wrong down."""

    hidden = model.encode(inputs)
    weight = model.item_vectors()
    active = targets != 0
    if not active.any():
        return hidden.new_zeros(())

    pos = targets[active]
    h = hidden[active]
    pos_logits = (h * weight[pos]).sum(dim=-1)

    # Uniform negative in 1..n_items excluding the positive id.
    draw = torch.randint(
        0,
        n_items - 1,
        (pos.numel(),),
        device=inputs.device,
        generator=generator,
    )
    neg = draw + 1
    neg = torch.where(neg >= pos, neg + 1, neg)

    neg_logits = (h * weight[neg]).sum(dim=-1)
    loss_pos = nn.functional.binary_cross_entropy_with_logits(
        pos_logits, torch.ones_like(pos_logits), reduction="mean"
    )
    loss_neg = nn.functional.binary_cross_entropy_with_logits(
        neg_logits, torch.zeros_like(neg_logits), reduction="mean"
    )
    return loss_pos + loss_neg


def _full_catalog_ndcg(
    model: SASRec,
    histories: list[list[int]],
    targets: np.ndarray,
    max_len: int,
    batch_size: int,
    n_items: int,
    k: int,
) -> float:
    scores = sasrec_scores(model, histories, max_len, batch_size)
    summary = evaluate_scores(histories, targets, scores, k, n_items)
    return float(summary["ndcg@10"])


def run_paper_style_under_ours_eval(config: Config | None = None) -> dict:
    """Hardening 1: paper-style train (all ratings + 1-neg BCE), full-catalog eval."""

    config = config or Config(use_side_features=False)
    dataset = prepare_paper_style_dataset(config)
    print(json.dumps(dataset.stats, indent=2), flush=True)

    set_seed = _set_seed
    set_seed(config.seed)
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
    windows = PaperWindowDataset(dataset.train, config.max_len, config.seed)
    loader = DataLoader(windows, batch_size=config.batch_size, shuffle=True)

    best_ndcg = -1.0
    best_epoch = 0
    stale = 0
    best_state = None
    curve: list[dict] = []
    generator = torch.Generator()
    generator.manual_seed(config.seed)

    for epoch in range(1, config.max_epochs + 1):
        windows.set_epoch(epoch)
        model.train()
        total = 0.0
        seen = 0
        for inputs, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = _bce_one_neg_loss(model, inputs, targets, dataset.n_items, generator)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(inputs)
            seen += len(inputs)
        train_loss = total / max(seen, 1)
        model.eval()
        val_ndcg = _full_catalog_ndcg(
            model,
            dataset.validation_histories(),
            dataset.val,
            config.max_len,
            config.batch_size,
            dataset.n_items,
            config.k,
        )
        curve.append({"epoch": epoch, "train_loss": train_loss, "val_ndcg@10": val_ndcg})
        print(
            f"paper-style epoch {epoch}: loss {train_loss:.4f}  "
            f"validation full-catalog NDCG@10 {val_ndcg:.4f}",
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
                print(
                    f"paper-style stopped after epoch {epoch}; best epoch was {best_epoch}",
                    flush=True,
                )
                break

    if best_state is None:
        raise RuntimeError("paper-style training produced no checkpoint")
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
            "model_version": "sasrec-ml1m-paper-style",
            "data_stats": dataset.stats,
        },
        CHECKPOINT_PAPER_STYLE_FILE,
    )
    CURVES_PAPER_STYLE_FILE.write_text(
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

    test_histories = dataset.test_histories()
    test_scores = sasrec_scores(model, test_histories, config.max_len, config.batch_size)
    full = evaluate_scores(
        test_histories, dataset.test, test_scores, config.k, dataset.n_items
    )
    sampled = evaluate_sampled_scores(
        test_histories,
        dataset.test,
        test_scores,
        dataset.n_items,
        config.k,
        n_negatives=100,
        seed=config.seed,
    )
    report = {
        "experiment": "paper_style_under_ours_full_catalog",
        "note": (
            "Approximate paper recipe in this codebase: all 1–5★ sequence events, "
            "1-neg BCE training. Primary eval is our full-catalog Hit/NDCG@10; "
            "sampled@100 also reported for sanity. Not an official Table III reproduction."
        ),
        "data_stats": dataset.stats,
        "best_epoch": best_epoch,
        "best_val_ndcg@10": best_ndcg,
        "full_catalog": _public_full(full),
        "sampled_100": _public_sampled(sampled),
    }
    return report


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def merge_and_save(ours_sampled: dict | None, paper_full: dict | None) -> dict:
    report: dict = {
        "note": (
            "Protocol bridge. No docs/web updates. "
            "ours_under_paper_sampled = our three flavors on paper exam; "
            "paper_style_under_ours_full_catalog = paper-like train on our exam."
        )
    }
    if METRICS_PROTOCOL_BRIDGE_FILE.exists():
        try:
            report = json.loads(METRICS_PROTOCOL_BRIDGE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    if ours_sampled is not None:
        report["ours_under_paper_sampled"] = ours_sampled
    if paper_full is not None:
        report["paper_style_under_ours_full_catalog"] = paper_full
    METRICS_PROTOCOL_BRIDGE_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Protocol bridge experiments (offline)")
    parser.add_argument(
        "step",
        choices=["ours_sampled", "paper_full", "all"],
        help="ours_sampled = Phase 1a; paper_full = Phase 1b; all = both",
    )
    args = parser.parse_args()
    config = Config(use_side_features=False)

    ours = None
    paper = None
    if args.step in ("ours_sampled", "all"):
        print("=== Phase 1a: our methods under paper sampled@100 ===", flush=True)
        ours = run_ours_under_paper_eval(config)
        print(json.dumps(ours, indent=2), flush=True)
        merge_and_save(ours, None)
    if args.step in ("paper_full", "all"):
        print("=== Phase 1b: paper-style train, full-catalog eval ===", flush=True)
        paper = run_paper_style_under_ours_eval(config)
        print(json.dumps(paper, indent=2), flush=True)
        merge_and_save(None, paper)

    final = json.loads(METRICS_PROTOCOL_BRIDGE_FILE.read_text(encoding="utf-8"))
    print("\n=== Bridge summary ===", flush=True)
    if "ours_under_paper_sampled" in final:
        models = final["ours_under_paper_sampled"]["models"]
        print("Ours → paper sampled@100", flush=True)
        for name, row in models.items():
            print(
                f"  {name:<14} NDCG@10 {row['ndcg@10']:.4f}  HR@10 {row['hr@10']:.4f}",
                flush=True,
            )
    if "paper_style_under_ours_full_catalog" in final:
        block = final["paper_style_under_ours_full_catalog"]
        full = block["full_catalog"]
        samp = block["sampled_100"]
        print("Paper-style → our full catalog", flush=True)
        print(
            f"  full_catalog   NDCG@10 {full['ndcg@10']:.4f}  HR@10 {full['hr@10']:.4f}",
            flush=True,
        )
        print(
            f"  sampled_100    NDCG@10 {samp['ndcg@10']:.4f}  HR@10 {samp['hr@10']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
