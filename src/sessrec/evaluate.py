"""Score the hidden test film once, with the same rules for every method."""

from __future__ import annotations

import json

import numpy as np
import torch

from sessrec.baselines import load_fitted, score_item_knn, score_popularity
from sessrec.config import Config
from sessrec.data import load_dataset
from sessrec.features import load_side_features
from sessrec.metrics import hit_rate_at_k, ndcg_at_k
from sessrec.model import SASRec
from sessrec.paths import (
    CHECKPOINT_FILE,
    DEMO_USERS_FILE,
    HISTORY_LENGTH_FILE,
    METRICS_FILE,
    METRICS_ID_ONLY_FILE,
    SLICES_FILE,
)
from sessrec.scoring import evaluate_scores, sasrec_scores


def load_model(config: Config | None = None) -> tuple[SASRec, dict]:
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
        dropout=float(cfg.get("dropout", config.dropout)),
        side_dim=side_dim,
    )
    if side_dim > 0:
        side = load_side_features()
        groups = payload.get("side_feature_groups") or list(side.groups)
        model.set_side_features(torch.from_numpy(side.masked(set(groups))))
    model.load_state_dict(payload["state_dict"], strict=False)
    # side_features buffer may be in state_dict; re-apply active mask after load
    if side_dim > 0:
        side = load_side_features()
        groups = payload.get("side_feature_groups") or list(side.groups)
        model.set_side_features(torch.from_numpy(side.masked(set(groups))))
    model.eval()
    return model, payload


def _public(summary: dict) -> dict:
    return {key: summary[key] for key in ("ndcg@10", "hr@10", "coverage@10")}


def _bucket_report(ranks: np.ndarray, mask: np.ndarray, k: int) -> dict | None:
    chosen = ranks[mask]
    if len(chosen) == 0:
        return None
    return {
        "n": int(mask.sum()),
        "ndcg@10": ndcg_at_k(chosen, k),
        "hr@10": hit_rate_at_k(chosen, k),
    }


def evaluate(config: Config | None = None) -> dict:
    config = config or Config()
    dataset = load_dataset()
    model, payload = load_model(config)
    popularity, neighbors, sims = load_fitted()
    histories = dataset.test_histories()
    targets = dataset.test
    popularity_scores = score_popularity(len(histories), popularity)
    knn_scores = score_item_knn(histories, neighbors, sims, dataset.n_items, popularity)
    sasrec = sasrec_scores(model, histories, config.max_len, config.batch_size)
    scored = {
        "popularity": evaluate_scores(
            histories, targets, popularity_scores, config.k, dataset.n_items
        ),
        "item_knn": evaluate_scores(histories, targets, knn_scores, config.k, dataset.n_items),
        "sasrec": evaluate_scores(histories, targets, sasrec, config.k, dataset.n_items),
    }
    report = {
        "split": "test",
        "k": config.k,
        "pass_rule": "SASRec NDCG@10 is higher than popularity",
        "pass_rule_met": scored["sasrec"]["ndcg@10"] > scored["popularity"]["ndcg@10"],
        "models": {name: _public(summary) for name, summary in scored.items()},
        "best_epoch": int(payload["best_epoch"]),
        "side_dim": int(payload.get("side_dim", 0)),
        "side_feature_groups": list(payload.get("side_feature_groups") or []),
        "model_version": str(payload.get("model_version", config.model_version)),
    }
    if METRICS_ID_ONLY_FILE.exists():
        id_only = json.loads(METRICS_ID_ONLY_FILE.read_text(encoding="utf-8"))
        report["id_only_baseline"] = {
            "models": id_only.get("models", {}),
            "best_epoch": id_only.get("best_epoch"),
            "note": "SASRec trained with movie ids only; lower bound for side-feature runs",
        }
    METRICS_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lengths = [5, 10, 20, 50, 100]
    by_length = {"lengths": lengths, "models": {name: [] for name in scored}}
    for length in lengths:
        short = [history[-length:] for history in histories]
        by_length["models"]["popularity"].append(
            _public(
                evaluate_scores(
                    short,
                    targets,
                    score_popularity(len(short), popularity),
                    config.k,
                    dataset.n_items,
                )
            )["ndcg@10"]
        )
        by_length["models"]["item_knn"].append(
            _public(
                evaluate_scores(
                    short,
                    targets,
                    score_item_knn(short, neighbors, sims, dataset.n_items, popularity),
                    config.k,
                    dataset.n_items,
                )
            )["ndcg@10"]
        )
        by_length["models"]["sasrec"].append(
            _public(
                evaluate_scores(
                    short,
                    targets,
                    sasrec_scores(model, short, config.max_len, config.batch_size),
                    config.k,
                    dataset.n_items,
                )
            )["ndcg@10"]
        )
    HISTORY_LENGTH_FILE.write_text(json.dumps(by_length, indent=2), encoding="utf-8")

    input_lengths = np.asarray([len(history) for history in histories])
    history_bins = [
        ("under 20", input_lengths < 20),
        ("20 to 49", (input_lengths >= 20) & (input_lengths < 50)),
        ("50 to 99", (input_lengths >= 50) & (input_lengths < 100)),
        ("100 or more", input_lengths >= 100),
    ]
    train_counts = popularity[1:]
    low, high = np.quantile(train_counts, [1 / 3, 2 / 3])
    target_counts = popularity[targets]
    popularity_bins = [
        ("rarely liked", target_counts <= low),
        ("middling", (target_counts > low) & (target_counts <= high)),
        ("widely liked", target_counts > high),
    ]
    slices = {
        "history_buckets": [
            {
                "name": name,
                "models": {
                    model_name: _bucket_report(scored[model_name]["ranks"], mask, config.k)
                    for model_name in scored
                },
            }
            for name, mask in history_bins
        ],
        "popularity_buckets": [
            {
                "name": name,
                "models": {
                    model_name: _bucket_report(scored[model_name]["ranks"], mask, config.k)
                    for model_name in scored
                },
            }
            for name, mask in popularity_bins
        ],
    }
    SLICES_FILE.write_text(json.dumps(slices, indent=2), encoding="utf-8")
    _write_demo_users(dataset, scored, config)
    print(json.dumps(report, indent=2))
    return report


def _write_demo_users(dataset, scored: dict, config: Config) -> None:
    generator = np.random.default_rng(config.seed)
    picked = np.sort(generator.choice(len(dataset.train), size=12, replace=False))
    users = []
    for index in picked.tolist():
        history = dataset.test_histories()[index]
        shown = history[-8:]
        heldout = int(dataset.test[index])
        record = {
            "user_index": int(index),
            "history_titles": [dataset.index_to_title[item] for item in shown],
            "history_raw_ids": [dataset.index_to_raw[item] for item in shown],
            "heldout_title": dataset.index_to_title[heldout],
            "heldout_raw_id": dataset.index_to_raw[heldout],
            "hit_at_10": {},
        }
        for name in ("sasrec", "item_knn", "popularity"):
            top = scored[name]["topk"][index]
            record[name] = [
                {
                    "rank": rank,
                    "movie_id": dataset.index_to_raw[int(item)],
                    "title": dataset.index_to_title[int(item)],
                }
                for rank, item in enumerate(top[:5], start=1)
                if item > 0
            ]
            record["hit_at_10"][name] = bool(scored[name]["ranks"][index] <= 10)
        users.append(record)
    DEMO_USERS_FILE.write_text(json.dumps({"users": users}, indent=2), encoding="utf-8")


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
