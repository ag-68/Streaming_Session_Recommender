"""Ranking scores for a test with one correct film per viewer."""

import numpy as np


def ndcg_at_k(ranks: np.ndarray, k: int) -> float:
    """Mean of 1/log2(rank+1) when the hidden film is inside the top k, else 0."""

    ranks = np.asarray(ranks)
    scores = np.zeros(len(ranks), dtype=np.float64)
    hit = ranks <= k
    scores[hit] = 1.0 / np.log2(ranks[hit] + 1.0)
    return float(scores.mean()) if len(ranks) else 0.0


def hit_rate_at_k(ranks: np.ndarray, k: int) -> float:
    """Share of viewers whose hidden film appears anywhere in the top k."""

    ranks = np.asarray(ranks)
    if len(ranks) == 0:
        return 0.0
    return float((ranks <= k).mean())


def coverage_at_k(topk: np.ndarray, n_items: int) -> float:
    """Share of the catalog that appears in at least one viewer's top k."""

    if n_items <= 0:
        return 0.0
    used = np.unique(topk)
    used = used[used > 0]
    return float(len(used) / n_items)


def ranks_of_targets(scores: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """1-based rank of each target. A tied film with a smaller id ranks higher."""

    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    target_scores = scores[np.arange(len(targets)), targets]
    ids = np.arange(scores.shape[1], dtype=np.int64)[None, :]
    better = (scores > target_scores[:, None]) | (
        (scores == target_scores[:, None]) & (ids < targets[:, None])
    )
    better[:, 0] = False
    return better.sum(axis=1).astype(np.int64) + 1


def topk_ids(scores: np.ndarray, k: int) -> np.ndarray:
    """Top k item ids. Higher score first. Smaller id wins a tie."""

    scores = np.asarray(scores, dtype=np.float64)
    out = np.zeros((len(scores), k), dtype=np.int64)
    for row_index, row in enumerate(scores):
        finite = np.flatnonzero(np.isfinite(row))
        finite = finite[finite > 0]
        if len(finite) == 0:
            continue
        take = min(k, len(finite))
        if len(finite) > take:
            chosen = finite[np.argpartition(row[finite], -take)[-take:]]
        else:
            chosen = finite
        order = np.lexsort((chosen, -row[chosen]))
        out[row_index, :take] = chosen[order]
    return out


def summarise(scores: np.ndarray, targets: np.ndarray, k: int, n_items: int) -> dict:
    """NDCG@k, hit rate@k, and catalog coverage@k for one set of scores."""

    masked = np.array(scores, dtype=np.float64, copy=True)
    masked[:, 0] = -np.inf
    ranks = ranks_of_targets(masked, targets)
    topk = topk_ids(masked, k)
    return {
        "ndcg@10": ndcg_at_k(ranks, k),
        "hr@10": hit_rate_at_k(ranks, k),
        "coverage@10": coverage_at_k(topk, n_items),
        "ranks": ranks,
        "topk": topk,
    }


def sampled_ranks_of_targets(
    scores: np.ndarray,
    histories: list[list[int]],
    targets: np.ndarray,
    n_items: int,
    n_negatives: int = 100,
    seed: int = 42,
    banned: list[list[int]] | None = None,
) -> np.ndarray:
    """1-based rank of each target among {target} ∪ n_negatives random distractors.

    Negatives are drawn from catalog ids 1..n_items excluding the viewer's history
    and the target. Matches the SASRec paper's sampled ranking protocol.
    """

    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    rng = np.random.default_rng(seed)
    ranks = np.zeros(len(targets), dtype=np.int64)
    catalog = np.arange(1, n_items + 1, dtype=np.int64)
    for row, (history, target) in enumerate(zip(histories, targets, strict=True)):
        forbidden = set(history)
        forbidden.add(0)
        forbidden.add(int(target))
        pool = catalog[~np.isin(catalog, list(forbidden))]
        if len(pool) < n_negatives:
            chosen = pool
        else:
            chosen = rng.choice(pool, size=n_negatives, replace=False)
        candidates = np.concatenate(([int(target)], chosen.astype(np.int64)))
        cand_scores = scores[row, candidates].copy()
        if banned is not None and banned[row]:
            ban = set(banned[row])
            for idx, item in enumerate(candidates):
                if idx == 0:
                    continue
                if int(item) in ban:
                    cand_scores[idx] = -np.inf
        target_score = cand_scores[0]
        better = (cand_scores > target_score) | (
            (cand_scores == target_score) & (candidates < int(target))
        )
        ranks[row] = int(better.sum()) + 1
    return ranks


def summarise_sampled(
    scores: np.ndarray,
    histories: list[list[int]],
    targets: np.ndarray,
    n_items: int,
    k: int = 10,
    n_negatives: int = 100,
    seed: int = 42,
    banned: list[list[int]] | None = None,
) -> dict:
    """Hit@k / NDCG@k under paper-style sampled ranking (no coverage)."""

    ranks = sampled_ranks_of_targets(
        scores,
        histories,
        targets,
        n_items,
        n_negatives=n_negatives,
        seed=seed,
        banned=banned,
    )
    return {
        "ndcg@10": ndcg_at_k(ranks, k),
        "hr@10": hit_rate_at_k(ranks, k),
        "ranks": ranks,
        "n_negatives": n_negatives,
        "protocol": "sampled",
    }


def mask_seen(scores: np.ndarray, histories: list[list[int]]) -> np.ndarray:
    """Remove padding and films the viewer has already been shown."""

    masked = np.array(scores, dtype=np.float64, copy=True)
    masked[:, 0] = -np.inf
    for row, history in enumerate(histories):
        if history:
            masked[row, history] = -np.inf
    return masked


def mask_banned(scores: np.ndarray, banned: list[list[int]]) -> np.ndarray:
    """Remove films this viewer rated low (when they are in the catalog)."""

    masked = np.array(scores, dtype=np.float64, copy=True)
    for row, items in enumerate(banned):
        if items:
            masked[row, items] = -np.inf
    return masked
