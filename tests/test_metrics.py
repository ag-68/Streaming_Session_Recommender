"""Hand-worked ranking scores."""

import numpy as np

from sessrec.metrics import (
    coverage_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    ranks_of_targets,
    summarise,
)


def test_hit_at_rank_1_scores_one() -> None:
    assert ndcg_at_k(np.array([1]), 10) == 1.0


def test_hit_at_rank_2_uses_log() -> None:
    assert ndcg_at_k(np.array([2]), 10) == np.log2(3) ** -1


def test_rank_11_scores_zero() -> None:
    assert ndcg_at_k(np.array([11]), 10) == 0.0
    assert hit_rate_at_k(np.array([11]), 10) == 0.0
    assert hit_rate_at_k(np.array([10, 11]), 10) == 0.5


def test_smaller_id_wins_a_tie() -> None:
    scores = np.array([[-np.inf, 1.0, 1.0, 0.0]])
    assert int(ranks_of_targets(scores, np.array([1]))[0]) == 1
    assert int(ranks_of_targets(scores, np.array([2]))[0]) == 2


def test_coverage_on_a_tiny_catalog() -> None:
    topk = np.array([[1, 2], [2, 3]])
    assert coverage_at_k(topk, n_items=4) == 0.75


def test_seen_film_does_not_take_the_top_slot() -> None:
    scores = np.array([[-np.inf, 9.0, 1.0, 0.5]])
    masked = scores.copy()
    masked[0, 1] = -np.inf
    summary = summarise(masked, np.array([2]), k=1, n_items=3)
    assert int(summary["topk"][0, 0]) == 2
    assert summary["ndcg@10"] == 1.0


def test_sampled_rank_among_fixed_pile() -> None:
    from sessrec.metrics import sampled_ranks_of_targets

    # Catalog size 5; history empty; target=1 has highest score among all.
    scores = np.array([[-np.inf, 5.0, 4.0, 3.0, 2.0, 1.0]])
    ranks = sampled_ranks_of_targets(
        scores,
        histories=[[]],
        targets=np.array([1]),
        n_items=5,
        n_negatives=2,
        seed=0,
    )
    assert int(ranks[0]) == 1
