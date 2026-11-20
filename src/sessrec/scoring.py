"""Shared helpers for turning histories into model inputs and metrics."""

from __future__ import annotations

import numpy as np
import torch

from sessrec.metrics import mask_banned, mask_seen, summarise
from sessrec.model import SASRec


def left_pad(sequences: list[list[int]], max_len: int) -> np.ndarray:
    """Keep the most recent films and put empty slots on the left."""

    padded = np.zeros((len(sequences), max_len), dtype=np.int64)
    for row, sequence in enumerate(sequences):
        window = sequence[-max_len:]
        if window:
            padded[row, -len(window) :] = window
    return padded


def left_pad_polarities(
    events: list[list[tuple[int, int]]],
    max_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Left-pad signed events into item ids and polarity rows."""

    items = np.zeros((len(events), max_len), dtype=np.int64)
    pols = np.ones((len(events), max_len), dtype=np.int64)
    for row, sequence in enumerate(events):
        window = sequence[-max_len:]
        if not window:
            continue
        item_row = [item for item, _pol in window]
        pol_row = [pol for _item, pol in window]
        items[row, -len(item_row) :] = item_row
        pols[row, -len(pol_row) :] = pol_row
    return items, pols


def sasrec_scores(
    model: SASRec,
    histories: list[list[int]],
    max_len: int,
    batch_size: int = 512,
    polarities: list[list[int]] | None = None,
) -> np.ndarray:
    """Scores for every film, taken from the most recent position."""

    model.eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(histories), batch_size):
            batch = histories[start : start + batch_size]
            sequences = torch.from_numpy(left_pad(batch, max_len))
            pol_tensor = None
            if polarities is not None:
                pol_batch = polarities[start : start + batch_size]
                pol_rows = np.ones((len(pol_batch), max_len), dtype=np.int64)
                for row, pols in enumerate(pol_batch):
                    window = pols[-max_len:]
                    if window:
                        pol_rows[row, -len(window) :] = window
                pol_tensor = torch.from_numpy(pol_rows)
            logits = model(sequences, pol_tensor)[:, -1, :]
            outputs.append(logits.cpu().numpy())
    return np.concatenate(outputs, axis=0)


def sasrec_scores_signed(
    model: SASRec,
    events: list[list[tuple[int, int]]],
    max_len: int,
    batch_size: int = 512,
) -> np.ndarray:
    """Scores from signed (item, polarity) histories."""

    model.eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(events), batch_size):
            batch = events[start : start + batch_size]
            items, pols = left_pad_polarities(batch, max_len)
            logits = model(torch.from_numpy(items), torch.from_numpy(pols))[:, -1, :]
            outputs.append(logits.cpu().numpy())
    return np.concatenate(outputs, axis=0)


def evaluate_scores(
    histories: list[list[int]],
    targets: np.ndarray,
    scores: np.ndarray,
    k: int,
    n_items: int,
    banned: list[list[int]] | None = None,
) -> dict:
    masked = mask_seen(scores, histories)
    if banned is not None:
        masked = mask_banned(masked, banned)
    summary = summarise(masked, targets, k, n_items)
    return {
        "ndcg@10": summary["ndcg@10"],
        "hr@10": summary["hr@10"],
        "coverage@10": summary["coverage@10"],
        "ranks": summary["ranks"],
        "topk": summary["topk"],
    }
