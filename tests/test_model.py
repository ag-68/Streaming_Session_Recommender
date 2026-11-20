"""Shape, no peeking, padding, and a sequence rule small enough to learn."""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sessrec.model import SASRec
from sessrec.scoring import left_pad


def _model(**kwargs) -> SASRec:
    defaults = {
        "n_items": 6,
        "max_len": 5,
        "hidden_dim": 8,
        "num_blocks": 1,
        "num_heads": 1,
        "dropout": 0.0,
    }
    defaults.update(kwargs)
    return SASRec(**defaults)


def test_output_covers_every_position_and_film() -> None:
    model = _model()
    sequences = torch.tensor([[0, 0, 1, 2, 3], [0, 1, 2, 3, 4]])
    output = model(sequences)
    assert output.shape == (2, 5, 7)


def test_changing_the_future_does_not_change_the_past() -> None:
    torch.manual_seed(0)
    model = _model(dropout=0.2)
    model.eval()
    original = torch.tensor([[0, 1, 2, 3, 4]])
    changed = original.clone()
    changed[0, 3] = 6
    with torch.inference_mode():
        before = model(original)
        after = model(changed)
    assert torch.allclose(before[:, :3], after[:, :3], atol=1e-6)
    assert not torch.isnan(before).any()


def test_heavy_left_padding_is_finite() -> None:
    torch.manual_seed(0)
    model = _model(dropout=0.2, max_len=8, n_items=4)
    model.train()
    sequences = torch.zeros(4, 8, dtype=torch.long)
    sequences[:, -1] = 1
    output = model(sequences)
    assert not torch.isnan(output).any()


def test_model_learns_that_each_film_is_followed_by_the_next() -> None:
    torch.manual_seed(0)
    sequences = [list(range(start, start + 8)) for start in range(1, 12)]
    model = SASRec(n_items=20, max_len=7, hidden_dim=32, num_blocks=1, num_heads=1, dropout=0.0)
    inputs = []
    targets = []
    for sequence in sequences:
        inputs.append(sequence[:-1])
        targets.append(sequence[1:])
    input_batch = torch.from_numpy(left_pad(inputs, 7))
    target_rows = []
    for target in targets:
        row = np.zeros(7, dtype=np.int64)
        row[-len(target) :] = target
        target_rows.append(row)
    target_batch = torch.tensor(np.stack(target_rows), dtype=torch.long)
    loader = DataLoader(TensorDataset(input_batch, target_batch), batch_size=8, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    model.train()
    for _ in range(40):
        for batch_inputs, batch_targets in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_inputs)
            loss = loss_fn(logits.reshape(-1, 21), batch_targets.reshape(-1))
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.inference_mode():
        prediction = model(input_batch)[:, -1, :].argmax(dim=-1)
    accuracy = float((prediction == target_batch[:, -1]).float().mean())
    assert accuracy > 0.9
