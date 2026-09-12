"""Resume-semantics tests (CPU-only).

Covers the exact-resume machinery without touching CUDA or data:
- advance_iterator fast-forward logic (skip math primitive)
- epoch-end checkpoint stores epoch+1 / epoch_start_step == step, so a
  resume re-enters a FRESH epoch (audit Critical #1 regression test)
"""

import torch
from torch import nn

from mw_jepa.trainer import Trainer, advance_iterator


def test_advance_exact():
    it = iter(range(10))
    consumed, exhausted = advance_iterator(it, 4)
    assert (consumed, exhausted) == (4, False)
    assert next(it) == 4  # stream continues exactly where it left off


def test_advance_exhaustion():
    it = iter(range(3))
    consumed, exhausted = advance_iterator(it, 5)
    assert (consumed, exhausted) == (3, True)


def test_advance_zero():
    it = iter(range(3))
    assert advance_iterator(it, 0) == (0, False)
    assert next(it) == 0


def _make_trainer(tmp_path):
    model = nn.Linear(4, 4)
    return Trainer(
        model=model,
        vae=None,
        config={"training": {}, "data": {}, "model": {}},
        ckpt_dir=tmp_path,
    )


def test_epoch_end_checkpoint_advances(tmp_path):
    """Simulates the train() epoch-end sequence: stored epoch must be E+1
    with epoch_start_step == step, so resume enters a fresh epoch."""
    t = _make_trainer(tmp_path)
    t.epoch = 0
    t.epoch_start_step = 0
    t.step = 100
    t.epoch = 0 + 1
    t.epoch_start_step = t.step
    t.save_checkpoint(0.5, epoch_label=0)

    ckpt = torch.load(tmp_path / "latest.pt", map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 1
    assert ckpt["epoch_start_step"] == 100
    assert ckpt["step"] == 100
    assert (tmp_path / "epoch_0000_loss_0.500000.pt").exists()


def test_mid_epoch_checkpoint_keeps_position(tmp_path):
    """save_latest mid-epoch must preserve the in-epoch position."""
    t = _make_trainer(tmp_path)
    t.epoch = 2
    t.epoch_start_step = 1000
    t.step = 1250
    t.save_latest(0.42)

    ckpt = torch.load(tmp_path / "latest.pt", map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 2
    assert ckpt["epoch_start_step"] == 1000
    assert ckpt["step"] == 1250
    # skip math the resume path will perform:
    assert ckpt["step"] - ckpt["epoch_start_step"] == 250
