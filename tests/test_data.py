# tests/test_data.py
import torch
from mw_jepa.data import WorldModelStream


def test_yield_sequences_shapes():
    s = WorldModelStream(sequence_length=4, shard_start=0, shard_end=0)
    frames = [torch.zeros(3, 256, 256) for _ in range(10)]
    acts = list(range(10))
    out = list(s._yield_sequences(frames, acts, item_kind="frames"))
    assert len(out) > 0
    assert out[0]["frames"].shape == (5, 3, 256, 256)
    assert out[0]["actions"].shape == (5,)


def test_short_trajectory_yields_nothing():
    s = WorldModelStream(sequence_length=4, shard_start=0, shard_end=0)
    out = list(s._yield_sequences([torch.zeros(3, 256, 256)] * 3, [28] * 3, item_kind="frames"))
    assert out == []
