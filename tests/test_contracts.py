# tests/test_contracts.py
import torch
import pytest
from mw_jepa.tensor_contracts import assert_frame_tensor, assert_latent_tensor, assert_action_tensor

def test_frame_ok():
    assert_frame_tensor(torch.zeros(2, 3, 256, 256), size=256, name="t")

def test_frame_rejects_bad_channels():
    with pytest.raises(ValueError):
        assert_frame_tensor(torch.zeros(2, 4, 256, 256), name="t")

def test_latent_ok():
    assert_latent_tensor(torch.zeros(2, 5, 4, 32, 32), grid_size=32, name="t")

def test_action_ok():
    assert_action_tensor(torch.zeros(2, 4, dtype=torch.long), sequence_length=4, name="t")
