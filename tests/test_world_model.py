# tests/test_world_model.py
import torch
from mw_jepa.world_model import MineWorldModel

def test_tiny_forward_finite():
    m = MineWorldModel(latent_grid_size=16, embed_dim=64, num_blocks=2, num_heads=4, num_actions=29, action_embed_dim=32)
    lat = torch.randn(1, 3, 4, 16, 16)
    act = torch.zeros(1, 3, dtype=torch.long)
    out = m(lat, act)
    assert out.shape == (1, 4, 16, 16)
    assert torch.isfinite(out).all()
