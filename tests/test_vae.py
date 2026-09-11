# tests/test_vae.py
import torch
from mw_jepa.vae import encode_frames, decode_latents

class FakeDist:
    def __init__(self, z): self.z = z
    def mode(self): return self.z

class FakeVAE(torch.nn.Module):
    expects_minus_one_to_one = False
    latent_scaling_factor = 1.0
    def encode(self, x): return FakeDist(torch.zeros(x.shape[0], 4, 32, 32))
    def decode(self, z): return torch.zeros(z.shape[0], 3, 256, 256)

def test_roundtrip_shapes():
    vae = FakeVAE()
    f = torch.rand(2, 3, 256, 256)
    z = encode_frames(vae, f)
    assert z.shape == (2, 4, 32, 32)
    r = decode_latents(vae, z)
    assert r.shape == (2, 3, 256, 256)
