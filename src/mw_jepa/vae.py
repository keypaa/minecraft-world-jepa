import torch
import torch.nn as nn

from mw_jepa.tensor_contracts import assert_frame_tensor, assert_latent_tensor


def encode_frames(vae: nn.Module, frames: torch.Tensor) -> torch.Tensor:
    """Encode [0, 1] RGB frames to canonical world-model latents."""
    assert_frame_tensor(frames, name="frames to encode")
    expects_unit = getattr(vae, "expects_minus_one_to_one", False)
    latent_scale = getattr(vae, "latent_scaling_factor", 1.0)
    vae_input = frames * 2.0 - 1.0 if expects_unit else frames
    latents = vae.encode(vae_input).mode()
    latents = latents * latent_scale
    assert_latent_tensor(latents, name="encoded latents")
    return latents


def decode_latents(vae: nn.Module, latents: torch.Tensor) -> torch.Tensor:
    """Decode canonical world-model latents to [0, 1] RGB frames."""
    assert_latent_tensor(latents, name="latents to decode")
    expects_unit = getattr(vae, "expects_minus_one_to_one", False)
    latent_scale = getattr(vae, "latent_scaling_factor", 1.0)
    decoded = vae.decode(latents / latent_scale)
    frames = (decoded + 1.0) / 2.0 if expects_unit else decoded
    frames = frames.clamp(0.0, 1.0)
    assert_frame_tensor(frames, name="decoded frames")
    return frames


def load_vae(
    model_name: str = "sd-vae",
    device: str = "cuda",
) -> nn.Module:
    """Load the frozen default SD-VAE.

    Only 'sd-vae' (Stable Diffusion VAE — pre-trained, H/8×W/8 latent) is
    supported on the default path. The Oasis conv stub is test-fixture-only
    (see tests); finetuning lives in mw_jepa.vae_experiments.
    """
    if model_name == "sd-vae":
        return _load_sd_vae(device)
    else:
        raise ValueError(f"Unknown VAE: {model_name}. Options: sd-vae")


def _load_oasis_vae(device: str) -> nn.Module:
    """Test-fixture-only stub. Oasis experiments moved out of default path."""
    raise NotImplementedError("oasis stub moved to tests")


def _load_sd_vae(device: str) -> nn.Module:
    """Load Stable Diffusion VAE (pre-trained, 8x spatial downsampling).
    Wraps diffusers API so .encode(x).mode() returns latent and .decode(z) returns tensor.
    """
    from diffusers import AutoencoderKL

    inner = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")

    class SDVAEWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = inner

        def encode(self, x):
            out = self.inner.encode(x)
            # diffusers >=0.38 returns AutoencoderKLOutput; .latent_dist is DiagonalGaussianDistribution
            return out.latent_dist if hasattr(out, "latent_dist") else out

        def decode(self, z):
            return self.inner.decode(z, return_dict=False)[0]

        def forward(self, x):
            return self.decode(self.encode(x).mode())

    m = SDVAEWrapper()
    m.expects_minus_one_to_one = True
    m.latent_scaling_factor = getattr(m.inner.config, "scaling_factor", 0.18215)
    m.inner.eval()
    return m.to(device)
