import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """Load a pre-trained VAE and prepare for fine-tuning on Minecraft frames.

    Supports: 'sd-vae' (Stable Diffusion VAE — default, pre-trained, H/8×W/8 latent),
              'oasis-vae' (Etched/Oasis ViT-VAE — conv stub for testing).
    """
    if model_name == "sd-vae":
        return _load_sd_vae(device)
    elif model_name == "oasis-vae":
        return _load_oasis_vae(device)
    else:
        raise ValueError(f"Unknown VAE: {model_name}. Options: sd-vae, oasis-vae")


def _load_oasis_vae(device: str) -> nn.Module:
    """Conv-based VAE stub, trainable from scratch."""
    class OasisVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 128, 4, 2, 1), nn.SiLU(),
                nn.Conv2d(128, 256, 4, 2, 1), nn.SiLU(),
                nn.Conv2d(256, 512, 4, 2, 1), nn.SiLU(),
                nn.Conv2d(512, 4, 3, 1, 1),
            )
            self.decoder = nn.Sequential(
                nn.Conv2d(4, 512, 3, 1, 1), nn.SiLU(),
                nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.SiLU(),
                nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.SiLU(),
                nn.ConvTranspose2d(128, 3, 4, 2, 1),
            )

        def encode(self, x):
            return LatentDistribution(self.encoder(x))

        def decode(self, z):
            return self.decoder(z).sigmoid()

        def forward(self, x):
            return self.decode(self.encode(x).mode())

    class LatentDistribution:
        def __init__(self, z):
            self.z = z
        def sample(self):
            return self.z
        def mode(self):
            return self.z

    m = OasisVAE().to(device)
    m.expects_minus_one_to_one = False
    m.latent_scaling_factor = 1.0
    return m


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


class Discriminator(nn.Module):
    """PatchGAN discriminator for adversarial VAE fine-tuning."""

    def __init__(self, in_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


def finetune_vae(
    vae: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_steps: int = 50000,
    lr: float = 1e-5,
    log_callback=None,
) -> nn.Module:
    """Fine-tune a pre-trained VAE on Minecraft frames with
    MSE + LPIPS + adversarial loss.
    """
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex").to("cuda")
    except ImportError:
        lpips_fn = None

    discriminator = Discriminator().to("cuda")
    opt_g = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=0.01)
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=lr * 2, betas=(0.5, 0.999))

    vae.train()
    data_iter = iter(dataloader)
    total_steps = 0

    while total_steps < num_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = batch.cuda()

        # --- Generator Step ---
        z = encode_frames(vae, batch)
        recon = decode_latents(vae, z)

        mse_loss = F.mse_loss(recon, batch)

        p_loss = torch.tensor(0.0, device=batch.device)
        if lpips_fn is not None:
            p_loss = lpips_fn(recon * 2.0 - 1.0, batch * 2.0 - 1.0).mean()

        fake_logits = discriminator(recon)
        adv_loss = torch.mean((fake_logits - 1) ** 2)

        g_loss = mse_loss + 0.5 * p_loss + 0.1 * adv_loss

        opt_g.zero_grad(set_to_none=True)
        g_loss.backward()
        opt_g.step()

        # --- Discriminator Step ---
        real_logits = discriminator(batch.detach())
        fake_logits = discriminator(recon.detach())
        d_loss = 0.5 * (
            torch.mean((real_logits - 1) ** 2) + torch.mean(fake_logits ** 2)
        )

        opt_d.zero_grad(set_to_none=True)
        d_loss.backward()
        opt_d.step()

        total_steps += 1

        if total_steps % 100 == 0 and log_callback:
            log_callback({
                "step": total_steps,
                "g_loss": g_loss.item(),
                "d_loss": d_loss.item(),
                "mse": mse_loss.item(),
                "perceptual": p_loss.item() if isinstance(p_loss, torch.Tensor) else 0,
            })

    return vae


def evaluate_vae(
    vae: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_frames: int = 1000,
    device: str = "cuda",
) -> dict:
    """Evaluate VAE reconstruction quality: PSNR, SSIM, LPIPS, FID."""
    vae.eval()

    # --- Perceptual loss ---
    lpips_fn = None
    try:
        import lpips as _lpips
        lpips_fn = _lpips.LPIPS(net="alex").to(device)
    except ImportError:
        pass

    # --- InceptionV3 for FID features ---
    inception = None
    try:
        from torchvision.models import inception_v3, Inception_V3_Weights
        class _FIDFeat(nn.Module):
            def __init__(self):
                super().__init__()
                inner = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
                inner.aux_logits = False
                self.layers = nn.Sequential(*list(inner.children())[:-1])
            def forward(self, x):
                x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
                x = (x - 0.5) / 0.5
                return self.layers(x).flatten(1)
        inception = _FIDFeat().to(device).eval()
    except ImportError:
        pass

    psnr_vals, ssim_vals, lpips_vals = [], [], []
    real_feats, recon_feats = [], []
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            if count >= num_frames:
                break
            n = batch.shape[0]
            batch = batch.to(device)

            recon = decode_latents(vae, encode_frames(vae, batch))

            # PSNR
            mse = F.mse_loss(recon, batch, reduction="none").view(n, -1).mean(dim=1)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
            psnr_vals.append(psnr)

            # SSIM (simplified single-scale)
            c1, c2 = (0.01 * 1) ** 2, (0.03 * 1) ** 2
            mu_x = F.avg_pool2d(recon, 11, 1, 5)
            mu_y = F.avg_pool2d(batch, 11, 1, 5)
            sigma2_x = F.avg_pool2d(recon ** 2, 11, 1, 5) - mu_x ** 2
            sigma2_y = F.avg_pool2d(batch ** 2, 11, 1, 5) - mu_y ** 2
            sigma_xy = F.avg_pool2d(recon * batch, 11, 1, 5) - mu_x * mu_y
            ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
                       ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma2_x + sigma2_y + c2))
            ssim_vals.append(ssim_map.view(n, -1).mean(dim=1))

            # LPIPS
            if lpips_fn is not None:
                lp_vals = lpips_fn(recon * 2.0 - 1.0, batch * 2.0 - 1.0).view(-1)
                lpips_vals.append(lp_vals.cpu())

            # Inception features for FID
            if inception is not None:
                real_feats.append(inception(batch).cpu())
                recon_feats.append(inception(recon).cpu())

            count += n

    results = {
        "psnr_mean": torch.cat(psnr_vals).mean().item(),
        "psnr_std": torch.cat(psnr_vals).std().item(),
        "ssim_mean": torch.cat(ssim_vals).mean().item(),
        "ssim_std": torch.cat(ssim_vals).std().item(),
    }

    if lpips_vals:
        results["lpips_mean"] = torch.cat(lpips_vals).mean().item()

    if inception is not None and len(real_feats) > 1:
        import numpy as _np
        real_f = torch.cat(real_feats, dim=0).numpy()
        recon_f = torch.cat(recon_feats, dim=0).numpy()
        try:
            from scipy import linalg
            mu_r, sigma_r = real_f.mean(0), _np.cov(real_f, rowvar=False)
            mu_g, sigma_g = recon_f.mean(0), _np.cov(recon_f, rowvar=False)
            diff = mu_r - mu_g
            covmean, _ = linalg.sqrtm(sigma_r @ sigma_g, disp=False)
            if _np.iscomplexobj(covmean):
                covmean = covmean.real
            results["fid"] = float(diff @ diff + _np.trace(sigma_r + sigma_g - 2 * covmean))
        except Exception:
            results["fid"] = None

    return results
