# Optional experiments. Not imported by default train path. Requires: pip install -e ".[experiments]"
import torch
import torch.nn as nn
import torch.nn.functional as F

from mw_jepa.vae import decode_latents, encode_frames


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
