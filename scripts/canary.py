"""End-to-end pre-flight canary (local single-GPU).

Proves the full spine before committing GPU hours:
  1. Frame stream -> SD-VAE encode/decode -> reconstruction grid
  2. Action token distribution (no-op ratio, camera vs keyboard)
  3. World model forward + backward pass
  4. Checkpoint save -> load -> verify round-trip

Usage: python scripts/canary.py --config configs/stage1_4ctx.yaml
"""
import argparse
import gc
from pathlib import Path

CKPT_DIR = Path("checkpoints/canary")


def main():
    import torch
    from torch.utils.data import DataLoader
    from torchvision.utils import make_grid, save_image
    from datasets import load_dataset

    from mw_jepa.vae import load_vae, encode_frames, decode_latents
    from mw_jepa.data import MinecraftFrameStream, WorldModelStream
    from mw_jepa.world_model import MineWorldModel
    from mw_jepa.trainer import Trainer, collate_stream
    from mw_jepa.action_tokenizer import parse_lumine_action, KEYBOARD_TOKENS
    from mw_jepa.config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage1_4ctx.yaml")
    args = ap.parse_args()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    errors = []

    # ── Phase 1: VAE reconstruction ──
    print("=" * 60)
    print("PHASE 1: Base SD-VAE Reconstruction")
    print("=" * 60)

    vae = load_vae(device="cuda")
    stream = MinecraftFrameStream(shard_start=0, shard_end=1, target_size=256)
    loader = DataLoader(stream, batch_size=4, num_workers=0, drop_last=True)

    loader_it = iter(loader)
    try:
        orig_batch = next(loader_it).cuda()
    finally:
        # Deterministically release the suspended infinite stream (open HF
        # download). Relying on interpreter teardown hangs: abandoned download
        # threads keep the process alive after CANARY PASSED.
        del loader_it
        del loader
        gc.collect()
    with torch.no_grad():
        z = encode_frames(vae, orig_batch)
        recon = decode_latents(vae, z)

    mse = torch.nn.functional.mse_loss(recon, orig_batch).item()
    psnr = 20 * torch.log10(1.0 / torch.sqrt(torch.tensor(mse + 1e-8))).item()
    print(f"  Latent shape: {tuple(z.shape)}")
    print(f"  Recon MSE:    {mse:.6f}  (lower is better)")
    print(f"  Recon PSNR:   {psnr:.2f} dB  (higher is better)")
    if mse > 0.05:
        errors.append(f"VAE recon MSE too high ({mse:.6f}) — check normalization")

    # Quick latent stat sanity (per-channel on one batch)
    latent_mean = z.mean(dim=(0, 2, 3))
    latent_std = z.std(dim=(0, 2, 3))
    print(f"  Latent mean (per ch): {latent_mean.tolist()}")
    print(f"  Latent std  (per ch): {latent_std.tolist()}")
    if (latent_std < 1e-4).any():
        errors.append(f"Near-zero latent std ({latent_std.tolist()}) — collapsed latent space")
    if latent_mean.abs().max() > 10:
        errors.append(f"Large latent mean magnitude ({latent_mean.tolist()}) — check scaling")

    error_map = (orig_batch - recon).abs()
    comparison = torch.cat([orig_batch.cpu(), recon.cpu(), error_map.cpu()], dim=0)
    grid = make_grid(comparison, nrow=4)
    save_image(grid, str(CKPT_DIR / "canary_reconstruction.png"))
    print(f"  Reconstruction grid → {CKPT_DIR}/canary_reconstruction.png")
    print("  Rows: original | reconstructed | per-pixel absolute error")
    print()

    # ── Phase 2: Action token distribution ──
    print("=" * 60)
    print("PHASE 2: Action Token Distribution (2000 samples, shard 0)")
    print("=" * 60)

    token_names = {v: k for k, v in KEYBOARD_TOKENS.items()}
    counts = {t: 0 for t in range(29)}
    keyboard_pressed = 0
    camera_only = 0
    noop = 0

    ds = load_dataset(
        "TESS-Computer/minecraft-vla-stage1",
        split="train",
        streaming=True,
        data_files="data/shard_00000.parquet",
    )
    ds_iter = iter(ds)
    try:
        for i, ex in enumerate(ds_iter):
            if i >= 2000:
                break
            token = parse_lumine_action(ex["action"])
            counts[token] = counts.get(token, 0) + 1
            if token <= 21:
                keyboard_pressed += 1
            elif token <= 27:
                camera_only += 1
            else:
                noop += 1
    finally:
        # Same as Phase 1: breaking out early abandons a live HF download.
        # GeneratorExit at the suspended yield lets datasets/fsspec release
        # the connection instead of hanging interpreter exit.
        close = getattr(ds_iter, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        del ds_iter
        del ds
        gc.collect()

    total = keyboard_pressed + camera_only + noop
    print(f"  Keyboard (0-21):     {keyboard_pressed:5d}  ({100*keyboard_pressed/total:.1f}%)")
    for tid in sorted(counts):
        if counts[tid] == 0:
            continue
        if tid <= 21:
            label = token_names.get(tid, f"keyboard_{tid}")
        elif tid <= 27:
            label = f"camera_yaw_{tid}"
        else:
            label = "center/no-op"
        pct = 100 * counts[tid] / total
        bar = "█" * max(1, int(pct // 2))
        print(f"    [{tid:2d}] {label:20s} {counts[tid]:5d} ({pct:5.1f}%) {bar}")

    print(f"  Camera only (22-27): {camera_only:5d}  ({100*camera_only/total:.1f}%)")
    print(f"  No-op / center (28): {noop:5d}  ({100*noop/total:.1f}%)")

    if noop / total > 0.8:
        errors.append(f"No-op ratio is {noop/total:.0%} — model may learn a screensaver")
    if keyboard_pressed == 0:
        errors.append("Zero keyboard actions detected — action parser may be broken")
    print()

    # ── Phase 3: World model forward pass ──
    print("=" * 60)
    print("PHASE 3: World Model Forward Pass")
    print("=" * 60)

    cfg = load_config(args.config)
    model = MineWorldModel(**cfg["model"]).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    wm_stream = WorldModelStream(
        sequence_length=4,
        shard_start=0,
        shard_end=1,
        target_size=256,
    )
    wm_loader = DataLoader(
        wm_stream, batch_size=2,
        collate_fn=collate_stream,
        num_workers=0, drop_last=True,
    )
    wm_loader_it = iter(wm_loader)
    try:
        batch = next(wm_loader_it)
    finally:
        del wm_loader_it
        del wm_loader
        gc.collect()

    has_latents = "latents" in batch
    if not has_latents:
        frames = batch["frames"].cuda()
        B, T = frames.shape[:2]
        with torch.no_grad():
            flat = frames.view(B * T, 3, 256, 256)
            latents = encode_frames(vae, flat).view(B, T, 4, 32, 32)
    else:
        latents = batch["latents"].cuda()
    actions = batch["actions"].cuda()

    pred = model(latents[:, :-1], actions[:, :-1])
    loss = torch.nn.functional.mse_loss(pred, latents[:, -1])
    print(f"  Input latents:  {tuple(latents.shape)}")
    print(f"  Predicted next: {tuple(pred.shape)}")
    print(f"  MSE loss:       {loss.item():.6f}")
    if torch.isnan(loss) or torch.isinf(loss):
        errors.append("WM forward loss is NaN/Inf — check model or data")
    print()

    # ── Phase 4: Checkpoint round-trip ──
    print("=" * 60)
    print("PHASE 4: Checkpoint Round-Trip")
    print("=" * 60)

    trainer = Trainer(model=model, vae=vae, config=cfg, ckpt_dir=CKPT_DIR)
    trainer.save_checkpoint(loss.item())
    print(f"  Checkpoint saved → {CKPT_DIR}/latest.pt")

    model2 = MineWorldModel(**cfg["model"]).cuda()
    trainer2 = Trainer(model=model2, vae=vae, config=cfg, ckpt_dir=CKPT_DIR)
    trainer2.load_checkpoint(str(CKPT_DIR / "latest.pt"))

    with torch.no_grad():
        pred2 = trainer2.model(latents[:, :-1], actions[:, :-1])
        loss2 = torch.nn.functional.mse_loss(pred2, latents[:, -1])

    match = abs(loss.item() - loss2.item()) < 1e-5
    print(f"  Pre-save loss:     {loss.item():.8f}")
    print(f"  Post-load loss:    {loss2.item():.8f}")
    print(f"  Round-trip match:  {'PASS' if match else 'FAIL'}")
    if not match:
        errors.append("Checkpoint round-trip mismatch")
    print()

    # ── Summary ──
    print("=" * 60)
    if errors:
        print(f"CANARY FAILED — {len(errors)} issue(s):")
        for e in errors:
            print(f"  ✗  {e}")
        raise RuntimeError("\n".join(errors))
    else:
        print("CANARY PASSED — all checks OK")
        print("  [OK]  VAE reconstruction plausible")
        print("  [OK]  Action distribution non-trivial")
        print("  [OK]  World model forward pass + loss finite")
        print("  [OK]  Checkpoint round-trip verified")
    print("=" * 60)

    # ── Teardown: free GPU allocations deterministically ──
    # The canary ends with ~8GB live (2x 338M models + VAE + latents).
    # Releasing before return avoids CUDA-teardown hangs on exit.
    try:
        del model, model2, vae, trainer, trainer2
    except NameError:
        pass
    try:
        del latents, pred, pred2
    except NameError:
        pass
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
