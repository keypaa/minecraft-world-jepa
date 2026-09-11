"""Local step-time + memory probe (local single-GPU).

Merges the former measure_step_time.py (synthetic fwd/bwd timing, SDPA
dispatch check, VAE encode timing, stage estimates) and
measure_real_step.py (real-data streaming step timing) probes.

Usage: python scripts/bench.py --config configs/stage1_4ctx.yaml [--real-data] [--steps 6]
"""
import argparse
import time

import torch

from mw_jepa.config import load_config

SEQ_COUNT_EST = 249_000  # 10 shards x 50K frames/shard, stride=2 sliding window


def check_sdpa_dispatch():
    """SDPA dispatch check via torch.backends.cuda.sdp_kernel."""
    print("--- SDPA Backend Check ---")
    print(f"  Flash sdp:          {torch.backends.cuda.flash_sdp_enabled()}")
    print(f"  Mem-efficient sdp:  {torch.backends.cuda.mem_efficient_sdp_enabled()}")
    print(f"  Math sdp:           {torch.backends.cuda.math_sdp_enabled()}")
    # Verify efficient backend works by forcing memory-efficient kernel on BF16 data
    try:
        with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=False, enable_mem_efficient=True):
            q = torch.randn(1, 1, 128, 8, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(1, 1, 128, 8, device="cuda", dtype=torch.bfloat16)
            v = torch.randn(1, 1, 128, 8, device="cuda", dtype=torch.bfloat16)
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            print(f"  mem-efficient dispatch (BF16): OK (output shape {tuple(out.shape)})")
    except RuntimeError as e:
        print(f"  mem-efficient dispatch (BF16): FAILED — {e}")
    try:
        with torch.backends.cuda.sdp_kernel(enable_math=False, enable_mem_efficient=False, enable_flash=True):
            q = torch.randn(1, 1, 128, 8, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(1, 1, 128, 8, device="cuda", dtype=torch.bfloat16)
            v = torch.randn(1, 1, 128, 8, device="cuda", dtype=torch.bfloat16)
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            print(f"  flash dispatch (BF16):        OK (output shape {tuple(out.shape)})")
    except RuntimeError as e:
        print(f"  flash dispatch (BF16):        FAILED — {e}")
    print()


def bench_synthetic(config, num_steps=6):
    from mw_jepa.world_model import MineWorldModel

    model_cfg = config["model"]
    B = config.get("batch_size", 32)
    T = config.get("context", 4)

    print(f"Batch size: {B}, Context: {T}")
    print(f"Tokens per step: {B} * {T} * 256 = {B * T * 256}")

    model = MineWorldModel(**model_cfg).cuda().train()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {param_count:,}")
    print(f"Model memory (BF16): {param_count * 2 / 1e9:.2f} GB")

    latent_dim = model_cfg["latent_dim"]
    grid_size = model_cfg["latent_grid_size"]
    num_actions = model_cfg["num_actions"]

    dummy_latents = torch.randn(B, T, latent_dim, grid_size, grid_size, device="cuda")
    dummy_actions = torch.randint(0, num_actions, (B, T), device="cuda")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)

    print(f"\n--- Forward + Backward Timing (batch={B}, ctx={T}) ---")
    times = []
    for step in range(num_steps):
        torch.cuda.synchronize()
        t0 = time.time()

        pred = model(dummy_latents, dummy_actions)
        target = dummy_latents[:, -1, :, :, :]
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        t = time.time() - t0
        times.append(t)

        mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"  Step {step}: {t:.3f}s | peak GPU mem: {mem:.2f} GiB")
        torch.cuda.reset_peak_memory_stats()

    warmup = times[0]
    steady = times[1:]
    avg_steady = sum(steady) / len(steady)
    print(f"\n  Step 1 (warmup):      {warmup:.3f}s")
    print(f"  Steps 2-{num_steps}:   {[f'{t:.3f}s' for t in steady]}")
    print(f"  Avg steady-state:     {avg_steady:.3f}s")
    print(f"  Range:                {min(steady):.3f}s - {max(steady):.3f}s")

    # Clean peak GPU memory measurement
    torch.cuda.reset_peak_memory_stats()
    pred = model(dummy_latents, dummy_actions)
    target = dummy_latents[:, -1, :, :, :]
    loss = torch.nn.functional.mse_loss(pred, target)
    loss.backward()
    torch.cuda.synchronize()
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"\nPeak GPU memory (fwd+bwd+optimizer state): {peak_mem:.2f} GiB")

    # Stage 1 estimate
    steps_per_epoch = SEQ_COUNT_EST // B
    epochs = config.get("epochs", 2)
    print(f"\n--- Stage 1 Estimate ({epochs} epochs, {T}-ctx) ---")
    print(f"Sequences (10 shards): {SEQ_COUNT_EST:,}")
    print(f"Steps/epoch:           {steps_per_epoch:,}")
    total_steady_h = steps_per_epoch * epochs * avg_steady / 3600
    print(f"Transformer only:      {total_steady_h:.1f}h")

    # VAE encode timing
    print(f"\n--- VAE Encode Timing (batch={B}) ---")
    from mw_jepa.vae import load_vae, encode_frames
    vae = load_vae(device="cuda").eval()
    vae_param_count = sum(p.numel() for p in vae.parameters())

    dummy_frames = torch.rand(B, 3, config["data"]["target_size"], config["data"]["target_size"], device="cuda")

    with torch.no_grad():
        _ = encode_frames(vae, dummy_frames)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        _ = encode_frames(vae, dummy_frames)
    torch.cuda.synchronize()
    vae_time = time.time() - t0
    vae_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"VAE params:            {vae_param_count:,}")
    print(f"VAE encode time:       {vae_time:.4f}s per batch of {B}")
    print(f"VAE peak GPU mem:      {vae_mem:.2f} GiB")

    total_step = avg_steady + vae_time
    total_h = steps_per_epoch * epochs * total_step / 3600
    print(f"\n--- With On-the-Fly VAE Encode ---")
    print(f"Combined step:         {avg_steady:.3f}s (transformer) + {vae_time:.3f}s (VAE) = {total_step:.3f}s")
    print(f"Total time:            {total_h:.1f}h")
    print(f"VAE share of step:     {vae_time / total_step * 100:.0f}%")
    print(f"\nCombined GPU memory (both models loaded): ~{peak_mem + vae_mem:.1f} GiB")
    print(f"\n--- Precompute Benefit ---")
    print(f"Effective speedup: 1 / (1 - {vae_time / total_step:.3f}) = {1 / (1 - vae_time / total_step):.1f}x")
    return avg_steady


def bench_real_data(config, target_steps=10):
    from mw_jepa.vae import load_vae, encode_frames
    from mw_jepa.world_model import MineWorldModel
    from mw_jepa.data import WorldModelStream
    from torch.utils.data import DataLoader
    from mw_jepa.trainer import collate_stream

    model_cfg = config["model"]
    data_cfg = config["data"]
    B = config.get("batch_size", 32)
    T = config.get("context", 4)

    print(f"Batch size: {B}, Context: {T}")

    print("\nLoading VAE...")
    vae = load_vae(device="cuda").eval()
    print(f"  VAE loaded ({sum(p.numel() for p in vae.parameters()):,} params)")

    print("Loading world model...")
    model = MineWorldModel(**model_cfg).cuda().train()
    print(f"  Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)

    print(f"\nStreaming from shard 0 (target_size={data_cfg['target_size']})...")
    stream = WorldModelStream(
        sequence_length=T,
        shard_start=0,
        shard_end=1,
        target_size=data_cfg["target_size"],
    )
    loader = iter(DataLoader(
        stream,
        batch_size=B,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_stream,
        drop_last=True,
    ))

    print("\n--- Timed Steps (real data, VAE encode on-the-fly) ---")
    times = []
    num_steps = 0

    while num_steps < target_steps:
        t0 = time.time()

        batch = next(loader)
        actions = batch["actions"].cuda()

        if "latents" in batch:
            latents = batch["latents"].cuda()
        else:
            frames = batch["frames"].cuda()
            B_actual, T_actual = frames.shape[:2]
            with torch.no_grad():
                flat_frames = frames.view(B_actual * T_actual, 3, data_cfg["target_size"], data_cfg["target_size"])
                latents = encode_frames(vae, flat_frames).detach()
                latents = latents.view(B_actual, T_actual, 4, model_cfg["latent_grid_size"], model_cfg["latent_grid_size"])

        input_latents = latents[:, :-1, :, :, :]
        target_latent = latents[:, -1, :, :, :]
        input_actions = actions[:, :-1]

        optimizer.zero_grad(set_to_none=True)
        pred = model(input_latents, input_actions)
        loss = torch.nn.functional.mse_loss(pred, target_latent)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        torch.cuda.synchronize()
        t = time.time() - t0
        times.append(t)

        mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"  Step {num_steps}: {t:.3f}s | loss={loss.item():.6f} | GPU mem: {mem:.2f} GiB")
        torch.cuda.reset_peak_memory_stats()
        num_steps += 1

    warmup = times[0]
    steady = times[1:]
    avg_steady = sum(steady) / len(steady)

    print(f"\n  Step 1 (warmup):      {warmup:.3f}s")
    print(f"  Steps 2-{target_steps}:   {[f'{t:.3f}s' for t in steady]}")
    print(f"  Avg steady-state:     {avg_steady:.3f}s")
    print(f"  Range:                {min(steady):.3f}s - {max(steady):.3f}s")

    steps_per_epoch = SEQ_COUNT_EST // B
    epochs = config.get("epochs", 2)
    total_h = steps_per_epoch * epochs * avg_steady / 3600
    print("\n--- Stage 1 Estimate (real data, VAE on-the-fly) ---")
    print(f"Steps/epoch:           {steps_per_epoch:,}")
    print(f"Total ({epochs} epochs):    {total_h:.1f}h")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage1_4ctx.yaml")
    ap.add_argument("--real-data", action="store_true",
                    help="stream real shard data instead of synthetic tensors")
    ap.add_argument("--steps", type=int, default=None,
                    help="timed steps (default: 6 synthetic, 10 real-data)")
    args = ap.parse_args()

    config = load_config(args.config)
    check_sdpa_dispatch()
    if args.real_data:
        bench_real_data(config, target_steps=args.steps or 10)
    else:
        bench_synthetic(config, num_steps=args.steps or 6)


if __name__ == "__main__":
    main()
