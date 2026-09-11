"""Measure one real step of the 338M world model on PRO 6000.

Isolates:
  - Step 1 time (CUDA warmup) vs steps 2-5 (steady-state)
  - GPU memory at batch_size=32 with 1024-token context
  - On-the-fly VAE encode time per batch (for precompute decision)

Usage:
  modal run measure_step_time.py
"""

import modal
import time
import torch
from pathlib import Path

from image_modal import world_model_image
from mw_jepa.config import load_config

app = modal.App("measure-step-time")

SEQ_COUNT_EST = 249_000  # 10 shards × 50K frames/shard, stride=2 sliding window


@app.function(
    image=world_model_image,
    gpu="RTX-PRO-6000",
    timeout=60 * 10,
)
def measure():
    config = load_config("/root/jepa/configs/stage1_4ctx.yaml")

    # --- SDPA backend verification ---
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
            print(f"  mem-efficient dispatch (BF16): OK (output shape {out.shape})")
    except RuntimeError as e:
        print(f"  mem-efficient dispatch (BF16): FAILED — {e}")
    try:
        with torch.backends.cuda.sdp_kernel(enable_math=False, enable_mem_efficient=False, enable_flash=True):
            q = torch.randn(1, 1, 128, 8, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(1, 1, 128, 8, device="cuda", dtype=torch.bfloat16)
            v = torch.randn(1, 1, 128, 8, device="cuda", dtype=torch.bfloat16)
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            print(f"  flash dispatch (BF16):        OK (output shape {out.shape})")
    except RuntimeError as e:
        print(f"  flash dispatch (BF16):        FAILED — {e}")
    print()
    model_cfg = config["model"]
    B = config.get("batch_size", 32)
    T = config.get("context", 4)

    print(f"Batch size: {B}, Context: {T}")
    print(f"Tokens per step: {B} * {T} * 256 = {B * T * 256}")

    # --- Model init ---
    from mw_jepa.world_model import MineWorldModel
    model = MineWorldModel(**model_cfg).cuda().train()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {param_count:,}")
    print(f"Model memory (BF16): {param_count * 2 / 1e9:.2f} GB")

    # --- Dummy data ---
    latent_dim = model_cfg["latent_dim"]
    grid_size = model_cfg["latent_grid_size"]
    num_actions = model_cfg["num_actions"]

    dummy_latents = torch.randn(B, T, latent_dim, grid_size, grid_size, device="cuda")
    dummy_actions = torch.randint(0, num_actions, (B, T), device="cuda")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)

    # --- Timed steps ---
    print(f"\n--- Forward + Backward Timing (batch={B}, ctx={T}) ---")
    times = []
    for step in range(6):
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
    print(f"  Steps 2-6 (steady):   {[f'{t:.3f}s' for t in steady]}")
    print(f"  Avg steady-state:     {avg_steady:.3f}s")
    print(f"  Range:                {min(steady):.3f}s - {max(steady):.3f}s")

    # --- Clean peak GPU memory measurement ---
    torch.cuda.reset_peak_memory_stats()
    pred = model(dummy_latents, dummy_actions)
    target = dummy_latents[:, -1, :, :, :]
    loss = torch.nn.functional.mse_loss(pred, target)
    loss.backward()
    torch.cuda.synchronize()
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"\nPeak GPU memory (fwd+bwd+optimizer state): {peak_mem:.2f} GiB")

    # --- Stage 1 estimate ---
    steps_per_epoch = SEQ_COUNT_EST // B
    epochs = config.get("epochs", 2)
    print(f"\n--- Stage 1 Estimate ({epochs} epochs, {T}-ctx) ---")
    print(f"Sequences (10 shards): {SEQ_COUNT_EST:,}")
    print(f"Steps/epoch:           {steps_per_epoch:,}")
    total_steady_h = steps_per_epoch * epochs * avg_steady / 3600
    total_steady_cost = total_steady_h * 3.03
    print(f"Transformer only:      {total_steady_h:.1f}h ~ ${total_steady_cost:.0f}")

    # --- VAE encode timing ---
    print(f"\n--- VAE Encode Timing (batch={B}) ---")
    from mw_jepa.vae import load_vae, encode_frames
    vae = load_vae(device="cuda").eval()
    vae_param_count = sum(p.numel() for p in vae.parameters())

    dummy_frames = torch.rand(B, 3, config["data"]["target_size"], config["data"]["target_size"], device="cuda")

    with torch.no_grad():
        # Warmup
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

    # --- With VAE overhead ---
    total_step = avg_steady + vae_time
    total_h = steps_per_epoch * epochs * total_step / 3600
    total_cost = total_h * 3.03
    print(f"\n--- With On-the-Fly VAE Encode ---")
    print(f"Combined step:         {avg_steady:.3f}s (transformer) + {vae_time:.3f}s (VAE) = {total_step:.3f}s")
    print(f"Total time:            {total_h:.1f}h")
    print(f"Total cost:            ${total_cost:.0f}")
    print(f"VAE share of step:     {vae_time / total_step * 100:.0f}%")

    # Peak memory with both loaded
    print(f"\nCombined GPU memory (both models loaded): ~{peak_mem + vae_mem:.1f} GiB")

    # --- Precompute benefit ---
    print(f"\n--- Precompute Benefit ---")
    print(f"VAE is {vae_time / total_step * 100:.0f}% of each step.")
    print(f"Precomputing removes this entirely (one-time cost of encoding all frames once).")
    print(f"Effective speedup: 1 / (1 - {vae_time / total_step:.3f}) = {1 / (1 - vae_time / total_step):.1f}x")
    print(f"Stage 1 with precompute: ${total_steady_cost:.0f} (vs ${total_cost:.0f})")


if __name__ == "__main__":
    measure.local()
