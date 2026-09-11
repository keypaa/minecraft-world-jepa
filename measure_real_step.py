"""Measure real-data step time for the 338M world model.

Streams actual data from HuggingFace, decodes JPEGs, encodes through VAE,
and times the full training step. Runs ~10 steps on shard 0.

Usage:
  modal run measure_real_step.py::measure_real_data
"""

import modal
import time
import torch
from pathlib import Path

from image_modal import world_model_image
from src.config import load_config

app = modal.App("measure-step-time")

SEQ_COUNT_EST = 249_000  # 10 shards × 50K frames/shard, stride=2 sliding window

MODEL_CKPT_DIR = "/checkpoints"

ckpt_volume = modal.Volume.from_name("minecraft-checkpoints", create_if_missing=True)


@app.function(
    image=world_model_image,
    gpu="RTX-PRO-6000",
    timeout=60 * 10,
    secrets=[modal.Secret.from_name("huggingface-token")],
    volumes={MODEL_CKPT_DIR: ckpt_volume},
)
def measure_real_data():
    config = load_config("/root/jepa/configs/stage1_4ctx.yaml")
    model_cfg = config["model"]
    data_cfg = config["data"]
    B = config.get("batch_size", 32)
    T = config.get("context", 4)

    print(f"Batch size: {B}, Context: {T}")
    print(f"Tokens per step: {B} * {T} * 256 = {B * T * 256}")

    # --- Load VAE ---
    from src.vae import load_vae, encode_frames
    print("\nLoading VAE...")
    vae = load_vae(device="cuda").eval()
    print(f"  VAE loaded ({sum(p.numel() for p in vae.parameters()):,} params)")

    # --- Load world model ---
    from src.world_model import MineWorldModel
    print("Loading world model...")
    model = MineWorldModel(**model_cfg).cuda().train()
    print(f"  Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)

    # --- Stream real data ---
    from src.data import WorldModelStream
    from torch.utils.data import DataLoader
    from src.trainer import collate_stream

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

    # --- Timed steps ---
    print(f"\n--- Timed Steps (real data, VAE encode on-the-fly) ---")
    times = []
    num_steps = 0
    target_steps = 10

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

    # --- Estimate ---
    steps_per_epoch = SEQ_COUNT_EST // B
    epochs = config.get("epochs", 2)
    total_h = steps_per_epoch * epochs * avg_steady / 3600
    total_cost = total_h * 3.03
    print(f"\n--- Stage 1 Estimate (real data, VAE on-the-fly) ---")
    print(f"Steps/epoch:           {steps_per_epoch:,}")
    print(f"Total ({epochs} epochs):    {total_h:.1f}h ~ ${total_cost:.0f}")

    # Compare with synthetic measurement
    print(f"\n  Compare: synthetic (tensor-only) was 1.251s — real-data overhead adds {avg_steady - 1.251:.3f}s/step")
