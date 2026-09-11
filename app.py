import modal
from pathlib import Path

from image_modal import world_model_image, vae_image, inference_image
from src.config import load_config

app = modal.App("minecraft-world-model")

# --- Persistent Storage (checkpoints only - no raw data) ---
ckpt_volume = modal.Volume.from_name("minecraft-checkpoints", create_if_missing=True)

# The TESS dataset is public (no HF auth needed). W&B is optional handled in each function.

CKPT_DIR = "/checkpoints"
TESS_SHARDS_TOTAL = 303


# ──────────────────────────────────────────
# Phase 3: VAE Fine-Tuning (streams 2 shards)
# ──────────────────────────────────────────

@app.function(
    image=vae_image,
    gpu="RTX-PRO-6000",
    timeout=60 * 60 * 12,
    volumes={CKPT_DIR: ckpt_volume},
)
def train_vae(
    shard_start: int = 0,
    shard_end: int = 2,
    num_steps: int = 50000,
    batch_size: int = 16,
    lr: float = 1e-5,
    run_name: str = "vae-finetune-v1",
):
    """Fine-tune VAE on streaming TESS VLA frames."""
    import torch
    from src.vae import load_vae, finetune_vae
    from src.data import MinecraftFrameStream
    from torch.utils.data import DataLoader

    # Optional W&B — no crash if unauthenticated
    wandb = None
    try:
        import wandb as _wandb
        _wandb.init(project="minecraft-world-model", name=run_name)
        wandb = _wandb
    except Exception:
        pass

    stream = MinecraftFrameStream(shard_start=shard_start, shard_end=shard_end, target_size=256)
    loader = DataLoader(stream, batch_size=batch_size, num_workers=0, pin_memory=True, drop_last=True)

    vae = load_vae(device="cuda")
    vae = finetune_vae(
        vae=vae,
        dataloader=loader,
        num_steps=num_steps,
        lr=lr,
        log_callback=lambda metrics: wandb.log(metrics) if wandb else None,
    )

    torch.save(vae.state_dict(), f"{CKPT_DIR}/{run_name}.pt")
    ckpt_volume.commit()
    if wandb:
        wandb.finish()


# ──────────────────────────────────────────
# VAE reconstruction evaluation (rFVD / PSNR / SSIM / LPIPS)
# ──────────────────────────────────────────

@app.function(
    image=vae_image,
    gpu="A10G",
    timeout=60 * 60 * 2,
    volumes={CKPT_DIR: ckpt_volume},
)
def evaluate_vae(
    shard: int = 3,
    num_frames: int = 500,
    batch_size: int = 32,
    use_finetuned: bool = False,
):
    """Evaluate VAE reconstruction quality on a held-out shard."""
    import torch
    from src.vae import load_vae, evaluate_vae as run_eval
    from src.data import MinecraftFrameStream
    from torch.utils.data import DataLoader

    vae = load_vae(device="cuda")
    if use_finetuned:
        ckpt_path = f"{CKPT_DIR}/vae-finetune-v1.pt"
        if Path(ckpt_path).exists():
            vae.load_state_dict(torch.load(ckpt_path))
            print(f"Loaded fine-tuned VAE from {ckpt_path}")
        else:
            print("No fine-tuned checkpoint found, using base VAE")

    stream = MinecraftFrameStream(shard_start=shard, shard_end=shard + 1, target_size=256)
    loader = DataLoader(stream, batch_size=batch_size, num_workers=0, drop_last=True)

    results = run_eval(vae, loader, num_frames=num_frames)
    for k, v in results.items():
        print(f"{k}: {v:.4f}" if v is not None else f"{k}: N/A")

    return results


# ──────────────────────────────────────────
# Latent statistics (streams 3 shards)
# ──────────────────────────────────────────

@app.function(
    image=vae_image,
    gpu="A10G",
    timeout=60 * 60 * 2,
    volumes={CKPT_DIR: ckpt_volume},
)
def compute_latent_stats(
    shard_start: int = 0,
    shard_end: int = 3,
    num_samples: int = 5000,
):
    """Compute channel-wise mean/std of VAE latents for normalization."""
    import torch
    from src.vae import load_vae
    from src.data import LatentStatsComputer

    vae = load_vae(device="cuda")
    computer = LatentStatsComputer(vae=vae, shard_start=shard_start, shard_end=shard_end, target_size=256)
    stats = computer.compute(num_samples=num_samples)

    torch.save(stats, f"{CKPT_DIR}/latent_stats.pt")
    ckpt_volume.commit()

    mean, std = stats["mean"], stats["std"]
    print(f"Latent stats saved: mean={mean}, std={std}")
    print(f"  Mean shape: {tuple(mean.shape)}  (expect [4])")
    print(f"  Std  shape: {tuple(std.shape)}   (expect [4])")

    # Sanity checks
    if mean.shape != (4,):
        print(f"  WARNING: mean has unexpected shape {mean.shape}")
    if std.shape != (4,):
        print(f"  WARNING: std has unexpected shape {std.shape}")
    if (std < 1e-4).any():
        print(f"  WARNING: near-zero std detected ({std.tolist()}) — latent space may be collapsed")
    if (std > 10).any():
        print(f"  WARNING: std > 10 ({std.tolist()}) — latent values may explode")
    if (mean.abs() > 10).any():
        print(f"  WARNING: |mean| > 10 ({mean.tolist()}) — check normalization")
    print("  Latent stats look plausible" if all(s > 1e-4 for s in std) and all(m.abs() < 10 for m in mean)
          else "  Latent stats need review")


# ──────────────────────────────────────────
# Pre-compute latents (offline, speeds up world model training)
# ──────────────────────────────────────────

@app.function(
    image=vae_image,
    gpu="A10G",
    timeout=60 * 60 * 6,
    volumes={CKPT_DIR: ckpt_volume},
)
def precompute_latents(
    shard_start: int = 0,
    shard_end: int = 10,
    batch_size: int = 32,
):
    """Encode all frames from shards into latents and save to volume.
    Speeds up world model training by avoiding on-the-fly VAE encoding.
    """
    import torch
    from pathlib import Path
    from src.vae import encode_frames, load_vae
    from src.data import MinecraftFrameStream
    from torch.utils.data import DataLoader

    vae = load_vae(device="cuda")

    latent_dir = Path(f"{CKPT_DIR}/latents")
    latent_dir.mkdir(parents=True, exist_ok=True)

    for shard in range(shard_start, shard_end):
        out_path = latent_dir / f"shard_{shard:05d}.pt"
        if out_path.exists():
            print(f"Shard {shard} already cached at {out_path}, skipping")
            continue

        stream = MinecraftFrameStream(shard_start=shard, shard_end=shard + 1)
        loader = DataLoader(stream, batch_size=batch_size, num_workers=0, drop_last=False)

        all_latents = []
        for batch in loader:
            with torch.no_grad():
                z = encode_frames(vae, batch.cuda()).cpu()
            all_latents.append(z)

        if all_latents:
            latents = torch.cat(all_latents, dim=0)
            torch.save(latents, out_path)
            print(f"Saved shard {shard}: {latents.shape} -> {out_path}")

        ckpt_volume.commit()

    print(f"Pre-computed latents for shards {shard_start}-{shard_end - 1}")


# ──────────────────────────────────────────
# Phase 4-5: World Model Training (streams shards)
# ──────────────────────────────────────────

@app.function(
    image=world_model_image,
    gpu="RTX-PRO-6000",
    timeout=86400,
    volumes={CKPT_DIR: ckpt_volume},
    retries=modal.Retries(max_retries=2),
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def train_world_model(
    config_path: str | None = None,
    shard_start: int = 0,
    shard_end: int = 10,
    resume_from: str | None = None,
    use_precomputed_latents: bool = False,
):
    """Train world model. Streams data from TESS VLA on HuggingFace."""
    import torch
    from pathlib import Path
    from src.world_model import MineWorldModel
    from src.data import WorldModelStream
    from src.trainer import Trainer

    if config_path is None:
        config_path = str(Path(__file__).parent / "configs" / "stage1_4ctx.yaml")
    cfg = load_config(config_path)

    wandb = None
    try:
        import wandb as _wandb
        _wandb.init(project="minecraft-world-model", name=cfg["run_name"], config=cfg)
        wandb = _wandb
    except Exception:
        pass

    model = MineWorldModel(**cfg["model"])
    model.cuda()

    latent_dir = f"{CKPT_DIR}/latents" if use_precomputed_latents else None

    if use_precomputed_latents:
        vae = None
    else:
        from src.vae import load_vae
        vae = load_vae(device="cuda")

    train_stream = WorldModelStream(
        sequence_length=cfg["context"],
        shard_start=shard_start,
        shard_end=shard_end,
        target_size=cfg.get("data", {}).get("target_size", 256),
        latent_dir=latent_dir,
    )

    trainer = Trainer(
        model=model,
        vae=vae,
        config=cfg,
        ckpt_dir=Path(CKPT_DIR),
    )

    if resume_from:
        trainer.load_checkpoint(resume_from)

    trainer.train(
        train_stream,
        batch_size=cfg.get("batch_size", cfg.get("training", {}).get("batch_size", 8)),
        num_epochs=cfg.get("epochs", cfg.get("training", {}).get("epochs", 10)),
    )

    if wandb:
        wandb.finish()


# ──────────────────────────────────────────
# Phase 6: Inference
# ──────────────────────────────────────────

@app.cls(
    image=inference_image,
    gpu="A10G",
    timeout=60 * 60,
    volumes={CKPT_DIR: ckpt_volume},
    scaledown_window=300,
    min_containers=1,
)
class WorldModelInference:
    """Deployed inference endpoint."""

    def __init__(self):
        self.model = None
        self.vae = None
        self.latent_mean = None
        self.latent_std = None

    @modal.enter()
    def load_models(self):
        import torch
        from src.world_model import MineWorldModel
        from src.vae import load_vae

        self.device = torch.device("cuda")
        self.vae = load_vae(device=self.device)

        # Load fine-tuned VAE if available
        vae_ckpt = f"{CKPT_DIR}/vae-finetune-v1.pt"
        if Path(vae_ckpt).exists():
            self.vae.load_state_dict(torch.load(vae_ckpt))

        # Load world model — gracefully handle missing or shape-mismatched checkpoints
        self.model = MineWorldModel(latent_grid_size=32)
        best_ckpt = f"{CKPT_DIR}/best.pt"
        if Path(best_ckpt).exists():
            try:
                ckpt = torch.load(best_ckpt, map_location=self.device)
                self.model.load_state_dict(ckpt["model"])
                print(f"Loaded world model from {best_ckpt} (loss {ckpt.get('loss', '?'):.6f})")
            except RuntimeError as e:
                print(f"Warning: checkpoint {best_ckpt} incompatible with model config — skipping ({e})")
        self.model.eval().to(self.device)

        # Load latent stats
        stats_path = Path(f"{CKPT_DIR}/latent_stats.pt")
        if stats_path.exists():
            stats = torch.load(stats_path, map_location=self.device)
            self.latent_mean = stats["mean"]
            self.latent_std = stats["std"]

    @modal.fastapi_endpoint(method="POST")
    def step(self, data: dict):
        import torch
        import numpy as np
        from src.vae import decode_latents

        latent_history = torch.tensor(
            data["latent_history"], device=self.device, dtype=torch.float32
        )
        action_token = data["action_token"]

        if self.latent_mean is not None:
            mean = self.latent_mean.to(self.device).view(1, 4, 1, 1)
            std = self.latent_std.to(self.device).view(1, 4, 1, 1)
            latent_history = (latent_history - mean) / (std + 1e-6)

        T = latent_history.shape[0]
        context = 16
        if T < context:
            pad = latent_history[0:1].expand(context - T, -1, -1, -1)
            latent_history = torch.cat([pad, latent_history], dim=0)
        else:
            latent_history = latent_history[-context:]

        actions = torch.full((context,), action_token, dtype=torch.long, device=self.device)

        with torch.no_grad():
            next_latent = self.model(latent_history.unsqueeze(0), actions.unsqueeze(0))
            if self.latent_mean is not None:
                mean = self.latent_mean.to(self.device).view(1, 4, 1, 1)
                std = self.latent_std.to(self.device).view(1, 4, 1, 1)
                next_latent = next_latent * (std + 1e-6) + mean
            frame = decode_latents(self.vae, next_latent)
            frame_np = frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
            frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)

        return {"frame": frame_np.tolist(), "latent": next_latent.squeeze(0).cpu().numpy().tolist()}

    @modal.fastapi_endpoint(method="POST")
    def reset(self, data: dict):
        import torch
        import numpy as np
        from src.vae import encode_frames

        seed = np.array(data["seed_frame"], dtype=np.float32)
        if seed.max() > 1.0:
            seed = seed / 255.0
        if seed.ndim == 3 and seed.shape[-1] == 3:
            seed = np.transpose(seed, (2, 0, 1))
        seed_t = torch.tensor(seed, device=self.device).unsqueeze(0)

        with torch.no_grad():
            latent = encode_frames(self.vae, seed_t)
        if self.latent_mean is not None:
            mean = self.latent_mean.to(self.device).view(1, 4, 1, 1)
            std = self.latent_std.to(self.device).view(1, 4, 1, 1)
            latent = (latent - mean) / (std + 1e-6)

        return {"status": "ok", "latent": latent.squeeze(0).cpu().numpy().tolist()}


# ──────────────────────────────────────────
# Pre-flight Canary (run before any real training)
# ──────────────────────────────────────────

@app.function(
    image=vae_image,
    gpu="A10G",
    timeout=60 * 30,
    volumes={CKPT_DIR: ckpt_volume},
)
def run_canary():
    """End-to-end pre-flight canary.

    Proves the full spine before committing GPU hours:
      1. Frame stream → SD-VAE encode/decode → reconstruction grid
      2. Action token distribution (no-op ratio, camera vs keyboard)
      3. World model forward + backward pass
      4. Checkpoint save → load → verify round-trip
    """
    import torch
    from pathlib import Path
    from torch.utils.data import DataLoader
    from torchvision.utils import make_grid, save_image
    from datasets import load_dataset

    from src.vae import load_vae, encode_frames, decode_latents
    from src.data import MinecraftFrameStream, WorldModelStream
    from src.world_model import MineWorldModel
    from src.trainer import Trainer, collate_stream
    from src.action_tokenizer import parse_lumine_action, KEYBOARD_TOKENS
    from src.config import load_config

    jepa_root = Path(__file__).parent / "jepa"

    errors = []

    # ── Phase 1: VAE reconstruction ──
    print("=" * 60)
    print("PHASE 1: Base SD-VAE Reconstruction")
    print("=" * 60)

    vae = load_vae(device="cuda")
    stream = MinecraftFrameStream(shard_start=0, shard_end=1, target_size=256)
    loader = DataLoader(stream, batch_size=4, num_workers=0, drop_last=True)

    orig_batch = next(iter(loader)).cuda()
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
    save_image(grid, f"{CKPT_DIR}/canary_reconstruction.png")
    ckpt_volume.commit()
    print(f"  Reconstruction grid → {CKPT_DIR}/canary_reconstruction.png")
    print(f"  Rows: original | reconstructed | per-pixel absolute error")
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
    for i, ex in enumerate(ds):
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

    cfg = load_config(str(jepa_root / "configs" / "stage1_4ctx.yaml"))
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
    batch = next(iter(wm_loader))

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

    trainer = Trainer(model=model, vae=vae, config=cfg, ckpt_dir=Path(CKPT_DIR))
    trainer.save_checkpoint(loss.item())
    print(f"  Checkpoint saved → {CKPT_DIR}/latest.pt")

    model2 = MineWorldModel(**cfg["model"]).cuda()
    trainer2 = Trainer(model=model2, vae=vae, config=cfg, ckpt_dir=Path(CKPT_DIR))
    trainer2.load_checkpoint(f"{CKPT_DIR}/latest.pt")

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


# ──────────────────────────────────────────
# Overfit Test (pre-flight: verify gradients + data work end-to-end)
# ──────────────────────────────────────────

@app.function(
    image=vae_image,
    gpu="A10G",
    timeout=60 * 60,
    volumes={CKPT_DIR: ckpt_volume},
)
def test_overfit(
    sequence_length: int = 4,
    num_trajectory_frames: int = 20,
    num_epochs: int = 20,
    shuffle_repeats: int = 20,
):
    """Overfit a tiny world model on a single repeated trajectory with real actions.

    Pass condition: final loss < 20% of initial loss (proves gradients flow).

    Post-training diagnostics:
    1. Copy-paste baseline — does trained loss beat "frame t+1 ≈ frame t"?
    2. Shuffled action comparison — are actions actually steering predictions?
    3. Per-quadrant error map — does action-driven error localise to expected regions?

    The trajectory is chosen from shard 0, action distribution is printed.
    """
    import torch
    from pathlib import Path
    from torch.utils.data import DataLoader, IterableDataset
    from torchvision.utils import make_grid, save_image
    from datasets import load_dataset

    from src.vae import load_vae, encode_frames, decode_latents
    from src.data import decode_jpeg, TESS_REPO
    from src.action_tokenizer import parse_lumine_action, KEYBOARD_TOKENS
    from src.world_model import MineWorldModel
    from src.trainer import Trainer, collate_stream
    from src.config import load_config

    jepa_root = Path(__file__).parent / "jepa"

    # ── Step 1: CPU-only action pre-scan (no GPU cost) ──
    print("=" * 60)
    print("OVERFIT TEST: Single trajectory memorisation (real actions)")
    print("=" * 60)
    print("\n  Scanning shard 0 for a trajectory with sufficient action diversity ...")
    token_names = {v: k for k, v in KEYBOARD_TOKENS.items()}

    def find_good_trajectory(shard, min_frames, max_static_ratio=0.80, min_unique_actions=2):
        """Stream dataset parsing actions only (CPU). Returns (action_tokens, frames_bytes) or None."""
        ds = load_dataset(
            TESS_REPO, split="train", streaming=True,
            data_files=f"data/shard_{shard:05d}.parquet",
        )
        action_tokens, frames_bytes = [], []
        last_video_id = None
        action_counts = {}
        for ex in ds:
            video_id = ex["video_id"]
            if last_video_id is None:
                last_video_id = video_id
            if video_id != last_video_id:
                if len(action_tokens) >= min_frames:
                    break
                action_tokens, frames_bytes = [], []
                action_counts = {}
                last_video_id = video_id
            if len(action_tokens) >= min_frames:
                break
            token = parse_lumine_action(ex["action"])
            action_tokens.append(token)
            frames_bytes.append(ex["image"])
            action_counts[token] = action_counts.get(token, 0) + 1

        if len(action_tokens) < min_frames:
            return None, None, None

        esc_count = action_counts.get(0, 0)
        noop_count = action_counts.get(28, 0)
        static_ratio = (esc_count + noop_count) / len(action_tokens)
        if static_ratio > max_static_ratio:
            return None, None, None

        # Require action diversity — at least `min_unique_actions` different
        # non-static action tokens. This ensures shuffled-vs-correct comparison
        # is meaningful (shuffling all-identical actions gives the same sequence).
        unique_non_static = set(t for t in action_tokens if t not in (0, 28))
        if len(unique_non_static) < min_unique_actions:
            return None, None, None

        return action_tokens, frames_bytes, action_counts

    # Scan shards until we find a good trajectory
    action_tokens, frames_bytes, action_counts = None, None, None
    for shard in range(5):  # try shards 0-4
        result = find_good_trajectory(shard, min_frames=num_trajectory_frames)
        if result[0] is not None:
            action_tokens, frames_bytes, action_counts = result
            print(f"  Found suitable trajectory in shard {shard}")
            break
        print(f"  Shard {shard}: too static, skipping ...")

    if action_tokens is None:
        raise RuntimeError(
            "No suitable trajectory found across shards 0-4. "
            "Try running `scripts/check_dataset.py` to find active shards."
        )

    # Print action distribution for this trajectory
    total_acts = len(action_tokens)
    print(f"\n  Trajectory action distribution ({total_acts} frames):")
    keyboard_acts = sum(c for t, c in action_counts.items() if t <= 21)
    camera_acts = sum(c for t, c in action_counts.items() if 22 <= t <= 27)
    center_acts = action_counts.get(28, 0)
    esc_acts = action_counts.get(0, 0)
    print(f"    Keyboard (0-21):    {keyboard_acts} ({100*keyboard_acts/total_acts:.0f}%)")
    if camera_acts:
        print(f"    Camera  (22-27):    {camera_acts} ({100*camera_acts/total_acts:.0f}%)")
    print(f"    Center  (28):       {center_acts} ({100*center_acts/total_acts:.0f}%)")
    for tid in sorted(action_counts):
        if tid > 21:
            continue
        label = token_names.get(tid, f"key_{tid}")
        print(f"      [{tid:2d}] {label:<12s} {action_counts[tid]}")

    # Print raw action token sequence
    seq_labels = []
    for t in action_tokens:
        if t <= 21:
            seq_labels.append(token_names.get(t, f"k{t}"))
        elif t <= 27:
            seq_labels.append(f"cam{t}")
        else:
            seq_labels.append("noop")
    print(f"\n  Action sequence ({len(seq_labels)} frames):")
    for row_start in range(0, len(seq_labels), 10):
        row = seq_labels[row_start:row_start + 10]
        print(f"    [{row_start:3d}] " + " ".join(f"{a:<8s}" for a in row))
    print()

    # ── Step 2: GPU work — load VAE, decode frames, encode latents ──
    print("  Loading VAE and encoding frames ...")
    vae = load_vae(device="cuda")

    def validate_frame(frame: torch.Tensor, tol: float = 1e-6) -> tuple[bool, str]:
        """Check frame is not corrupted: non-flat, non-NaN, non-degenerate."""
        if torch.isnan(frame).any():
            return False, "NaN values detected"
        mean = frame.mean().item()
        if mean < tol:
            return False, f"all-black frame (mean={mean:.6f})"
        if (frame > tol).sum() < 10:
            return False, f"frame has only {(frame > tol).sum()} non-zero pixels"
        return True, "OK"

    frames = []
    for idx, fb in enumerate(frames_bytes):
        f = decode_jpeg(fb, 256)
        ok, msg = validate_frame(f)
        if not ok:
            raise RuntimeError(
                f"Frame {idx} in trajectory failed validation: {msg}. "
                "Trajectory contains corrupted data; re-run will scan a different shard."
            )
        frames.append(f)

    frames_tensor = torch.stack(frames).cuda()
    with torch.no_grad():
        latents = encode_frames(vae, frames_tensor).cpu()
    action_tensor = torch.tensor(action_tokens, dtype=torch.long)

    # Build sliding-window sequences
    seq_len = sequence_length + 1
    sequences = []
    for i in range(0, len(frames) - seq_len + 1, seq_len // 2):
        sequences.append({
            "latents": latents[i:i + seq_len],
            "actions": action_tensor[i:i + seq_len],
        })

    print(f"  {len(sequences)} sequences of length {seq_len} (stride {seq_len // 2})")

    class RepeatDataset(IterableDataset):
        def __init__(self, seqs, repeats):
            self.seqs = seqs
            self.repeats = repeats
        def __iter__(self):
            for _ in range(self.repeats):
                for s in self.seqs:
                    yield s

    train_stream = RepeatDataset(sequences, repeats=shuffle_repeats)

    # ── Miniature model ──
    cfg = load_config(str(jepa_root / "configs" / "stage1_4ctx.yaml"))
    cfg["model"].update({
        "embed_dim": 256,
        "num_blocks": 4,
        "num_heads": 4,
    })
    cfg["epochs"] = num_epochs
    cfg["batch_size"] = min(8, len(sequences))

    model = MineWorldModel(**cfg["model"]).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} params")

    # ── Train ──
    trainer = Trainer(model=model, vae=None, config=cfg, ckpt_dir=Path(CKPT_DIR) / "overfit_test")
    trainer.train(train_stream, batch_size=cfg["batch_size"], num_epochs=num_epochs)

    # ── Post-training diagnostics ──
    print()
    print("=" * 60)
    print("POST-TRAINING DIAGNOSTICS")
    print("=" * 60)

    # Hold-out batch (first sequence, not used during training if drop_last rounds down)
    diag_batch = sequences[0]
    diag_latents = diag_batch["latents"].unsqueeze(0).cuda()  # [1, T, 4, 32, 32]
    diag_actions = diag_batch["actions"].unsqueeze(0).cuda()    # [1, T]
    in_latents = diag_latents[:, :-1]
    target = diag_latents[:, -1]
    in_actions = diag_actions[:, :-1]

    model.eval()
    with torch.no_grad():
        # 1. Copy-paste baseline: predict last input frame as next frame
        last_input = in_latents[:, -1]
        baseline_loss = torch.nn.functional.mse_loss(last_input, target).item()

        # 2. Correct actions
        pred_correct = model(in_latents, in_actions)
        correct_loss = torch.nn.functional.mse_loss(pred_correct, target).item()

        # 3. Shuffled actions (reverse temporal order, preserves distribution)
        shuffled = in_actions.flip(dims=[1])
        with torch.no_grad():
            pred_shuffled = model(in_latents, shuffled)
        shuffled_loss = torch.nn.functional.mse_loss(pred_shuffled, target).item()

        # 4. Decode to pixels for visual comparison
        frame_target = decode_latents(vae, target).cpu()
        frame_correct = decode_latents(vae, pred_correct).cpu()
        frame_shuffled = decode_latents(vae, pred_shuffled).cpu()
        frame_baseline = decode_latents(vae, last_input).cpu()

    print(f"  Copy-paste baseline:  {baseline_loss:.6f}  (predict frame t+1 = frame t)")
    print(f"  Correct actions:      {correct_loss:.6f}")
    print(f"  Shuffled actions:     {shuffled_loss:.6f}")
    print(f"  Action-conditioning delta: {shuffled_loss - correct_loss:+.6f} "
          f"(positive = actions are steering predictions)")
    print()

    # 5. Per-quadrant error breakdown (latent space)
    print("  Per-quadrant latent error (correct vs shuffled):")
    _, C, H, W = target.shape
    q_h, q_w = H // 2, W // 2
    quadrant_names = ["top-left", "top-right", "bottom-left", "bottom-right"]
    quadrant_deltas = []
    for idx, name in enumerate(quadrant_names):
        row, col = idx // 2, idx % 2
        sl = (slice(None), slice(None),
              slice(row * q_h, (row + 1) * q_h),
              slice(col * q_w, (col + 1) * q_w))
        err_correct = torch.nn.functional.mse_loss(pred_correct[sl], target[sl]).item()
        err_shuff = torch.nn.functional.mse_loss(pred_shuffled[sl], target[sl]).item()
        delta = err_shuff - err_correct
        quadrant_deltas.append(delta)
        print(f"    {name:14s}  correct={err_correct:.6f}  shuffled={err_shuff:.6f}  "
              f"delta={delta:+.6f}")

    max_delta = max(quadrant_deltas)
    min_delta = min(quadrant_deltas)
    spatial_spread = max_delta - min_delta
    print(f"  Quadrant delta spread: {spatial_spread:.6f} "
          f"(larger = action effects localise to specific regions)")
    print()

    # 6. Tensor stats for each grid cell (avoids "looks black" ambiguity)
    print("  Pixel-level stats per grid cell (values in [0,1]):")
    for label, tensor in [("target", frame_target), ("correct", frame_correct),
                          ("shuffled", frame_shuffled), ("baseline", frame_baseline)]:
        err = (tensor - frame_target).abs()
        print(f"    {label:12s}  pred  min={tensor.min():.4f}  max={tensor.max():.4f}  mean={tensor.mean():.4f}")
        print(f"    {'':12s}  error min={err.min():.4f}      max={err.max():.4f}      mean={err.mean():.4f}")

    # 7. Save comparison grid to volume
    vis_orig = decode_latents(vae, in_latents[:, -1]).cpu()  # last input frame
    vis_rows = []
    for label, tensor in [("target", frame_target), ("correct", frame_correct),
                          ("shuffled", frame_shuffled), ("baseline", frame_baseline)]:
        err_map = (tensor - frame_target).abs()
        stacked = torch.cat([tensor, err_map], dim=0)
        vis_rows.append(stacked)

    grid = make_grid(torch.cat(vis_rows, dim=0), nrow=2)
    save_image(grid, f"{CKPT_DIR}/overfit_diagnostics.png")
    ckpt_volume.commit()
    print(f"  Diagnostic grid → {CKPT_DIR}/overfit_diagnostics.png")
    print(f"  Columns: prediction | error heatmap")
    print(f"  Rows:    target | correct actions | shuffled actions | copy-paste baseline")
    print()

    # ── Results ──
    print("=" * 60)
    print("OVERFIT RESULTS")
    print("=" * 60)

    losses = trainer.loss_history
    initial, final = losses[0], losses[-1]
    print(f"  Initial loss:          {initial:.6f}")
    print(f"  Final loss:            {final:.6f}")
    print(f"  Ratio:                 {final / initial:.4f} (goal: < 0.20)")

    spark = "".join(
        "█" if v == max(losses)
        else "▄" if v > (max(losses) + min(losses)) / 2
        else " " for v in losses
    )
    print(f"  Trend:                 [{spark}]")
    print()

    # Pass/fail logic
    gradient_pass = final < initial * 0.20
    conditioning_pass = shuffled_loss > correct_loss * 1.10  # >10% worse under shuffled actions
    baseline_pass = correct_loss < baseline_loss * 0.95      # >5% better than copy-paste

    failures = []
    if not gradient_pass:
        failures.append("loss did not drop sufficiently (gradient issue)")
    if not conditioning_pass:
        failures.append(
            f"shuffled actions only {((shuffled_loss / correct_loss) - 1) * 100:.1f}% worse "
            f"(threshold: 10%) — action conditioning may be dead"
        )
    if not baseline_pass:
        failures.append(
            f"trained loss {correct_loss:.4f} does not beat copy-paste baseline "
            f"{baseline_loss:.4f}"
        )

    if failures:
        print(f"  RESULT: FAIL — {failures[0]}")
        raise RuntimeError("Overfit test failed: " + "; ".join(failures))
    else:
        print("  RESULT: PASS")
        if conditioning_pass:
            print("  [OK]  Actions are steering predictions (shuffled actions increase loss)")
        if baseline_pass:
            print("  [OK]  Model beats the copy-paste baseline")
        print("  [OK]  Training loop verified")

    print("=" * 60)


# ──────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────

@app.local_entrypoint()
def main():
    print("Minecraft World Model — Modal App")
    print()
    print("Storage strategy: data streams from HuggingFace (zero persistent storage)")
    print("Checkpoints only on volume (~10GB max = ~$1/month)")
    print()
    print("PRE-FLIGHT (always run first):")
    print("  modal run app.py::run_canary                  # Spine check (~5 min, ~$0.09)")
    print("  modal run app.py::test_overfit                # Overfit a single trajectory (~2 min, ~$0.04)")
    print()
    print("Commands:")
    print("  modal run app.py::precompute_latents           # Phase 3: Pre-encode latents (speeds up WM)")
    print("  modal run app.py::train_vae                    # Phase 3: VAE fine-tuning")
    print("  modal run app.py::compute_latent_stats         # Pre-compute latent normalization (validates)")
    print("  modal run app.py::train_world_model            # Phase 4-5: World model training")
    print("  modal serve inference_app.py                   # Phase 6: Web inference server")
