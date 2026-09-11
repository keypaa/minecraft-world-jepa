# Minecraft World Model — Status & Tracking

*Plan: `minecraft_world_model_plan.md` (925 lines)*

---

## Phase 1: Environment & Storage Prep

**Hardware**: Windows 11, AMD Ryzen 5 4500U, iGPU only (2GB). Training via **Modal cloud GPUs**.

**Completed**:
- [x] WSL2 Ubuntu 24.04 available
- [x] Modal CLI 1.4.2 authenticated (workspace: `keypaa`)
- [x] Modal volumes created: `minecraft-checkpoints`, `vae-checkpoints`
- [x] Project structure: 17 source files across `src/`, `configs/`, scripts
- [x] All source modules import-clean and smoke-tested

**Key decisions**:
- Training runs on Modal cloud GPUs; data streams from HuggingFace (zero persistent storage)
- `train_vae` + `train_world_model` on PRO 6000 ($3.03/h, 3-4× faster than A10G for ~same total cost)
- Everything else on A10G ($1.10/h)
- Pre-computed latents on volume to halve world model training time by skipping on-the-fly VAE encode
- Resolution: **256×256** (clean output, 256 tokens/frame, ~$22 total, fits PRO 6000)
- No `minecraft-data` volume — data is transient per container

---

## Phase 2: Data Factory (VPT Pipeline)

**Completed**:
- [x] Lumine action parser → JARVIS-VLA 51-token scheme
- [x] `MinecraftFrameStream`: streaming frames for VAE fine-tuning
- [x] `WorldModelStream`: streaming trajectory sequences for world model
- [x] `LatentStatsComputer`: compute channel-wise mean/std of latents
- [x] `collate_stream` for batching streaming data
- [x] Trainer updated to encode latents on-the-fly during training

**Source**: `TESS-Computer/minecraft-vla-stage1` (303 parquet shards, 15M frames, 5 FPS, Lumine action format)

**Key decisions**:
- Zero data on Modal volumes — stream parquet shards from HF per run
- Pre-computed latents on volume for world model training (halves training time)
- Resolution: 256×256 (clean output, 256 tokens/frame)

**Ready to run** (in order):
```bash
modal run app.py::precompute_latents       # Pre-encode latents (optional but speeds up WM 2×)
modal run app.py::train_vae                # Phase 3: 2 shards, 50K steps
modal run app.py::compute_latent_stats     # Phase 3: 3 shards, 5K samples
modal run app.py::train_world_model --config configs/stage1_4ctx.yaml  # Phase 4: 10 shards
```

---

## Phase 3: Spatial Autoencoder

**Implemented** (in `src/vae.py`, `app.py`):
- [x] `load_vae()` — loads SD-VAE (Stable Diffusion VAE) from HuggingFace; 4×16×16 continuous latent
- [x] `finetune_vae()` — MSE + LPIPS + adversarial loss with PatchGAN discriminator
- [x] `evaluate_vae()` — PSNR, SSIM, LPIPS, FID reconstruction metrics
- [x] `Discriminator` — PatchGAN for adversarial training
- [x] Modal function: `train_vae` (streams data, fine-tunes, saves to volume)
- [x] Modal function: `evaluate_vae` (evaluates base or fine-tuned VAE on held-out shard)
- [x] Modal function: `compute_latent_stats` (channel-wise mean/std of latents)

**Status**:
- [x] VAE switched from conv-net stub → real pre-trained SD-VAE (pre-trained weights, works out of the box)
- [x] `diffusers` + `transformers` added to all Modal images
- [x] Deprecated Modal API params fixed (`keep_warm` → `min_containers`, `container_idle_timeout` → `scaledown_window`, `web_endpoint` → `fastapi_endpoint`)
- [ ] First training run on Modal — **IN PROGRESS** (https://modal.com/apps/keypaa/main/ap-yT7RRIn97j8eyAHLhEo4KO)
- [ ] Latent stats computation — **IN PROGRESS**
- [ ] Evaluation of latent reconstruction quality (pending training completion)

---

## Phase 4: Generative Transformer (World Model)

**Implemented** (in `src/world_model.py`):
- [x] `LatentPatchEmbed` — 4×16×16 → 64 tokens of 2×2 patches
- [x] `FlashAttention` — causal self-attention with RoPE; falls back to PyTorch SDPA if flash-attn not installed
- [x] `ActionAdapter` — DWS-style per-block MLP (2 linear layers + SiLU + zero-init scale)
- [x] `WorldModelBlock` — attention + action adapter + MLP, RMSNorm
- [x] `MineWorldModel` — full AR transformer, forward pass with action embedding expansion
- [x] Forward + backward pass verified (5.4M params test config)

**Pending**:
- [ ] Diagonal Decoding optimization (post-training)
- [ ] KV cache implementation (post-training)
- [ ] Full 300M+ param config (pre-training)

---

## Phase 5: Training Engine

**Implemented** (in `src/trainer.py`):
- [x] `Trainer` class — bfloat16 mixed precision, gradient clipping, checkpointing
- [x] Progressive context expansion schedule (4→8→16→32)
- [x] AdamW optimizer with weight decay
- [x] Modal integration with H100:4 multi-GPU, retries, W&B logging

**Pending**:
- [ ] Multi-GPU (FSDP/DeepSpeed ZeRO) — H100:4 in config but not tested
- [ ] Gradient checkpointing optimization (every 2-4 blocks)
- [ ] Learning rate scheduling (cosine decay)
- [ ] Validation loop (held-out shards)
- [ ] Perceptual loss on decoded frames during training

---

## Phase 6: Inference & Client

**Implemented** (in `inference_app.py`):
- [x] `MinecraftInference` class — model loading, step, reset endpoints
- [x] Web endpoints: `/step` (predict next frame), `/reset` (seed frame)
- [x] Latent normalization with pre-computed stats
- [x] Context window padding/truncation

**Pending**:
- [ ] Deploy inference server on Modal (`modal deploy inference_app.py`)
- [ ] PyGame client (keyboard input → model → display loop)
- [ ] KV cache for fast inference
- [ ] Re-grounding every 32 frames (drift mitigation)
- [ ] Diagonal Decoding for parallel token prediction

---

## Project Structure

```
jepa/
├── app.py                  # Modal: train VAE, compute stats, train WM
├── inference_app.py        # Modal: deployable inference web server
├── image_modal.py          # Modal container images (3 variants)
├── minecraft_world_model_plan.md  # Full architectural plan
├── STATUS.md               # ← You are here
├── requirements.txt        # Local dev deps
├── configs/
│   ├── base.yaml           # Shared config
│   ├── stage[1-4]_*ctx.yaml  # Progressive training stages
├── src/
│   ├── action_tokenizer.py # JARVIS-VLA 51-token + µ-law camera + Lumine parser
│   ├── data.py             # Streaming datasets (HF parquet)
│   ├── vae.py              # VAE loader + fine-tuning loop
│   ├── world_model.py      # AR transformer + DWS action injection
│   └── trainer.py          # BF16 training loop + checkpointing
└── scripts/
    ├── test_phase2.py      # Smoke tests
    └── check_dataset.py    # Dataset analysis
```

---

## Pre-Flight Checks

Always run **both** of these before any training:

### 1. Canary (`run_canary`)
Proves the full pipeline spine end-to-end:

```bash
modal run app.py::run_canary
```

- SD-VAE encode/decode + reconstruction grid (saved to volume)
- Latent stat sanity check (per-channel mean/std, collapsed-space detection)
- Action token distribution over 2000 samples (no-op ratio, keyboard vs camera)
- World model forward pass (shape + finite loss)
- Checkpoint save/load round-trip

~5 min on A10G, ~$0.09.

### 2. Overfit Test (`test_overfit`)
Proves the model can actually learn (memorise one trajectory):

```bash
modal run app.py::test_overfit
```

- Trains a 7M-param mini model on 20 repeated frames
- Pass condition: loss drops to <20% of initial value
- If this fails, don't bother with real training — architecture/dataloader is broken

~2 min on A10G, ~$0.04.

### 3. Latent Statistics (`compute_latent_stats`)
Prerequisite for world model training (normalization stats):

```bash
modal run app.py::compute_latent_stats
```

- Computes channel-wise mean/std across 3 shards
- Now **validates** output: checks shape [4], no near-zero std, no implausible magnitudes

## How to Run

```bash
# 0a. Pre-flight: pipeline spine check
modal run app.py::run_canary                           # ~5 min, ~$0.09

# 0b. Pre-flight: verify model can actually learn
modal run app.py::test_overfit                          # ~2 min, ~$0.04

# 1. Pre-compute latents (speeds up world model later)
modal run app.py::precompute_latents

# 2. VAE fine-tuning (2 shards, ~100K frames, 1.5h on PRO 6000)
modal run app.py::train_vae

# 3. Latent statistics (prerequisite for WM training, 5 min)
modal run app.py::compute_latent_stats

# 4. World model training (10 shards, 4-ctx warmup, ~7-8h total PRO 6000)
modal run app.py::train_world_model --config configs/stage1_4ctx.yaml

# 5. Deploy inference server
modal deploy inference_app.py
```

---

## Decisions Log

| Date | Decision | Rationale |
|---|---|---|
| 2026-06-08 | Use Modal (H100) for all GPU compute | No local NVIDIA GPU; Modal has existing auth |
| 2026-06-08 | Stream data from HF, zero volume storage | Modal volume costs $0.10/GB/month; 2TB = $200/mo |
| 2026-06-08 | TESS VLA over MineStudio for data | MineStudio is gated; TESS VLA is public parquet |
| 2026-06-08 | On-the-fly VAE encode during training | VAE encode on H100 is ~1ms/frame; no pre-compute needed |
| 2026-06-08 | JARVIS-VLA 51-token action scheme | Faithful to VPT action space, compact (1 token/frame) |
| 2026-06-08 | DWS-style per-block action adapter | Architecture-agnostic, zero-init for stable training start |
| 2026-06-08 | SD-VAE over Oasis conv-net stub | Pre-trained weights produce meaningful latents; Oasis ViT-VAE conv stub was untrainable placeholder |
| 2026-06-08 | `diffusers` + `transformers` deps added | Required for SD-VAE loading |
| 2026-06-08 | Modal `::` syntax (not `:`) | Modal v1.4.2 requires double-colon function refs |
| 2026-06-08 | Removed `secrets` from Modal functions | Public HF dataset doesn't need auth; W&B handled gracefully in code |
| 2026-06-08 | **Resolution changed 128×128 → 256×256** | 128×128 blurred output; 256×256 is clean Minecraft at 4× tokens, still fits single GPU |
| 2026-06-08 | **GPU split: PRO 6000 for training, A10G for rest** | `train_vae`+`train_world_model` total cost is ~same or cheaper on PRO 6000 due to 3-4× speedup; short jobs stay on A10G |
| 2026-06-08 | **Pre-computed latents for WM training** | Removes VAE encode bottleneck during training, halves world model training time — no GPU upgrade needed |
| 2026-06-08 | **`latent_grid_size` param added to world model** | Removes hardcoded 16×16 latent grid; supports 128/256/512 resolutions via config |
| 2026-06-10 | **`run_canary` pre-flight check added** | Catches expensive failures early (VAE norm bugs, action parser issues, WM shape errors) before Modal GPU hours |
| 2026-06-10 | **`test_overfit` overfit test added** | Proves gradients flow and model can learn by memorising a single trajectory (~7M mini model, 1-2 min) |
| 2026-06-10 | **Trainer throughput logging** | Each epoch now logs steps/sec, tokens/sec, GPU memory (GiB) for cost monitoring |
| 2026-06-10 | **Latent stat validation** | `compute_latent_stats` now sanity-checks shape, near-zero std, and implausible magnitudes |
