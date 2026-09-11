# minecraft-world-jepa

Local-first Minecraft world model. 256x256, frozen SD-VAE, AR transformer.
Measured (PRO 6000, 338M, ctx4, batch32, SDPA): 58.81 GiB peak, 1.25s/step steady.
`uv sync && make test && make canary && make overfit`
`python scripts/train.py --config configs/stage1_4ctx.yaml`

## What this is

Autoregressive world model trained on the TESS Minecraft VLA dataset
(`TESS-Computer/minecraft-vla-stage1`). A frozen Stable Diffusion VAE
compresses 256x256 frames to 4x32x32 latents; a decoder-only transformer
predicts the next latent conditioned on discrete action tokens (29 ids:
22 keyboard + 6 camera yaw/pitch bins + 1 no-op/center).

## Quickstart

```bash
uv sync
make test     # CPU unit tests
make canary   # GPU pre-flight: VAE recon + action histogram + fwd/bwd + ckpt round-trip
make overfit  # GPU gradient check: single-trajectory memorisation must reach <20% of init loss
python scripts/train.py --config configs/stage1_4ctx.yaml
python scripts/infer.py --ckpt checkpoints/<run>/best.pt  # WASD+space, ESC quits
```

## Measured numbers (do not edit without re-measuring)

All from the 2026-06-10 canary on a PRO 6000 (96 GiB), 338,431,520-param
model, context 4, batch 32, `F.scaled_dot_product_attention` (native flash):

| Metric | Value |
|---|---|
| Steady-state step time | 1.25s (warmup 1.79s) |
| Peak GPU memory (transformer) | 58.81 GiB |
| VAE encode (batch 32) | 0.119s, 10.47 GiB |
| VAE recon MSE / PSNR (batch 4) | 0.0024 / 26.21 dB |

VAE latent stats (batch of 4, per channel): means
(-0.007, -0.569, 0.200, 0.458), stds (0.618, 0.585, 0.574, 0.530) —
no collapse, no implausible magnitudes.

Action distribution (2000 samples, shard 0): keyboard 75.5%, camera-only
0.5%, no-op 24.1%. ESC is 31.9% (dataset menu/pause artifact); camera
actions are near-absent, so camera control may not be learned.

Pre-SDPA baseline for reference: 89.83 GiB peak, 1.76s/step steady —
the manual-matmul attention fallback was removed; there is no reason to
go back.

## Layout

```text
src/mw_jepa/      world_model.py, vae.py, data.py, trainer.py,
                  action_tokenizer.py, inference.py, config.py, tensor_contracts.py
scripts/          train.py, canary.py, overfit.py, infer.py, bench.py, eval_vae.py
configs/          stage1_4ctx.yaml (+ base)
tests/            contracts, data, vae, world model, action tokenizer
```

Inference keeps a rolling context (default 16) and re-grounds on the
latest decoded frame every 32 steps to bound autoregressive drift.

Normalization is intentionally disabled by default: checkpoints carry no
mean/std, so inference runs unnormalized. See `vae_experiments.py` for
future latent-stats work.

## Gates

- `make test` — CPU-only unit tests, must pass.
- `make canary` — GPU pre-flight (recon plausibility, non-trivial action
  distribution, finite loss, checkpoint round-trip match).
- `make overfit` — must reach final loss < 20% of initial, beat the
  copy-paste baseline, and show shuffled actions hurting loss (proves
  action conditioning is alive).
