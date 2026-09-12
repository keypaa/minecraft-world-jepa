---
library_name: pytorch
tags:
- minecraft
- world-model
- video-generation
- autoregressive
license: other
---

# Minecraft World Model — Checkpoints

Autoregressive video world model for Minecraft. A frozen Stable Diffusion VAE
compresses 256×256 frames to 4×32×32 latents; a decoder-only transformer
(338M params, RoPE, DWS-style per-block action adapters, SDPA flash attention)
predicts the next latent conditioned on discrete action tokens
(29 ids: 22 keyboard + 6 camera bins + 1 no-op).

- Data: `TESS-Computer/minecraft-vla-stage1` (streaming parquet, 303 shards)
- Code & training: https://github.com/keypaa/minecraft-world-jepa
- Precision: bfloat16. Normalization: disabled (checkpoints carry no mean/std).

## Layout

Each run directory holds a full resumable state (model + optimizer +
scheduler + step/epoch counters), so any file below restarts training
bit-identically with the matching config and `--resume`:

```
world-model-stage1-4ctx/
  latest.pt                  # freshest state (≤4 min old during training)
  best.pt                    # lowest-loss state
  epoch_0000_loss_0.266776.pt
  epoch_0001_loss_0.238446.pt
```

## Stage 1 (ctx 4, batch 32) — complete

| Epoch | Loss |
|---|---|
| 1 | 0.266776 |
| 2 | 0.238446 |

Measured on RTX PRO 6000: ~1.2s/step steady, ~64GB reserved VRAM.

## License note

Weights derive from public TESS-VLA gameplay data and an `sd-vae-ft-mse`
base. No license is asserted here — check the dataset and base-model terms
before commercial use.
