# Canary Results — 2026-06-10

**Run**: `modal run app.py::run_canary` on A10G (~5 min, ~$0.09)
**Commit**: current working state (pre-flight)

---

## Phase 1: VAE Reconstruction

| Metric | Value | Verdict |
|---|---|---|
| Latent shape | `(4, 4, 32, 32)` | Expected |
| Recon MSE | 0.0024 | Good |
| Recon PSNR | 26.21 dB | Good |

**Per-channel latent stats (batch of 4):**

| Channel | Mean | Std |
|---|---|---|
| 0 | -0.007 | 0.618 |
| 1 | -0.569 | 0.585 |
| 2 | 0.200 | 0.574 |
| 3 | 0.458 | 0.530 |

No collapse, no implausible magnitudes. Reconstruction grid saved to `/checkpoints/canary_reconstruction.png`.

## Phase 2: Action Token Distribution

**2000 samples from shard 0 of TESS-Computer/minecraft-vla-stage1.**

| Category | Count | % |
|---|---|---|
| Keyboard (0-21) | 1509 | 75.5% |
| Camera only (22-27) | 10 | 0.5% |
| No-op / center (28) | 481 | 24.1% |

**Top keyboard tokens:**

| Token | Action | Count | % |
|---|---|---|---|
| 0 | ESC | 637 | 31.9% |
| 20 | attack | 312 | 15.6% |
| 14 | jump | 187 | 9.3% |
| 21 | use | 129 | 6.5% |
| 1 | back | 76 | 3.8% |
| 4 | hotbar.1 | 54 | 2.7% |
| 5 | hotbar.2 | 48 | 2.4% |
| 2 | drop | 24 | 1.2% |
| 3 | forward | 21 | 1.1% |

**Notes:**
- Camera-only actions are extremely rare (0.5%) — TESS dataset is keyboard-heavy. The model may not learn camera control well unless this is addressed.
- ESC dominance (32%) is a known TESS dataset artifact (menus, pause screens).
- No-op ratio (24%) is manageable — model will not learn a screensaver.

## Phase 3: World Model Forward Pass

| Metric | Value |
|---|---|
| Model params | 338,431,520 |
| Input latents | `(2, 5, 4, 32, 32)` (B=2, T=5, C=4, H=32, W=32) |
| Predicted next | `(2, 4, 32, 32)` |
| MSE loss | 0.469 |
| NaN/Inf | Clean |

## Phase 4: Checkpoint Round-Trip

| Test | Result |
|---|---|
| Pre-save loss | 0.46903476 |
| Post-load loss | 0.46903476 |
| **Match** | **PASS** |

## Summary

**CANARY PASSED** — pipeline spine verified. Ready for training.

### Action items before real training
- [ ] Set `HF_TOKEN` env var on Modal to avoid unauthenticated rate limits (speeds up data loading)
- [ ] Consider TESS dataset camera-action sparsity (0.5%) may limit world model camera control
- [ ] Fix `WorldModelInference.__init__` deprecation warning (Modal will drop support for custom constructors)
