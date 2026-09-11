# Minecraft World JEPA — Deep Clean + Local RTX PRO 6000 (Design)

Date: 2026-09-11
Approach: A — Portable local-first cleanup (approved)
Scope: Full cleanup, Remove Modal, portable env (decide-for-me), freeze-default VAE (approved)

## 1. Repo layout + tooling

```
minecraft-world-jepa/
├── pyproject.toml          # uv, ruff, pytest, torch>=2.5, single CUDA 12.6 base
├── Makefile                # train, canary, overfit, eval, infer, bench
├── Dockerfile              # portable: same env bare-metal or molab container
├── configs/                # base.yaml + stage1_4ctx..stage4_32ctx (overrides only)
├── src/mw_jepa/            # action_tokenizer, data, vae, vae_experiments,
│                           # world_model, trainer, config, tensor_contracts, inference
├── scripts/                # train.py, canary.py, overfit.py, eval_vae.py, infer.py, bench.py
├── tests/                  # tokenizer, contracts, model fwd, data windowing
└── zenno/specs/            # this file
```

Deletes: `app.py` (994-line monolith), `image_modal.py`, `inference_app.py` wrapper,
`measure_real_step.py` / `measure_step_time.py` (fold into `scripts/bench.py`),
`.bat`/`.ps1` launchers, stale cost tables in `STATUS.md` / `minecraft_world_model_plan.md`
(replaced by single `README.md` with measured numbers only).

Tooling: `uv sync`, `ruff check + format`, `pytest`, `python scripts/*.py --config`.
No Hydra / Lightning — plain argparse + YAML, single-GPU BF16 + SDPA
(proven 58.81 GiB path on 338M / ctx4 / batch32, 1.25s/step steady-state).

## 2. Components (model + data + VAE)

- `world_model.py`: keep `MineWorldModel.forward(latents [B,T,4,32,32], actions [B,T]) -> next_latent`.
  Keep zero-init ActionAdapter fix + `_zero_init_action_adapters` re-apply after `_init_weights`.
  Keep SDPA-only `FlashAttention` (no manual matmul fallback). Add KV-cache hook stub only.
- `data.py`: keep `WorldModelStream` sliding-window logic. Drop `latent_dir` precompute
  branch from default path (9% speedup per debrief, not worth volume complexity;
  keep `--use-precomputed` hidden/deprecated flag). Fix infinite `while True` in
  `MinecraftFrameStream` (breaks epoch semantics). Wire `resume_shard` / `resume_row`
  through to `Trainer`.
- `vae.py` (freeze-default, approved): frozen SD-VAE is the default path
  (`encode_frames` / `decode_latents` with `expects_minus_one_to_one` + `scaling_factor` stay).
  Move `finetune_vae` + `Discriminator` + FID/Inception eval block into
  `src/mw_jepa/vae_experiments.py` — importable, not in default train path.
  Oasis conv stub becomes test fixture only.
  Rationale: SD-VAE works out-of-box (debrief); finetune adds LPIPS+GAN+2 optimizers
  for unmeasured gain; frozen gives deterministic latents and stable stats; reversible.

## 3. Data flow + training engine

Flow (default):
`TESS parquet streaming -> decode_jpeg 256 -> frozen SD-VAE encode -> normalize(mean/std)`
`-> WorldModelStream [T=ctx+1] -> MineWorldModel -> MSE on next latent`.

- `target_size` + `latent_grid_size` single-sourced from config (`grid = target // 8`).
  Removes hardcoded 32s (`app.py` canary/overfit paths).
- `trainer.py`: keep BF16 autocast + grad-clip 1.0 + NaN guard. Fix epoch semantics
  (remove infinite stream, add `steps_per_epoch` + `max_steps`). Add cosine LR +
  proper `resume_from` (shard/row + optimizer state). Keep throughput logging
  (st/s, tok/s, GiB). Batch sizes from measured baseline, not guessed comments.
- Configs: `base.yaml` single source (338M, 256 tokens/frame). Stage files override
  only `context / batch / lr / epochs`. Move 128/512 comment blocks to `docs/reference/`.
  Replace stale `$22 / 3GB / 20min` tables with measured-only table.

## 4. Error handling + testing + inference

- Contracts stay: `assert_frame_tensor` / `assert_latent_tensor` / `assert_action_tensor`
  on every boundary. Add `validate_frame` (NaN/black) at decode, fail-fast with
  `shard/row` in message.
- Tests (`pytest`, CPU where possible): Lumine parse + camera bins, contracts reject
  bad shapes, tiny `MineWorldModel` fwd finite loss, trajectory split + sliding window,
  VAE round-trip shape (mocked, no GPU).
- Integration gates: `scripts/canary.py` (~5 min spine: VAE recon + action dist +
  WM fwd + ckpt round-trip), `scripts/overfit.py` (3 pass conditions stay:
  final <20% initial, shuffled >10% worse than correct, beats copy-paste baseline).
- Inference: single `scripts/infer.py` (pygame) + `src/mw_jepa/inference.py`
  (context 16, re-ground every 32). No Modal `Cls` / `fastapi_endpoint`,
  no `inference_app.py` duplicate. Usage: `python scripts/infer.py --ckpt best.pt`.

## Decisions log

| Choice | Value |
|---|---|
| Scope | Full cleanup (all features kept, reorganized) |
| Modal | Remove completely (no fallback) |
| Env | Portable: bare-metal SSH or molab container via same Dockerfile/uv |
| VAE | Freeze-default, finetune isolated as experiment |
| Precompute latents | Off by default (deprecated flag) |
| Success | `make canary && make overfit` green local, `pytest` green, docs measured-only |

## Non-goals (YAGNI)

Hydra/Lightning/DeepSpeed, multi-GPU FSDP, VSA sparse attention, RLT pruning,
diagonal decoding full impl, KV-cache optimization, 128/512 resolution support,
W&B mandatory dependency (optional callback only).
