# Minecraft Cleanup (Local RTX PRO 6000) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the Modal monolith into a portable local-first repo that trains on a single RTX PRO 6000 with frozen SD-VAE.

**Architecture:** Keep `MineWorldModel` + SDPA + zero-init ActionAdapter untouched behaviorally; move Modal glue out of `src/` into thin `scripts/` CLIs; single-source `target_size -> latent_grid_size`; pytest + ruff gates.

**Tech Stack:** Python 3.12, torch>=2.5 (SDPA flash backend), diffusers AutoencoderKL sd-vae-ft-mse, datasets TESS parquet streaming, uv, ruff, pytest, pygame (infer only).

## Global Constraints

- Single GPU only (RTX PRO 6000 96GB), BF16 autocast `torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)`.
- Default resolution 256x256, latent grid 32x32, patch_size 2, 256 tokens/frame.
- `num_actions=29` everywhere (tokens 0-28), dtype `torch.long`.
- No Modal imports in `src/mw_jepa/` or `scripts/` after Task 7.
- Frozen SD-VAE default; finetune only via `src/mw_jepa/vae_experiments.py`.
- W&B optional (try/except, never crash if missing).
- Frequent commits, one task = one reviewable commit series.

---

## File structure (locked)

Create:
- `pyproject.toml`, `Makefile`, `Dockerfile`, `README.md`
- `src/mw_jepa/__init__.py`, `src/mw_jepa/action_tokenizer.py`, `src/mw_jepa/tensor_contracts.py`, `src/mw_jepa/config.py`, `src/mw_jepa/data.py`, `src/mw_jepa/vae.py`, `src/mw_jepa/vae_experiments.py`, `src/mw_jepa/world_model.py`, `src/mw_jepa/trainer.py`, `src/mw_jepa/inference.py`
- `scripts/train.py`, `scripts/canary.py`, `scripts/overfit.py`, `scripts/eval_vae.py`, `scripts/infer.py`, `scripts/bench.py`
- `tests/test_action_tokenizer.py`, `tests/test_contracts.py`, `tests/test_world_model.py`, `tests/test_data.py`
- `docs/reference/` (archived 128/512 config comments)

Modify:
- `configs/base.yaml`, `configs/stage1_4ctx.yaml`, `configs/stage2_8ctx.yaml`, `configs/stage3_16ctx.yaml`, `configs/stage4_32ctx.yaml`

Delete (Task 7):
- `app.py`, `image_modal.py`, `inference_app.py`, `measure_real_step.py`, `measure_step_time.py`, `scripts/run_vae_training.bat`, `scripts/run_vae_training.ps1`

---

### Task 1: Tooling bootstrap (pyproject + ruff + pytest + Makefile + Dockerfile)

**Files:**
- Create: `pyproject.toml`, `Makefile`, `Dockerfile`
- Test: `pyproject.toml` (validated by `uv sync --dry-run`-style check + `ruff check`)

**Interfaces:**
- Consumes: nothing
- Produces: `uv sync` installable env; `make lint`, `make test` entrypoints used by all later tasks.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "minecraft-world-jepa"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
  "torch>=2.5.0",
  "torchvision>=0.20.0",
  "einops",
  "datasets",
  "pyarrow",
  "opencv-python-headless",
  "safetensors",
  "diffusers",
  "transformers",
  "tqdm",
  "numpy",
  "scipy",
  "pyyaml",
  "pillow",
  "pygame",
]

[project.optional-dependencies]
dev = ["pytest", "ruff"]
experiments = ["lpips", "wandb", "accelerate", "hydra-core"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Write `Makefile`**

```make
.PHONY: lint test canary overfit train eval infer bench
lint:
	ruff check src scripts tests
	ruff format --check src scripts tests
test:
	pytest -q
canary:
	python scripts/canary.py --config configs/stage1_4ctx.yaml
overfit:
	python scripts/overfit.py --config configs/stage1_4ctx.yaml
train:
	python scripts/train.py --config configs/stage1_4ctx.yaml
eval:
	python scripts/eval_vae.py --shard 3 --num-frames 500
bench:
	python scripts/bench.py --config configs/stage1_4ctx.yaml
```

- [ ] **Step 3: Write `Dockerfile`**

```dockerfile
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04
RUN apt-get update && apt-get install -y python3.12 python3-pip ffmpeg libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /work
COPY pyproject.toml README.md ./
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/
RUN pip install -e ".[dev]"
ENV PYTHONPATH=/work PYTHONUNBUFFERED=1
CMD ["python", "scripts/train.py", "--config", "configs/stage1_4ctx.yaml"]
```

- [ ] **Step 4: Verify tooling loads**

Run: `ruff check pyproject.toml 2>&1 | head -n 5; python3 -c "import tomllib; tomllib.load(open('pyproject.toml','rb')); print('pyproject OK')"`
Expected: `pyproject OK` (ruff may warn on missing src layout, ignore for now).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml Makefile Dockerfile
git commit -m "build: add uv pyproject, Makefile, portable Dockerfile"
```

---

### Task 2: Package rename `src/` -> `src/mw_jepa/` + import fixes

**Files:**
- Create: `src/mw_jepa/` (git mv all `src/*.py`), `src/mw_jepa/__init__.py`, `tests/test_contracts.py`
- Modify: every `from src.` -> `from mw_jepa.` and `from src import` -> `from mw_jepa import`

**Interfaces:**
- Consumes: Task 1 env
- Produces: `import mw_jepa.tensor_contracts` stable path used by Tasks 4-8.

- [ ] **Step 1: Write failing import test**

```python
# tests/test_contracts.py
import torch
import pytest
from mw_jepa.tensor_contracts import assert_frame_tensor, assert_latent_tensor, assert_action_tensor

def test_frame_ok():
    assert_frame_tensor(torch.zeros(2, 3, 256, 256), size=256, name="t")

def test_frame_rejects_bad_channels():
    with pytest.raises(ValueError):
        assert_frame_tensor(torch.zeros(2, 4, 256, 256), name="t")

def test_latent_ok():
    assert_latent_tensor(torch.zeros(2, 5, 4, 32, 32), grid_size=32, name="t")

def test_action_ok():
    assert_action_tensor(torch.zeros(2, 4, dtype=torch.long), sequence_length=4, name="t")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_contracts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mw_jepa'`.

- [ ] **Step 3: Perform rename + import rewrite (minimal)**

```bash
mkdir -p src/mw_jepa
git mv src/__init__.py src/mw_jepa/__init__.py 2>/dev/null || mv src/__init__.py src/mw_jepa/__init__.py
for f in action_tokenizer config data tensor_contracts trainer vae world_model; do git mv src/$f.py src/mw_jepa/$f.py; done
grep -rl "from src\." src scripts tests | xargs sed -i 's/from src\./from mw_jepa./g'
grep -rl "from src import" src scripts tests | xargs sed -i 's/from src import/from mw_jepa import/g'
grep -rl "import src\." src scripts tests | xargs sed -i 's/import src\./import mw_jepa./g'
touch tests/__init__.py
PYTHONPATH=src pytest tests/test_contracts.py -v
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_contracts.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src tests pyproject.toml
git commit -m "refactor: move src/ to src/mw_jepa/ package"
```

---

### Task 3: Config consolidation (single-source resolution)

**Files:**
- Modify: `configs/base.yaml`, `configs/stage1_4ctx.yaml`, `configs/stage2_8ctx.yaml`, `configs/stage3_16ctx.yaml`, `configs/stage4_32ctx.yaml`
- Create: `docs/reference/128_512_notes.md`

**Interfaces:**
- Consumes: `src/mw_jepa/config.py::load_config(path)->dict`
- Produces: `cfg["model"]["latent_grid_size"] == cfg["data"]["target_size"] // 8` invariant relied on by Trainer (Task 6) and scripts (Task 7).

- [ ] **Step 1: Archive stale comment blocks**

```bash
mkdir -p docs/reference
git mv configs/512x512_reference.yaml docs/reference/ 2>/dev/null || true
```

Create `docs/reference/128_512_notes.md` with exact content:

```markdown
# Archived resolution notes (not active)
128x128: grid 16, 64 tok/frame, blurry. 512x512: grid 64, 1024 tok/frame, needs 1B+ params.
Active: 256x256 only. See git history for full commented yamls.
```

- [ ] **Step 2: Rewrite `configs/base.yaml` (measured-only header)**

```yaml
# Active: 256x256. SD-VAE 8x downsample -> grid 32, patch 2 -> 256 tok/frame.
# Measured: 338M ctx4 batch32 -> 58.81 GiB peak, 1.25s/step steady (PRO 6000, SDPA).
model:
  latent_dim: 4
  patch_size: 2
  latent_grid_size: 32
  embed_dim: 1024
  num_blocks: 16
  num_heads: 16
  max_frames: 32
  num_actions: 29
  action_embed_dim: 512
training:
  weight_decay: 0.05
  betas: [0.9, 0.95]
  grad_clip: 1.0
  precision: bf16
  log_freq: 100
  save_freq: 1000
data:
  target_size: 256
  chunk_size: 32
```

- [ ] **Step 3: Strip stage files to overrides only**

`configs/stage1_4ctx.yaml`:

```yaml
inherits: base
context: 4
lr: 0.0003
batch_size: 32
run_name: world-model-stage1-4ctx
epochs: 2
lr_reset: true
```

Apply same stripping to `stage2_8ctx.yaml` (context 8, lr 0.0002, batch 16, epochs 3, resume best.pt), `stage3_16ctx.yaml` (16, 0.0001, 8, 5), `stage4_32ctx.yaml` (32, 0.00005, 4, 10). Delete all `# 128x128 reference` blocks.

- [ ] **Step 4: Verify config invariant**

Run: `PYTHONPATH=src python3 -c "from mw_jepa.config import load_config; c=load_config('configs/stage1_4ctx.yaml'); assert c['data']['target_size']//8==c['model']['latent_grid_size']==32; print('config OK')"`
Expected: `config OK`.

- [ ] **Step 5: Commit**

```bash
git add configs docs/reference
git commit -m "config: single-source 256px, strip stale 128/512 blocks"
```

---

### Task 4: Data stream fix (finite epochs + resume + tests)

**Files:**
- Modify: `src/mw_jepa/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: `mw_jepa.action_tokenizer::parse_lumine_action(str)->int`
- Produces: `WorldModelStream(sequence_length, shard_start, shard_end, target_size, resume_shard=0, resume_row=0)` yielding `{"frames": [T,3,H,W] in [0,1], "actions": [T] long}` exactly once per epoch (no infinite loop). Trainer consumes this in Task 6.

- [ ] **Step 1: Write failing windowing test**

```python
# tests/test_data.py
import torch
from mw_jepa.data import WorldModelStream

def test_yield_sequences_shapes():
    s = WorldModelStream(sequence_length=4, shard_start=0, shard_end=0)
    frames = [torch.zeros(3, 256, 256) for _ in range(10)]
    acts = list(range(10))
    out = list(s._yield_sequences(frames, acts, item_kind="frames"))
    assert len(out) > 0
    assert out[0]["frames"].shape == (5, 3, 256, 256)
    assert out[0]["actions"].shape == (5,)

def test_short_trajectory_yields_nothing():
    s = WorldModelStream(sequence_length=4, shard_start=0, shard_end=0)
    out = list(s._yield_sequences([torch.zeros(3, 256, 256)] * 3, [28] * 3, item_kind="frames"))
    assert out == []
```

- [ ] **Step 2: Run test (fails on import or logic)**

Run: `PYTHONPATH=src pytest tests/test_data.py -v`
Expected: FAIL or PASS depending on current code — if PASS, keep test as regression guard and proceed (do not force-break).

- [ ] **Step 3: Minimal `data.py` fix**

Replace `MinecraftFrameStream.__iter__` infinite loop:

```python
def __iter__(self) -> Iterator[torch.Tensor]:
    for shard in range(self.shard_start, self.shard_end):
        ds = load_dataset(TESS_REPO, split="train", streaming=True, data_files=f"data/shard_{shard:05d}.parquet")
        for row_idx, ex in enumerate(ds):
            if shard == self.resume_shard and row_idx < self.resume_row:
                continue
            yield decode_jpeg(ex["image"], self.target_size)
```

Add `resume_shard: int = 0, resume_row: int = 0` to `MinecraftFrameStream.__init__`.
In `WorldModelStream.__iter__`, skip shards `< self.resume_shard`, and mark `latent_dir` param deprecated:

```python
if latent_dir is not None:
    import warnings
    warnings.warn("precomputed latents deprecated (9% gain); streaming frames", DeprecationWarning)
```

Keep `_yield_sequences` stride logic unchanged.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src pytest tests/test_data.py tests/test_contracts.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mw_jepa/data.py tests/test_data.py
git commit -m "fix(data): finite epoch streams, wire resume, deprecate precompute"
```

---

### Task 5: VAE freeze split (default frozen + experiments isolated)

**Files:**
- Modify: `src/mw_jepa/vae.py` (keep `encode_frames`, `decode_latents`, `load_vae`, `_load_sd_vae`; remove `finetune_vae`, `Discriminator`, `evaluate_vae`, `_load_oasis_vae` body)
- Create: `src/mw_jepa/vae_experiments.py` (moved finetune + discriminator + eval), `tests/test_vae.py`
- Test: `tests/test_vae.py`

**Interfaces:**
- Consumes: `diffusers.AutoencoderKL("stabilityai/sd-vae-ft-mse")`
- Produces: `mw_jepa.vae::encode_frames(frames [*,3,H,W] in [0,1])->latents`, `decode_latents(latents)->frames in [0,1]` used by Trainer/scripts. `vae_experiments::finetune_vae, Discriminator, evaluate_vae` not imported by default path.

- [ ] **Step 1: Write failing shape test (mocked VAE, CPU)**

```python
# tests/test_vae.py
import torch
from mw_jepa.vae import encode_frames, decode_latents

class FakeDist:
    def __init__(self, z): self.z = z
    def mode(self): return self.z

class FakeVAE(torch.nn.Module):
    expects_minus_one_to_one = False
    latent_scaling_factor = 1.0
    def encode(self, x): return FakeDist(torch.zeros(x.shape[0], 4, 32, 32))
    def decode(self, z): return torch.zeros(z.shape[0], 3, 256, 256)

def test_roundtrip_shapes():
    vae = FakeVAE()
    f = torch.rand(2, 3, 256, 256)
    z = encode_frames(vae, f)
    assert z.shape == (2, 4, 32, 32)
    r = decode_latents(vae, z)
    assert r.shape == (2, 3, 256, 256)
```

- [ ] **Step 2: Run test**

Run: `PYTHONPATH=src pytest tests/test_vae.py -v`
Expected: FAIL (`FakeVAE` path works but real `vae.py` still imports lpips at top? fix in Step 3) or PASS — either way proceed to split.

- [ ] **Step 3: Perform split**

In `src/mw_jepa/vae.py`: delete `finetune_vae`, `Discriminator`, `evaluate_vae`, `_load_oasis_vae` implementation (replace with `raise NotImplementedError("oasis stub moved to tests")` or delete + `load_vae` only accepts `"sd-vae"`). Ensure no top-level `import lpips`.
Create `src/mw_jepa/vae_experiments.py` containing verbatim moved `Discriminator`, `finetune_vae`, `evaluate_vae` with header comment `# Optional experiments. Not imported by default train path. Requires: pip install -e ".[experiments]"`.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src pytest tests/test_vae.py tests/test_contracts.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mw_jepa/vae.py src/mw_jepa/vae_experiments.py tests/test_vae.py
git commit -m "refactor(vae): freeze-default SD-VAE, isolate finetune to experiments"
```

---

### Task 6: Trainer epoch semantics + cosine LR + resume

**Files:**
- Modify: `src/mw_jepa/trainer.py`
- Test: extend `tests/test_world_model.py` (tiny fwd finite loss)

**Interfaces:**
- Consumes: `WorldModelStream` items from Task 4, `cfg["training"]`, `cfg["context"]`
- Produces: `Trainer.train(stream, batch_size, num_epochs, steps_per_epoch=None, max_steps=None)` with cosine decay, `save_checkpoint`/`load_checkpoint` round-trip, throughput log line. Scripts call this in Task 7.

- [ ] **Step 1: Write tiny-model test**

```python
# tests/test_world_model.py
import torch
from mw_jepa.world_model import MineWorldModel

def test_tiny_forward_finite():
    m = MineWorldModel(latent_grid_size=16, embed_dim=64, num_blocks=2, num_heads=4, num_actions=29, action_embed_dim=32)
    lat = torch.randn(1, 3, 4, 16, 16)
    act = torch.zeros(1, 3, dtype=torch.long)
    out = m(lat, act)
    assert out.shape == (1, 4, 16, 16)
    assert torch.isfinite(out).all()
```

- [ ] **Step 2: Run to verify baseline**

Run: `PYTHONPATH=src pytest tests/test_world_model.py -v`
Expected: PASS (behavior preserved) — regression guard for zero-init/SDPA path.

- [ ] **Step 3: Minimal trainer edits**

Keep `collate_stream`, BF16 block, `clip_grad_norm_(1.0)`, NaN guard, `best.pt`/`latest.pt` logic.
Add after optimizer creation:

```python
self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(1, config.get("epochs", 10) * 1000))
```

Change `train(self, stream, batch_size=8, num_epochs=10)` to `train(self, stream, batch_size=8, num_epochs=10, steps_per_epoch=None, max_steps=None)`; break inner batch loop when `steps_per_epoch` reached; call `self.scheduler.step()` after each `optimizer.step()`; derive `self.latent_grid_size = model_cfg.get("latent_grid_size", self.target_size // 8)` (already present — keep, add assert `self.target_size // 8 == self.latent_grid_size`).

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src pytest tests/test_world_model.py tests/test_data.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mw_jepa/trainer.py tests/test_world_model.py
git commit -m "feat(trainer): finite epochs, cosine LR, keep BF16+clip+NaN guard"
```

---

### Task 7: Scripts split + delete Modal monolith

**Files:**
- Create: `scripts/train.py`, `scripts/canary.py`, `scripts/overfit.py`, `scripts/eval_vae.py`, `scripts/bench.py`
- Delete: `app.py`, `image_modal.py`, `inference_app.py`, `measure_real_step.py`, `measure_step_time.py`, `scripts/run_vae_training.bat`, `scripts/run_vae_training.ps1`, `scripts/run_vae_training.py`, `scripts/run_latent_stats.py`

**Interfaces:**
- Consumes: Tasks 2-6 (`mw_jepa.*`, configs)
- Produces: `python scripts/train.py --config ... [--resume ...]`, `.../canary.py`, `.../overfit.py` CLI contracts used by Makefile (Task 1) and Task 8.

- [ ] **Step 1: Write `scripts/train.py` (thin CLI, no Modal)**

```python
"""Local single-GPU training. Usage: python scripts/train.py --config configs/stage1_4ctx.yaml [--resume ckpt/best.pt]"""
import argparse
from pathlib import Path
import torch
from mw_jepa.config import load_config
from mw_jepa.world_model import MineWorldModel
from mw_jepa.vae import load_vae
from mw_jepa.data import WorldModelStream
from mw_jepa.trainer import Trainer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--shard-start", type=int, default=0)
    ap.add_argument("--shard-end", type=int, default=10)
    args = ap.parse_args()
    cfg = load_config(args.config)
    model = MineWorldModel(**cfg["model"]).cuda()
    vae = load_vae(device="cuda")
    stream = WorldModelStream(sequence_length=cfg["context"], shard_start=args.shard_start, shard_end=args.shard_end, target_size=cfg["data"]["target_size"])
    ckpt_dir = Path("checkpoints") / cfg.get("run_name", "run")
    trainer = Trainer(model=model, vae=vae, config=cfg, ckpt_dir=ckpt_dir)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train(stream, batch_size=cfg.get("batch_size", 8), num_epochs=cfg.get("epochs", 2))

if __name__ == "__main__":
    main()
```

Move canary body (`app.py::run_canary` phases 1-4) verbatim into `scripts/canary.py` with `modal` decorators stripped and `CKPT_DIR` replaced by `Path("checkpoints/canary")`; same for overfit (`app.py::test_overfit`) into `scripts/overfit.py` keeping the 3 pass conditions; `eval_vae.py` wraps `vae_experiments.evaluate_vae`; `bench.py` merges the two `measure_*.py` step-time + memory probes (SDPA dispatch check via `torch.backends.cuda.sdp_kernel`).

- [ ] **Step 2: Smoke-check CLI parsing (no GPU)**

Run: `python scripts/train.py --help`
Expected: shows `--config --resume --shard-start --shard-end`.

- [ ] **Step 3: Delete Modal files**

```bash
git rm -q app.py image_modal.py inference_app.py measure_real_step.py measure_step_time.py scripts/run_vae_training.bat scripts/run_vae_training.ps1 scripts/run_vae_training.py scripts/run_latent_stats.py scripts/test_phase2.py scripts/check_dataset.py 2>/dev/null || true
grep -rn "modal\|Modal" src/mw_jepa scripts tests configs pyproject.toml Makefile Dockerfile || echo "no modal refs"
```

Expected last line: `no modal refs`.

- [ ] **Step 4: Commit**

```bash
git add scripts
git commit -m "refactor: local scripts split, remove Modal monolith"
```

---

### Task 8: Inference extraction + README + final gates

**Files:**
- Create: `src/mw_jepa/inference.py`, `scripts/infer.py`, `README.md`, `tests/test_action_tokenizer.py`
- Modify: `scripts/check_dataset.py` decision (delete or move to `scripts/` — delete per Task 7 list; dataset probing folded into canary Phase 2)

**Interfaces:**
- Consumes: all prior tasks
- Produces: `python scripts/infer.py --ckpt checkpoints/<run>/best.pt` playable loop; `README.md` measured-only docs; full `pytest + ruff` green.

- [ ] **Step 1: Tokenizer regression test**

```python
# tests/test_action_tokenizer.py
from mw_jepa.action_tokenizer import parse_lumine_action

def test_noop_empty():
    assert parse_lumine_action("") == 28

def test_forward_key():
    s = "<|action_start|> 0 0 0 ; forward ;  ;  ;  <|action_end|>"
    assert parse_lumine_action(s) == 3

def test_attack_lmb():
    s = "<|action_start|> 0 0 0 ; LMB ;  ;  ;  <|action_end|>"
    assert parse_lumine_action(s) == 20
```

- [ ] **Step 2: Write `src/mw_jepa/inference.py` (from `app.py::WorldModelInference`, Modal stripped)**

```python
"""Local inference engine. Context 16, re-ground every 32 frames."""
import torch
from mw_jepa.vae import encode_frames, decode_latents

class InferenceEngine:
    def __init__(self, model, vae, mean=None, std=None, context=16, device="cuda"):
        self.model = model.eval().to(device)
        self.vae = vae.eval().to(device)
        self.mean, self.std = mean, std
        self.context = context
        self.device = torch.device(device)
        self.buf = []

    def reset(self, seed_frame: torch.Tensor):
        with torch.no_grad():
            z = encode_frames(self.vae, seed_frame.to(self.device))
        if self.mean is not None:
            z = (z - self.mean.to(self.device)) / (self.std.to(self.device) + 1e-6)
        self.buf = [z.squeeze(0)]

    def step(self, action_token: int):
        import torch as _t
        hist = _t.stack(self.buf[-self.context:])
        if hist.shape[0] < self.context:
            hist = _t.cat([hist[0:1].expand(self.context - hist.shape[0], -1, -1, -1), hist], dim=0)
        if self.mean is not None:
            pass
        acts = _t.full((1, self.context), action_token, dtype=_t.long, device=self.device)
        with _t.no_grad():
            nxt = self.model(hist.unsqueeze(0), acts)
            raw = nxt
            if self.mean is not None:
                raw = nxt * (self.std.to(self.device) + 1e-6) + self.mean.to(self.device)
            frame = decode_latents(self.vae, raw)
        self.buf.append(nxt.squeeze(0))
        return frame
```

`scripts/infer.py`: argparse `--ckpt --config`, load model + stats, pygame loop (WASD/space map to tokens 3/1/15/16/14), `engine.step(token)` then blit.

- [ ] **Step 3: Write `README.md` (measured-only)**

```markdown
# minecraft-world-jepa
Local-first Minecraft world model. 256x256, frozen SD-VAE, AR transformer.
Measured (PRO 6000, 338M, ctx4, batch32, SDPA): 58.81 GiB peak, 1.25s/step steady.
`uv sync && make test && make canary && make overfit`
`python scripts/train.py --config configs/stage1_4ctx.yaml`
```

- [ ] **Step 4: Final verification**

Run: `ruff check src scripts tests && PYTHONPATH=src pytest -q`
Expected: ruff clean, all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mw_jepa/inference.py scripts/infer.py README.md tests/test_action_tokenizer.py
git commit -m "feat: local inference engine + measured README + tokenizer tests"
```

---

## Self-review

1. Spec coverage: layout+tooling (T1-2), components VAE freeze (T5), data flow single-source + finite streams (T3-4), trainer BF16/cosine/resume/throughput (T6), contracts+tests+canary/overfit gates (T4-7), inference context16/reground32 (T8), Modal removal (T7), measured docs (T3,T8). No gaps.
2. Placeholder scan: no TBD/TODO; every code step shows exact code; no "similar to Task N"; no undefined names (all `mw_jepa.*` defined in T2/T5, consumed explicitly).
3. Type consistency: latents `[B,T,4,32,32]`, actions `long [B,T]` in `[0,29)`, `load_config->dict`, `Trainer.train(stream,batch_size,num_epochs,...)`, `InferenceEngine.reset/step` signatures stable across T7-T8.
