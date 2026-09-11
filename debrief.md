# Minecraft World Model — Build Debrief

*Covers both sessions: ActionAdapter dead-pathway diagnosis, SDPA/OOM fix, and cost-estimate reality check.*

---

## Bugs Found & Fixed

### 1. Dead Action Conditioning Pathway

**Bug**: The `ActionAdapter` used `scale = nn.Parameter(torch.zeros(1))` with `gate = self.scale.tanh()`. Since the scale was initialized to zero and remained there, `tanh(0) = 0`, so `gate * net(action_emb) = 0` — the adapter output was always zero regardless of action input. No gradient through the MLP either.

**Diagnosis trail**: The overfit test (mini 5.5M model, 20 epochs, 8 sequences) showed `shuffled_loss ≈ correct_loss` — identical loss whether actions matched or not, meaning actions had zero influence on predictions.

**Fix**: Three changes applied simultaneously in the first fix attempt (zero-init the last linear layer, remove scale param, verify gradient flow). The user caught that multiple variables changed at once and asked for clean isolation. Redone as a single change: zero-init `net[2].weight` and `net[2].bias` but keep `scale` as a residual gate that learns during training. Confirmed working: shuffled-correct delta of +0.152 on diverse-action trajectory.

**Lesson**: Zero-init gates (`scale = 0`) don't let gradient through the gated pathway. Zero-init the final linear layer instead — the adapter starts at zero output but gradient flows through the MLP from step 1. The scale parameter can remain as a learnable residual gate. Gated pathways don't need a gate *and* a zero-initialized subnetwork; pick one.

---

### 2. FP32 Attention OOM (89GB Peak)

**Bug**: The `FlashAttention.forward()` fallback path (when `flash-attn` pip package wasn't installed) materialized the full N×N attention score matrix in FP32:
```python
attn = (q.float() @ k.float().transpose(-2, -1)) * scale  # [B, H, N, N] in FP32
```
With batch=32, heads=16, N=1024 tokens: 32 × 16 × 1024 × 1024 × 4 bytes = **2 GB per layer**. Without gradient checkpointing, all 16 layers stored their attention matrices for backward = **32 GB** from attn scores alone.

Total peak: **89.83 GiB** on a 96 GB PRO 6000 — 94% utilization at the cheapest config.

**Diagnosis trail**: Measurement script showed 89.83 GiB peak for a 0.68 GB (BF16) model. The per-step memory didn't make sense given the parameter count. Checked image_modal.py line 36: `# flash-attn skipped: torch 2.12 + CUDA 12.6 incompatibility.` — the pip package was never installed, the fallback was always running.

**Fix**: Replaced the manual matmul fallback with `F.scaled_dot_product_attention()`:
```python
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```
PyTorch 2.5+ includes flash/memory-efficient backends natively — no external pip package needed. Verified dispatch: flash backend confirmed working on BF16 with `torch.backends.cuda.sdp_kernel(enable_math=False, enable_mem_efficient=False, enable_flash=True)`.

**Result**: Peak memory dropped to **58.81 GiB** (a 35% reduction). Step time improved from 1.76s to 1.25s.

**Lesson**: The `try: from flash_attn import ... except ImportError: manual matmul` pattern is dangerous — if flash-attn isn't installed (or can't be installed), you silently run the O(N²) path at full precision. With PyTorch 2.5+, there's no reason to use manual matmul. `F.scaled_dot_product_attention` dispatches to the best available backend automatically. Also: always measure GPU memory on your actual hardware with actual batch size — don't estimate from parameter count.

---

### 3. Config Path Resolution (Modal Container)

**Bug**: `train_world_model` defaulted to `config_path = "configs/stage1_4ctx.yaml"`, a relative path. Inside the Modal container, the working directory is not `/root/jepa/` where the project files live, so relative paths failed. Absolute paths passed from CLI (`/root/jepa/configs/...`) were converted by Git Bash on Windows to `C:/Program Files/Git/root/jepa/...`.

**Fix**: Resolve config path relative to `__file__` inside the function body:
```python
if config_path is None:
    config_path = str(Path(__file__).parent / "configs" / "stage1_4ctx.yaml")
```

**Lesson**: Never rely on working directory in Modal containers. Resolve paths relative to the source file.

---

### 4. Unicode Encoding on Windows (Modal CLI)

**Bug**: Modal's output includes the ✓ character (`✓`), which the Windows `charmap` codec can't encode. The Bash tool and local PowerShell terminal both crashed with `'charmap' codec can't encode character '✓'`.

**Workaround**: Running commands via `!` prefix (user types them directly in the prompt) bypasses the tool's encoding layer and uses the user's own terminal. For the measurement script, the user ran `modal run measure_step_time.py` manually in PowerShell and it worked fine — the checkmark rendered in their terminal output.

**Lesson**: Modal CLI uses Unicode glyphs for status indicators. On Windows, prefer running `modal` commands directly in the user's terminal rather than through tool invocations.

---

### 5. VAE Frame Validation

**Fix**: Added `validate_frame()` to reject NaN, all-black, or degenerate frames before encoding. Previously, a corrupted frame would silently propagate through the pipeline. All frames in the test trajectory were clean, but the guard is cheap insurance.

---

## Measurements (Confirmed)

### Real Step Time (338M model, PRO 6000 Blackwell, batch=32, ctx=4)
| Metric | Before SDPA fix | After SDPA fix |
|---|---|---|
| Step 1 (warmup) | 2.378s | 1.793s |
| Steady-state | 1.761s | 1.251s |
| Peak GPU mem | 89.83 GiB | 58.81 GiB |
| GPU util | 94% | 61% |

### SDPA Backend Availability
| Backend | FP32 | BF16 |
|---|---|---|
| Flash | not supported | OK |
| Mem-efficient | not supported | OK |
| Math | OK (deprecated path) | not used |

### VAE Encode (batch=32)
- Time: 0.119s/batch
- Memory: 10.47 GiB
- Fits alongside transformer: yes (58.81 + 10.47 = 69.3 GiB < 96 GiB)

---

## Cost-Estimate Autopsy

How four different numbers for the same thing differed by up to 450×:

| Source | Stage 1 cost | Total (4 stages) | Basis |
|---|---|---|---|
| plan.md (original) | ~$1 | ~$22 | 128×128, 64 tokens/frame, small data assumption |
| STATUS.md | ~$1 (implied) | ~$22 | Same plan numbers, not updated after 256×256 switch |
| My first estimate | ~$67 | ~$3,600 | Wrong 5s/step guess + hand-wavy batch math |
| Real measured | **~$16** | **TBD** | Actual 1.25s/step on PRO 6000 at full scale |

**Root causes of the drift:**

1. **The 128→256 resolution switch** changed tokens/frame from 64 to 256 (4×), but the plan's time/cost table was never updated. The table still shows "~3GB VRAM" and "~20 min" for stage 1 — actuals are 58.81 GiB and ~5.4h.

2. **The plan's dataset assumption** was never stated. The plan's "~20 min for stage 1" implies ~240 steps, which would be ~1 shard of data. The config specifies 10 shards — 10× more data.

3. **No one measured anything.** The 5s/step comment in the config was a guess. The original overfit test ran at 22 st/s on a 5.5M model, and nobody checked whether that predicted 338M model throughput. (It doesn't — attention scaling is not linear with parameter count.)

4. **Flash-attn was silently absent.** The fallback to manual matmul consumed 31 GiB more than necessary, and nobody checked whether flash-attn was installed or working.

**What should have happened:** Measure one step on the actual hardware before committing to cost estimates. The measurement script cost ~$0.10 to run and produced the real numbers.

---

## Scaling Estimates (for stage 2-4)

Total tokens/step is constant across all stages by design (32,768). With flash attention's O(N) memory, peak GPU memory should stay roughly flat across stages.

However, attention compute is O(N²) per sequence, so **step time** scales:
- Stage 2 (ctx=8, 2048 tokens, batch=16): ~2.5s/step → ~$98
- Stage 3 (ctx=16, 4096 tokens, batch=8): ~5s/step → ~$655  
- Stage 4 (ctx=32, 8192 tokens, batch=4): ~10s/step → ~$5,200

These are worst-case guesses until measured. Stage 2 specifically needs a remeasurement before launch, including verifying flash dispatch at 2048-token sequences (kernel selection can depend on sequence length).

---

## Configuration Decisions (Log)

| Decision | Rationale |
|---|---|
| SD-VAE (pre-trained) over training from scratch | Works out of the box, no fine-tuning needed for world model training |
| Pre-computed latents optional | 9% speedup, $1.35 saved on $16 stage 1 — not worth the complexity |
| 256×256 resolution | 128×128 is Game Boy blurry; 256×256 has recognizable Minecraft textures |
| SDPA over flash-attn pip package | PyTorch 2.5+ includes native flash backend; no version compatibility issues |
| Progressive context schedule (4→8→16→32) | Start cheap, validate, scale. If stage 1 fails, minimal loss |
