# Minecraft Video World Model — Complete Engineering Plan

*Synthesized from SOTA research: MineWorld (Microsoft, 2025), Matrix-Game (SkyworkAI, 2025), Oasis (Decart/Etched, 2024), DWS (AAAI 2026), VSA (2025), MA-GC (2024), VPT pipeline (OpenAI)*

---

## Resolution Decision

### Chosen: 256×256

**Latent geometry**: 4×32×32 (SD-VAE downsamples 8×) → patch_size=2 → **256 tokens/frame**

**Why not 128×128 (original plan)?**
128×128 works (64 tokens/frame, ~8h training), but output is Game Boy blurry. You can see blocks but not textures. At 256×256 the VAE decodes clean recognizable Minecraft — grass blocks look like grass blocks. The 4× token increase (~30h training) is a one-time cost worth paying for output you can actually look at.

**Why not 512×512?**
512×512 → 64×64 latent → 1024 tokens/frame. The attention cost grows O(n²) in tokens. A 300M model can't meaningfully model 1024-token frames — you'd need 1B+ params and multi-GPU (H100:2 minimum). Estimated cost: ~$500-1000 in GPU time. Massive overkill for Minecraft block world.

**GPU fit** (256×256):

| Stage | Tokens | Batch | VRAM (est.) | Time (PRO 6000) | Cost |
|---|---|---|---|---|---|
| 1 (ctx=4) | 1,024 | 32 | ~3GB | ~20 min | ~$1 |
| 2 (ctx=8) | 2,048 | 16 | ~4GB | ~1h | ~$3 |
| 3 (ctx=16) | 4,096 | 8 | ~5GB | ~2h | ~$6 |
| 4 (ctx=32) | 8,192 | 4 | ~8GB | ~4h | ~$12 |
| **Total** | | | | **~7-8h** | **~$22** |

All fit comfortably on PRO 6000 (96GB) or even A10G (22GB).

**Switching resolution is easy**:
- VAE fine-tune: NO redo (SD-VAE is resolution-agnostic)
- Pre-compute latents: NO redo (same function, just `target_size` changes)
- Latent stats: NO redo (same function, 2 min rerun)
- World model training: YES retrain from scratch
- Inference deployment: YES redeploy model

---

## Phase 1: Environment & Storage Prep

### Hardware Requirements
- **OS**: Ubuntu 24.04 (native or WSL2) — latest CUDA toolkits, custom attention kernels
- **Storage**: NVMe SSDs mandatory (HDD will starve GPU). Minimum 2TB dedicated for trimmed dataset
- **GPU**: 96GB VRAM target (H100 80GB / A100 80GB / 4090 24GB × 4)
- **RAM**: 256GB+ system RAM for data loading workers

### Software Stack
```bash
# Core
torch>=2.3, xformers>=0.0.27, flash-attn>=2.5
# Data
decord, opencv-python, pillow, safetensors
# Utils
einops, tqdm, hydra-core, wandb
# Optional (for scaling)
deepspeed, accelerate, grain (if JAX)
```

---

## Phase 2: Data Factory (VPT Pipeline)

### 2.1 Don't Rebuild — Use Pre-Converted Data
The MineStudio framework has already converted 15M+ VPT frames to LMDB:
```python
# HuggingFace datasets (ready to stream)
from datasets import load_dataset
dataset = load_dataset("CraftJarvis/minestudio-data-6xx")  # + 7xx, 8xx, 9xx, 10xx
```

Each shard: LMDB with 32-frame chunks, JPEG-compressed frames + factored actions.

### 2.2 If Building From Scratch (Reference Implementation)
```python
# Based on OpenAI data_loader.py + MineStudio ConvertManager
import decord, cv2, json, lmdb, torch
from pathlib import Path

def process_vpt_segment(mp4_path: Path, jsonl_path: Path, lmdb_env: lmdb.Environment, chunk_size=32):
    vr = decord.VideoReader(str(mp4_path), num_threads=4)
    frames = vr.get_batch(range(len(vr))).asnumpy()  # [T, H, W, 3], RGB
    
    with open(jsonl_path) as f:
        json_data = [json.loads(line) for line in f]
    
    # Convert actions using VPT's exact mapping
    actions = [json_action_to_env_action(step) for step in json_data]
    
    # Write to LMDB in chunks
    with lmdb_env.begin(write=True) as txn:
        for i in range(0, len(frames), chunk_size):
            chunk_frames = frames[i:i+chunk_size]
            chunk_actions = actions[i:i+chunk_size]
            
            # JPEG compress frames
            encoded = [cv2.imencode('.jpg', f[:,:,::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes() 
                      for f in chunk_frames]
            
            key = f"{mp4_path.stem}_{i//chunk_size:06d}".encode()
            value = msgpack.packb({"frames": encoded, "actions": chunk_actions})
            txn.put(key, value)

# Action conversion — exact from OpenAI run_inverse_dynamics_model.py
def json_action_to_env_action(step_data):
    env_action = NOOP_ACTION.copy()
    is_null = True
    
    # Keyboard
    for key in step_data["keyboard"]["keys"]:
        if key in KEYBOARD_BUTTON_MAPPING:
            env_action[KEYBOARD_BUTTON_MAPPING[key]] = 1
            is_null = False
    
    # Mouse camera (μ-law discretization later)
    camera_action = np.array([step_data["mouse"]["dy"], step_data["mouse"]["dx"]], dtype=np.float32)
    camera_action *= CAMERA_SCALER  # 360/2400
    
    # Mouse buttons
    if step_data["mouse"]["buttons"][0]:  # attack
        env_action["attack"] = 1; is_null = False
    if step_data["mouse"]["buttons"][1]:  # use
        env_action["use"] = 1; is_null = False
    if step_data["mouse"]["buttons"][2]:  # pickItem
        env_action["pickItem"] = 1; is_null = False
    
    # Hotbar tracking (scroll wheel workaround)
    current_hotbar = step_data["hotbar"]
    if current_hotbar != last_hotbar:
        env_action[f"hotbar.{current_hotbar + 1}"] = 1
        last_hotbar = current_hotbar
    
    env_action["camera"] = camera_action
    return env_action, is_null
```

### 2.3 Action Tokenization — JARVIS-VLA 51-Token Scheme
```python
# 22 keyboard buttons (binary) + 29 camera bins (μ-law, 21 bins per axis + center)
ACTION_TOKENS = {
    # Keyboard (22): one-hot indices 0-21
    "keyboard": {
        "ESC": 0, "back": 1, "drop": 2, "forward": 3,
        "hotbar.1": 4, "hotbar.2": 5, "hotbar.3": 6, "hotbar.4": 7,
        "hotbar.5": 8, "hotbar.6": 9, "hotbar.7": 10, "hotbar.8": 11,
        "hotbar.9": 12, "inventory": 13, "jump": 14, "left": 15,
        "right": 16, "sneak": 17, "sprint": 18, "swapHands": 19,
        "attack": 20, "use": 21,
    },
    # Camera (29): pitch 0-13, yaw 14-27, center=28
    "camera": {"pitch_bins": 21, "yaw_bins": 21, "center_token": 28}
}

# μ-law quantization from OpenAI lib/actions.py
class CameraQuantizer:
    def __init__(self, binsize=2, maxval=10, mu=10):
        self.binsize = binsize
        self.maxval = maxval
        self.mu = mu
    
    def discretize(self, xy):  # xy: [pitch, yaw] in degrees
        xy = np.clip(xy, -self.maxval, self.maxval)
        xy = xy / self.maxval
        v_encode = np.sign(xy) * (np.log(1 + self.mu * np.abs(xy)) / np.log(1 + self.mu))
        v_encode *= self.maxval
        return np.round((v_encode + self.maxval) / self.binsize).astype(np.int64)  # 0-20

---

## Phase 3: Spatial Autoencoder (Visual Compression)

### 3.1 Why Compress?
Predicting 128×128×3 = 49,152 values per frame is intractable. Compress to 4×16×16 = 1,024 latent values — **48× reduction**.

### 3.2 Don't Train From Scratch — Fine-Tune a Pre-Trained Tokenizer

| Tokenizer | Architecture | Latent Shape | Quality | Source |
|---|---|---|---|---|
| **BSQ-ViT** (CVPR 2024) | ViT encoder/decoder + binary spherical quantization | 16×16, 1024 codebook | rFVD 4.10 (SOTA), 2.4× faster than SDXL-VAE | [`bsq-vit`](https://github.com/google-research/big_vision) |
| **ViT-VQGAN** (Yu et al., 2022) | ViT encoder/decoder + standard VQ | 16×16, 8192 codebook | Used in Oasis | [`oasis-500m`](https://huggingface.co/Etched/oasis-500m) |
| **VQGAN-LC** (2024) | CNN-based + 100K frozen CLIP codebook | 16×16, 100000 codebook | rFID 2.37 (best reconstruction) | [`vqgan-lc`](https://github.com/CompVis/taming-transformers) |
| **VideoVAE** (Open-Sora, 2024) | 3D causal CNN VAE | 4×16×16, continuous latent | Temporal coherence, 8× spatial + 4× temporal | [`Open-Sora`](https://github.com/hpcaitech/Open-Sora) |

**Recommendation**: Fine-tune **BSQ-ViT** on 50K VPT frames (1-2 epochs). If unavailable, fine-tune **Oasis ViT-VAE** (500M weights available on HF).

### 3.3 Fine-Tuning Script

```python
import torch
import torch.nn.functional as F
from einops import rearrange

def fine_tune_tokenizer(vae, dataloader, num_steps=50000):
    """Fine-tune a pre-trained VAE on Minecraft frames."""
    
    # Loss weights (VQGAN-style)
    lambda_mse = 1.0
    lambda_perceptual = 0.5   # LPIPS
    lambda_adversarial = 0.1  # Discriminator
    lambda_commitment = 0.25  # VQ commitment loss
    
    opt = torch.optim.AdamW(vae.parameters(), lr=1e-5, weight_decay=0.01)
    discriminator = Discriminator()
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=2e-5, betas=(0.5, 0.999))
    
    for step in range(num_steps):
        batch = next(dataloader)  # [B, 3, 128, 128]
        
        # Forward through VAE
        if hasattr(vae, 'encode'):
            # ViT-VQGAN / BSQ-ViT
            z_q, _, commit_loss = vae.encode(batch)
            recon = vae.decode(z_q)
        else:
            # Continuous VAE (SD-style)
            posterior = vae.encode(batch)
            z = posterior.sample()
            recon = vae.decode(z)
            kl_loss = posterior.kl().mean()
        
        # MSE loss (on latents for continuous VAE, on pixels for VQ)
        if hasattr(vae, 'encode'):
            mse_loss = F.mse_loss(recon, batch)
        else:
            # Encode target to latent space
            with torch.no_grad():
                target_z = vae.encode(batch).mode()
            mse_loss = F.mse_loss(z, target_z)
        
        # LPIPS perceptual loss
        p_loss = lpips_loss(recon, batch).mean()
        
        # Adversarial loss
        real_logits = discriminator(batch)
        fake_logits = discriminator(recon.detach())
        g_loss = torch.mean((fake_logits - 1) ** 2)  # hinge loss
        d_loss = (torch.mean((real_logits - 1) ** 2) + torch.mean(fake_logits ** 2)) / 2
        
        # Total loss
        loss = (lambda_mse * mse_loss +
                lambda_perceptual * p_loss +
                lambda_adversarial * g_loss +
                lambda_commitment * commit_loss if hasattr(vae, 'encode') else 0)
        
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        
        # Train discriminator
        opt_d.zero_grad(set_to_none=True)
        d_loss.backward()
        opt_d.step()

### 3.4 Encoding Pipeline (for Training Data)
batch = [...]  # [B, T, 3, 128, 128]
flat = rearrange(batch, 'b t c h w -> (b t) c h w')

with torch.no_grad():
    z = vae.encode(flat).mode() if hasattr(vae, 'encode') else vae.encode(flat).sample()
    # z shape: [(b*t), 4, 16, 16]
    
video_latents = rearrange(z, '(b t) c h w -> b t c h w', b=batch_size)
# Now: [B, T, 4, 16, 16] — 1,024 values per frame instead of 49,152
```

### 3.5 Latent Space Statistics (Pre-Compute for Normalization)
```python
# Run once over dataset
all_means = []
all_stds = []
for batch in dataloader:
    with torch.no_grad():
        z = vae.encode(batch.cuda()).mode()
    all_means.append(z.mean(dim=(0, 2, 3)))
    all_stds.append(z.std(dim=(0, 2, 3)))

channel_mean = torch.stack(all_means).mean(dim=0)  # [4]
channel_std = torch.stack(all_stds).mean(dim=0)     # [4]
torch.save({"mean": channel_mean, "std": channel_std}, "latent_stats.pt")
# During training: latents = (latents - channel_mean) / (channel_std + 1e-6)
```

---

## Phase 4: The Generative Transformer (World Model)

### 4.1 Architecture Overview
The brain of the system. Predicts the next latent frame given past latents + current action.

**Key Design Decisions**:
| Decision | Choice | Rationale |
|---|---|---|
| **Backbone** | Causal autoregressive Transformer (MineWorld-style) | Interactive inference, well-understood training |
| **Tokenization** | 2×2 patch tokens (256 tokens/frame) | 4× reduction vs 1,024 flat tokens |
| **Action injection** | Per-block MLP adapter (DWS paper — 2 linear layers) | Architecture-agnostic, minimal params |
| **Position encoding** | RoPE (Rotary Position Embedding) | Better length generalization than learned |
| **Decoding** | Diagonal Decoding (parallel token groups) | 3× inference speedup |

### 4.2 Patch Embedding (Spatial Tokenization)

```python
class LatentPatchEmbed(nn.Module):
    """Tokenize [4, 16, 16] latent into 256 patch tokens with RoPE positions."""
    
    def __init__(self, latent_dim=4, patch_size=2, embed_dim=1024):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (16 // patch_size) ** 2  # 64 patches of size 2
        self.proj = nn.Conv2d(latent_dim, embed_dim, kernel_size=patch_size, stride=patch_size)
        # RoPE — applied later in attention
        
    def forward(self, x):
        # x: [B, T, 4, 16, 16]
        B, T = x.shape[:2]
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.proj(x)  # [(B*T), embed_dim, 8, 8]
        x = rearrange(x, 'bt d h w -> bt (h w) d')  # [(B*T), 64, d]
        x = rearrange(x, '(b t) n d -> b t n d', b=B, t=T)
        return x  # [B, T, 64, embed_dim]
```

### 4.3 Position Encoding (RoPE for Spatial + Temporal)

```python
class VideoRoPE(nn.Module):
    """Rotary Position Embedding for 2D spatial grid + temporal index."""
    
    def __init__(self, dim, max_spatial=32, max_temporal=1024):
        super().__init__()
        self.dim = dim
        
        # Precompute spatial frequencies (per patch grid position)
        inv_freq_spatial = 1.0 / (10000 ** (torch.arange(0, dim, 4).float() / dim))
        self.register_buffer("inv_freq_spatial", inv_freq_spatial)
        
        # Precompute temporal frequencies
        inv_freq_temporal = 1.0 / (10000 ** (torch.arange(0, dim, 4).float() / dim))
        self.register_buffer("inv_freq_temporal", inv_freq_temporal)
        
    def forward(self, x, patch_positions, temporal_positions):
        # patch_positions: [n_patches, 2] — (row, col) in [0, 8)
        # temporal_positions: [T] — frame indices
        
        sp = self.inv_freq_spatial
        tp = self.inv_freq_temporal
        
        # Spatial RoPE
        spat_rope = torch.cat([
            patch_positions[..., 0:1] * sp[None, None, :],  # row
            patch_positions[..., 1:2] * sp[None, None, :],  # col
        ], dim=-1)
        
        # Temporal RoPE
        temp_rope = temporal_positions[None, :, None, None] * tp[None, None, None, :]
        
        return spat_rope, temp_rope
```

### 4.4 Action Injection (DWS-Style Per-Block MLP Adapter)

```python
class ActionAdapter(nn.Module):
    """
    Frame-level action conditioning via 2-linear-layer MLP per transformer block.
    From DWS paper (AAAI 2026) — validated as architecture-agnostic.
    """
    def __init__(self, embed_dim=1024, action_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.scale = nn.Parameter(torch.zeros(1))  # Initialize to zero (no effect at start)
    
    def forward(self, x, action_emb):
        # x: [B, n_patches, embed_dim] — frame tokens
        # action_emb: [B, action_dim] — embedded action
        gate = self.scale.tanh()
        return x + gate * self.net(action_emb).unsqueeze(1)  # Broadcast across patches
```

### 4.5 Full Transformer Block

```python
class WorldModelBlock(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=16, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.RMSNorm(embed_dim)
        self.attn = FlashAttention(embed_dim, num_heads, causal=True, use_rope=True)
        self.norm2 = nn.RMSNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
            nn.Dropout(dropout),
        )
        self.action_adapter = ActionAdapter(embed_dim)
        
    def forward(self, x, action_emb, rope_params):
        # Self-attention with RoPE + causal masking
        x = x + self.attn(self.norm1(x), rope_params=rope_params)
        
        # Action injection (per frame)
        x = self.action_adapter(x, action_emb)
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x
```

### 4.6 Full World Model Forward Pass

```python
class MineWorldModel(nn.Module):
    """
    Autoregressive transformer that predicts the next latent frame.
    Input:  latents [B, T, 4, 16, 16], actions [B, T]
    Output: next_latent [B, 4, 16, 16]
    """
    def __init__(self, 
                 latent_dim=4, patch_size=2, embed_dim=1024, 
                 num_blocks=16, num_heads=16, max_frames=64,
                 num_actions=51, action_embed_dim=512):
        super().__init__()
        
        self.patch_embed = LatentPatchEmbed(latent_dim, patch_size, embed_dim)
        self.action_embedder = nn.Embedding(num_actions, action_embed_dim)
        self.action_proj = nn.Linear(action_embed_dim, embed_dim)  # Project to embed_dim
        
        self.blocks = nn.ModuleList([
            WorldModelBlock(embed_dim, num_heads) for _ in range(num_blocks)
        ])
        
        self.norm = nn.RMSNorm(embed_dim)
        
        # Output head: [embed_dim] → [latent_dim * patch_size * patch_size]
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, latent_dim * patch_size * patch_size),
        )
        
        # Register patch grid positions (for RoPE)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(8), torch.arange(8), indexing='ij'
        )
        self.register_buffer("patch_positions", 
                            torch.stack([grid_y, grid_x], dim=-1).reshape(-1, 2))  # [64, 2]
        
    def forward(self, latent_seq, action_seq, kv_cache=None):
        # latent_seq: [B, T, 4, 16, 16]
        # action_seq: [B, T]
        
        B, T = latent_seq.shape[:2]
        
        # Patch embed: [B, T, 64, embed_dim]
        x = self.patch_embed(latent_seq)
        
        # Embed actions and project to same space
        action_emb = self.action_embedder(action_seq)  # [B, T, action_embed_dim]
        action_emb = self.action_proj(action_emb)       # [B, T, embed_dim]
        
        # Reshape: [B, T, 64, embed_dim] → [B, T*64, embed_dim]
        x = rearrange(x, 'b t n d -> b (t n) d')
        
        # RoPE: temporal + spatial positions for each token
        temporal_pos = torch.arange(T, device=x.device).repeat_interleave(64)
        spatial_pos = self.patch_positions.repeat(T, 1)
        
        rope_params = self.compute_rope(x, spatial_pos, temporal_pos)
        
        # Through transformer blocks
        for block in self.blocks:
            x = block(x, action_emb[:, :, None, :].expand(-1, -1, 64, -1)
                      .reshape(B, T * 64, -1), rope_params)
        
        x = self.norm(x)
        
        # Get last T patches (for next frame prediction)
        last_frame_tokens = x[:, -64:, :]  # [B, 64, embed_dim]
        
        # Decode to latent
        patches = self.output_head(last_frame_tokens)  # [B, 64, 4*2*2=16]
        patches = patches.view(B, 8, 8, 4, 2, 2)
        next_latent = rearrange(patches, 'b h w c ph pw -> b c (h ph) (w pw)')
        # [B, 4, 16, 16]
        
        return next_latent
    
    def compute_rope(self, x, spatial_pos, temporal_pos):
        return {"spatial": spatial_pos, "temporal": temporal_pos}
```

### 4.7 Diagonal Decoding (Parallel Inference Optimization)

```python
@torch.no_grad()
def diagonal_decode(model, seed_latent, actions, num_frames=64):
    """
    MineWorld-style Diagonal Decoding.
    Predicts spatially-adjacent token groups in parallel for 3× speedup.
    """
    past_latents = [seed_latent]
    kv_cache = None
    
    for t in range(num_frames):
        # Standard AR step for current frame
        next_latent = model(
            torch.stack(past_latents[-5:], dim=1),  # context window of 5
            actions[t:t+1],
            kv_cache
        )
        
        # Diagonal Decoding: refine patches within the frame
        # Split 64 patches into 4 groups of 16 spatially-adjacent patches
        refined = next_latent.clone()
        for group_idx in range(4):
            # Group patches (e.g., top-left, top-right, bottom-left, bottom-right quadrants)
            mask = get_group_mask(group_idx)  # 16 patches selected
            refined = refine_patches(model, refined, mask, actions[t])
        
        past_latents.append(refined)
    
    return torch.stack(past_latents[1:])  # [num_frames, 4, 16, 16]
```

---

## Phase 5: Training Engine (VRAM & Optimization)

### 5.1 Memory Optimization Stack (Priority Order for 96GB VRAM)

| Priority | Technique | Memory Savings | Throughput Impact | Complexity |
|---|---|---|---|---|
| 1 | **BF16 mixed precision** | 2× activation memory | +40% speed | Trivial (1 line) |
| 2 | **FlashAttention-2** | Eliminates O(N²) attention | 1.5-2× speed | Built into PyTorch 2.0+ |
| 3 | **Selective gradient checkpointing** (every 2-4 blocks, skip norms) | 40-70% on activations | -10-20% | Low (policy function) |
| 4 | **Multi-Axis GC (MA-GC)** | O(S) vs O(L·S) memory | -25% | Medium (checkpoint along time + layer axes) |
| 5 | **ZeRO-3 + CPU optimizer offload** | 4-8× on model states | -15-20% | Low (config change via DeepSpeed) |
| 6 | **VSA sparse attention** (87.5% sparsity) | 2.53× FLOPs reduction | 1.7× speed | High (custom kernel) |
| 7 | **Activation offloading** (GPU → CPU/NVMe) | 30-47% peak reduction | <5% (async) | Medium (trl or DeepSpeed) |
| 8 | **RLT token pruning** (static backgrounds) | 35% token reduction | 30% speed | Medium (heuristic) |

**What 96GB Can Hold**:
- Model: ~7B params → 14GB (BF16) + 56GB (Adam states) + 14GB (gradients) = **84GB**
- Remaining: ~12GB for activations (with checkpointing, enough for 16-frame context)
- If 7B doesn't fit: drop to 3B params → 6GB + 24GB + 6GB = **36GB** → 60GB free for activations

### 5.2 Gradient Checkpointing Policy

```python
import torch.utils.checkpoint as checkpoint

def ckpt_policy_fn(ctx, op, *args, **kwargs):
    """
    Selective activation checkpointing policy.
    Always save: matmuls (expensive to recompute)
    Prefer recompute: pointwise operations (LayerNorm, activations, residuals)
    """
    if op in [torch.ops.aten.mm, torch.ops.aten.bmm, torch.ops.aten.addmm]:
        return checkpoint.CheckpointPolicy.MUST_SAVE
    if op in [torch.ops.aten._softmax, torch.ops.aten._softmax_backward_data]:
        return checkpoint.CheckpointPolicy.MUST_SAVE
    return checkpoint.CheckpointPolicy.PREFER_RECOMPUTE

# Alternative: use PyTorch 2.5+ budget-based approach (simpler)
torch._dynamo.config.activation_memory_budget = 0.5  # 50% of default
```

### 5.3 MA-GC (Multi-Axis Gradient Checkpointing) for Video

```python
class MAGCCheckpointFunction(torch.autograd.Function):
    """
    Checkpoints along both layer AND time axes.
    Memory: O(S) instead of O(L·S) for L layers × S sequence length.
    """
    @staticmethod
    def forward(ctx, run_function, *args):
        ctx.run_function = run_function
        ctx.input_tensors = [x.detach() for x in args]
        ctx.save_for_backward(*ctx.input_tensors)
        with torch.no_grad():
            return run_function(*args)
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        inputs = ctx.saved_tensors
        with torch.enable_grad():
            outputs = ctx.run_function(*inputs)
        torch.autograd.backward(outputs, grad_outputs)
        return (None,) + tuple(inp.grad for inp in inputs)

# In training loop: checkpoint every 4th layer AND every 8th time step
def apply_magc(model, latent_seq, action_seq, checkpoint_freq_layer=4, checkpoint_freq_time=8):
    h = latent_seq
    for i, block in enumerate(model.blocks):
        if i % checkpoint_freq_layer == 0:
            h = checkpoint.checkpoint(block, h, action_seq, rope_params, use_reentrant=False)
        else:
            h = block(h, action_seq, rope_params)
    return h
```

### 5.4 Progressive Training Schedule

```python
def get_training_schedule():
    """
    Start with small context, double every N epochs.
    From Matrix-Game, Vchitect-2.0 proven approach.
    """
    return [
        # Stage 1: 4-frame context (warmup, fast iteration)
        {"context": 4,  "epochs": 2,  "lr": 3e-4,  "batch_size": 64,  "lr_reset": True},
        # Stage 2: 8-frame context
        {"context": 8,  "epochs": 3,  "lr": 2e-4,  "batch_size": 32,  "lr_reset": True},
        # Stage 3: 16-frame context
        {"context": 16, "epochs": 5,  "lr": 1e-4,  "batch_size": 16,  "lr_reset": True},
        # Stage 4: 32-frame context (final training)
        {"context": 32, "epochs": 10, "lr": 5e-5,  "batch_size": 8,   "lr_reset": True},
    ]
```

### 5.5 Core Training Loop

```python
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.optim import AdamW

def train_epoch(model, dataloader, vae, optimizer, stage, scaler=None):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        # batch: {"latents": [B, T, 4, 16, 16], "actions": [B, T]}
        latents = batch["latents"].cuda()
        actions = batch["actions"].cuda()
        
        # Normalize latents (channel-wise)
        latents = (latents - channel_mean.cuda()) / (channel_std.cuda() + 1e-6)
        
        # Input: first T-1 frames + actions
        input_latents = latents[:, :-1, :, :, :]        # [B, T-1, 4, 16, 16]
        target_latent = latents[:, -1, :, :, :]          # [B, 4, 16, 16]
        input_actions = actions[:, :-1]                   # [B, T-1]
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            # Predict next latent
            pred_latent = model(input_latents, input_actions)
            
            # Primary loss: MSE on latents
            mse_loss = F.mse_loss(pred_latent, target_latent)
            
            # Secondary loss: decode and compute LPIPS (every 4th step)
            if step % 4 == 0:
                with torch.no_grad():
                    target_frame = vae.decode(target_latent)
                pred_frame = vae.decode(pred_latent)
                perceptual_loss = lpips_loss(pred_frame, target_frame).mean()
            else:
                perceptual_loss = 0.0
            
            loss = mse_loss + 0.5 * perceptual_loss
        
        # Backward with optional gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

# Launch training
model = MineWorldModel()
optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.05, betas=(0.9, 0.95))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000)

for stage in get_training_schedule():
    print(f"Training stage: context={stage['context']}, lr={stage['lr']}")
    
    # Update model context window
    model.max_frames = stage["context"]
    
    # LR reset for new stage
    if stage["lr_reset"]:
        for g in optimizer.param_groups:
            g["lr"] = stage["lr"]
    
    for epoch in range(stage["epochs"]):
        loss = train_epoch(model, dataloader, vae, optimizer, stage)
        print(f"Epoch {epoch}: loss={loss:.6f}")
        
        # Save checkpoint every epoch
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "stage": stage,
            "epoch": epoch,
            "loss": loss,
        }, f"checkpoints/stage_{stage['context']}ctx_epoch{epoch}.pt")
```

---

## Phase 6: Inference & Playable Client

### 6.1 Inference Engine with KV Cache

```python
@torch.no_grad()
class MinecraftInferenceEngine:
    """
    Real-time inference loop for the world model.
    Maintains KV cache for efficient autoregressive generation.
    """
    def __init__(self, world_model, vae, device='cuda'):
        self.model = world_model.eval().to(device)
        self.vae = vae.eval().to(device)
        self.device = device
        self.kv_cache = None
        self.latent_buffer = []  # Rolling window of past latents
        self.context_window = 16
        self.frame_count = 0
        
    def reset(self, seed_frame):
        """Initialize with a real frame from the dataset or environment."""
        with torch.no_grad():
            latent = self.vae.encode(seed_frame.to(self.device)).mode()
        # Normalize
        latent = (latent - channel_mean.to(self.device)) / (channel_std.to(self.device) + 1e-6)
        self.latent_buffer = [latent]
        self.kv_cache = None
        self.frame_count = 0
        
    def step(self, action_token):
        """
        One inference step: latent + action → next frame.
        Returns decoded RGB frame.
        """
        # Prepare input: pad buffer to context window
        context = self.latent_buffer[-self.context_window:]
        if len(context) < self.context_window:
            # Pad with copies of first frame
            pad_len = self.context_window - len(context)
            context = [context[0]] * pad_len + context
        
        input_latents = torch.stack(context, dim=0).unsqueeze(0)  # [1, T, 4, 16, 16]
        input_actions = torch.full((1, input_latents.shape[1]), 
                                    action_token, dtype=torch.long, device=self.device)
        
        # Forward through world model (with KV cache)
        next_latent = self.model(input_latents, input_actions, self.kv_cache)
        
        # Store in buffer
        self.latent_buffer.append(next_latent)
        self.frame_count += 1
        
        # Re-grounding: every 32 frames, re-encode from a real frame
        if self.frame_count % 32 == 0 and hasattr(self, 'real_frame'):
            self._re_ground(self.real_frame)
        
        # Decode to RGB
        denorm_latent = next_latent * (channel_std.to(self.device) + 1e-6) + channel_mean.to(self.device)
        frame = self.vae.decode(denorm_latent)
        
        return frame.clamp(0, 1)
    
    def _re_ground(self, real_frame):
        """Reset latent buffer from a real environment frame to combat drift."""
        with torch.no_grad():
            latent = self.vae.encode(real_frame.to(self.device)).mode()
        latent = (latent - channel_mean.to(self.device)) / (channel_std.to(self.device) + 1e-6)
        self.latent_buffer[-1] = latent
```

### 6.2 Keyboard → Action Token Mapping

```python
# Minecraft key bindings to action token mapping
KEY_TO_ACTION = {
    'w': 3,    # forward
    's': 1,    # back
    'a': 15,   # left
    'd': 16,   # right
    ' ': 14,   # jump
    'shift': 17,  # sneak
    'ctrl': 18,   # sprint
    'q': 2,    # drop
    'e': 13,   # inventory
    '1': 4,    # hotbar.1
    '2': 5,    # hotbar.2
    '3': 6,    # hotbar.3
    '4': 7,    # hotbar.4
    '5': 8,    # hotbar.5
    '6': 9,    # hotbar.6
    '7': 10,   # hotbar.7
    '8': 11,   # hotbar.8
    '9': 12,   # hotbar.9
    'left': 15,   # left (camera left would be handled separately)
    'right': 16,  # right
    'up': 3,      # forward
    'down': 1,    # back
}

# Mouse → camera bins
mouse_to_camera = quantizer.discretize(np.array([dy, dx]))  # returns [pitch_bin, yaw_bin]

def build_action_token(keys_pressed, mouse_dx, mouse_dy):
    """Build a single 51-token action from keyboard state + mouse delta."""
    token_id = 0
    
    # Keyboard token (0-21)
    for key in keys_pressed:
        if key in KEY_TO_ACTION:
            token_id = KEY_TO_ACTION[key]
            break
    
    # Camera token (28 = center, 0-13 = pitch, 14-27 = yaw)
    if abs(mouse_dx) > 1 or abs(mouse_dy) > 1:
        cam = quantizer.discretize(np.array([mouse_dy, mouse_dx]))
        # cam: [pitch_bin(0-20), yaw_bin(0-20)]
        # Map to token range: pitch=0-13, yaw=14-27, center=28
        if cam[0] != 10 or cam[1] != 10:  # Not center
            pitch_token = max(0, min(13, cam[0] - 4))   # 0-13
            yaw_token = max(14, min(27, cam[1] + 14))   # 14-27
            token_id = yaw_token  # Prioritize yaw for horizontal mouse movement
        # If keyboard key was pressed too, combine information
    
    return token_id
```

### 6.3 PyGame Client Loop

```python
import pygame
import torch
import numpy as np

def run_game_client(engine, seed_frame, fps_target=15):
    """
    Interactive Minecraft world model client.
    Captures keyboard input → feeds to world model → renders predicted frames.
    """
    pygame.init()
    screen = pygame.display.set_mode((512, 512))
    clock = pygame.time.Clock()
    running = True
    
    # Initialize
    engine.reset(torch.tensor(seed_frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0)
    
    # Pre-allocate display surface
    frame_surface = pygame.Surface((128, 128))
    
    while running:
        keys_pressed = []
        mouse_dx, mouse_dy = 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    # Re-ground with real frame
                    pass
            elif event.type == pygame.MOUSEMOTION:
                mouse_dx += event.rel[0] * 0.5
                mouse_dy += event.rel[1] * 0.5
        
        # Get currently pressed keys
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_w]: keys_pressed.append('w')
        if pressed[pygame.K_s]: keys_pressed.append('s')
        if pressed[pygame.K_a]: keys_pressed.append('a')
        if pressed[pygame.K_d]: keys_pressed.append('d')
        if pressed[pygame.K_SPACE]: keys_pressed.append(' ')
        if pressed[pygame.K_LSHIFT]: keys_pressed.append('shift')
        if pressed[pygame.K_LCTRL]: keys_pressed.append('ctrl')
        
        # Build action token
        action_token = build_action_token(keys_pressed, mouse_dx, mouse_dy)
        
        # World model step
        frame = engine.step(action_token)
        
        # Render: latent → RGB → display
        frame_np = frame.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [128, 128, 3]
        frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)
        
        # Scale up to 512x512 for display
        pygame.surfarray.blit_array(frame_surface, frame_np.swapaxes(0, 1))
        scaled = pygame.transform.scale(frame_surface, (512, 512))
        screen.blit(scaled, (0, 0))
        
        pygame.display.flip()
        clock.tick(fps_target)
    
    pygame.quit()
```

### 6.4 Terminal-Based Client (ratatui/lower-overhead alternative)

For lower latency or headless operation:
```python
# Use a terminal-based approach with python-pygame or raw GL
# Options:
# 1. PyGame window (above) — simplest, ~15 FPS achievable
# 2. raw GLFW + OpenGL — lower latency, harder to set up
# 3. Terminal ASCII render with color — novelty, not recommended for quality
# 4. WebSocket server → browser client — decouple model from display

# Recommendation: Start with PyGame client (6.3), profile latency, then:
# - If model inference >60ms: optimize model (Diagonal Decoding, INT8 quant)
# - If display >16ms: use GLFW/pygame.display alternative
# - If input latency >50ms: use raw input polling instead of event loop
```

### 6.5 Error Accumulation Mitigation

```python
"""
Known Problem: Autoregressive video generation drifts within 50-100 frames.
Solutions (apply in order as needed):

1. Re-grounding (easiest, most effective):
   - Every 32 steps, encode a real frame from the environment
   - Resets the latent trajectory to ground truth
   
2. Classifier-free guidance (CFG) on latents:
   - Noise the latent slightly before each prediction
   - guidance_scale = 1.5 → sharper, more stable rollouts
   
3. Sliding window with overlap:
   - Generate in overlapping 16-frame windows
   - Average the overlapping regions
   - Reduces boundary artifacts
   
4. Diffusion Denoising (Oasis-style):
   - Instead of direct AR prediction, use diffusion over latents
   - Each step: add noise, then denoise conditioned on previous frame + action
   - More stable but slower (multiple denoising steps per frame)
   
5. Temporal smoothing:
   - After decoding, blend adjacent frames: frame_t = 0.7 * frame_t + 0.3 * frame_{t-1}
   - Reduces flickering artifacts
"""
```
```