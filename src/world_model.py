import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.tensor_contracts import assert_action_tensor, assert_latent_tensor


class LatentPatchEmbed(nn.Module):
    """Tokenize [4, H, W] latent into (H/ps)*(W/ps) patch tokens."""

    def __init__(self, latent_dim=4, patch_size=2, embed_dim=1024):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(latent_dim, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.proj(x)
        _, _, h, w = x.shape
        x = rearrange(x, "bt d h w -> bt (h w) d")
        x = rearrange(x, "(b t) n d -> b t n d", b=B, t=T)
        return x


class ActionAdapter(nn.Module):
    """Frame-level action conditioning via per-block MLP adapter (DWS paper).
    action_emb: [B, S, D] where S is the full token sequence length.

    Initialized so the adapter output starts at ~0 (zero-initialised last linear
    layer), allowing gradient to flow through the full network from step 1.
    """

    def __init__(self, embed_dim=1024, action_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        # Zero-init last layer → adapter output starts at 0
        self.net[2].weight.data.zero_()
        self.net[2].bias.data.zero_()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x, action_emb):
        gate = self.scale.tanh()
        return x + gate * self.net(action_emb)


class FlashAttention(nn.Module):
    """Multi-head causal self-attention with optional RoPE."""

    def __init__(self, embed_dim=1024, num_heads=16, causal=True, use_rope=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal
        self.use_rope = use_rope

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        if use_rope:
            self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        if self.use_rope:
            q = self.rope(q)
            k = self.rope(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # PyTorch 2.5+ SDPA dispatches flash or mem-efficient backend automatically,
        # avoiding O(N^2) attention score materialization. This is critical: stages 3-4
        # with 8192-token sequences would OOM with a naive matmul fallback.
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=self.causal,
        )
        out = out.transpose(1, 2).reshape(B, N, D)

        return self.out_proj(out)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim, max_seq_len=4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)[None, :, None, :]
        return x * emb.cos() + self._rotate_half(x) * emb.sin()

    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)


class WorldModelBlock(nn.Module):
    """Transformer block with attention, action injection, and MLP."""

    def __init__(self, embed_dim=1024, num_heads=16, mlp_ratio=4, action_dim=512):
        super().__init__()
        self.norm1 = nn.RMSNorm(embed_dim)
        self.attn = FlashAttention(embed_dim, num_heads, causal=True, use_rope=True)
        self.norm2 = nn.RMSNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
        )
        self.action_adapter = ActionAdapter(embed_dim, action_dim)

    def forward(self, x, action_emb):
        x = x + self.attn(self.norm1(x))
        x = self.action_adapter(x, action_emb)
        x = x + self.mlp(self.norm2(x))
        return x


class MineWorldModel(nn.Module):
    """Autoregressive world model that predicts the next latent frame.
    Architecture: MineWorld-style AR transformer with DWS action injection.
    """

    def __init__(
        self,
        latent_dim=4,
        patch_size=2,
        latent_grid_size=32,
        embed_dim=1024,
        num_blocks=16,
        num_heads=16,
        max_frames=32,
        num_actions=29,
        action_embed_dim=512,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.num_patches = (latent_grid_size // patch_size) ** 2
        self.latent_grid_size = latent_grid_size
        self.max_frames = max_frames

        self.patch_embed = LatentPatchEmbed(latent_dim, patch_size, embed_dim)
        self.action_embedder = nn.Embedding(num_actions, action_embed_dim)
        self.action_proj = nn.Linear(action_embed_dim, embed_dim)

        self.blocks = nn.ModuleList([
            WorldModelBlock(embed_dim, num_heads, action_dim=embed_dim)
            for _ in range(num_blocks)
        ])

        self.norm = nn.RMSNorm(embed_dim)

        # Output head: [embed_dim] → latent patch [latent_dim * patch_size^2]
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, latent_dim * patch_size * patch_size),
        )

        self._init_weights()

        # Re-apply ActionAdapter zero-init AFTER _init_weights, since _init_weights
        # iterates self.modules() and overwrites ActionAdapter.net[2] with Xavier.
        self._zero_init_action_adapters()

    def _zero_init_action_adapters(self):
        """Zero-init the last linear layer of every ActionAdapter."""
        for block in self.blocks:
            adapter = block.action_adapter
            adapter.net[2].weight.data.zero_()
            adapter.net[2].bias.data.zero_()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, latent_seq, action_seq):
        B, T = latent_seq.shape[:2]
        assert_latent_tensor(latent_seq, grid_size=self.latent_grid_size, name="model latent_seq")
        assert_action_tensor(action_seq, sequence_length=T, num_actions=self.action_embedder.num_embeddings)

        x = self.patch_embed(latent_seq)  # [B, T, N, D]

        action_emb = self.action_embedder(action_seq)  # [B, T, A]
        action_emb = self.action_proj(action_emb)       # [B, T, D]

        # Expand action_emb to match flattened token sequence [B, T*N, D]
        action_emb = action_emb[:, :, None, :].expand(-1, -1, self.num_patches, -1)
        action_emb = rearrange(action_emb, "b t n d -> b (t n) d")

        x = rearrange(x, "b t n d -> b (t n) d")

        for block in self.blocks:
            x = block(x, action_emb)

        x = self.norm(x)

        last_frame_tokens = x[:, -self.num_patches:, :]
        patches = self.output_head(last_frame_tokens)
        grid_h = grid_w = int(self.num_patches ** 0.5)
        patches = patches.view(B, grid_h, grid_w, self.latent_dim, self.patch_size, self.patch_size)
        next_latent = rearrange(
            patches, "b h w c ph pw -> b c (h ph) (w pw)"
        )

        return next_latent
