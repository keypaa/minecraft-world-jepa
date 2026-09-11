"""Smoke test for Phase 2 components."""
import sys
sys.path.insert(0, ".")
import torch

from src.action_tokenizer import parse_lumine_action
from src.config import load_config
from src.trainer import collate_stream
from src.world_model import MineWorldModel

cfg = load_config("configs/stage1_4ctx.yaml")
assert cfg["model"]["latent_grid_size"] == 32
assert cfg["data"]["target_size"] == 256
assert cfg["batch_size"] == 32
print("Config inheritance OK")

# Test Lumine parser
action1 = '<|action_start|> 0 0 0 ; LMB ; LMB ; LMB ; LMB <|action_end|>'
action2 = '<|action_start|> 0 0 0 ; forward ; forward ; forward ; forward <|action_end|>'
action3 = '<|action_start|> 5 -3 0 ; ; ; ; <|action_end|>'
action4 = '<|action_start|> 0 0 0 ; ; ; ; <|action_end|>'
print(f"Attack token:  {parse_lumine_action(action1)}")    # expect 20 (attack)
print(f"Forward token: {parse_lumine_action(action2)}")    # expect 3 (forward)
print(f"Camera token:  {parse_lumine_action(action3)}")    # expect 27 (yaw)
print(f"No-op token:   {parse_lumine_action(action4)}")    # expect 28 (center)

# Test world model
model = MineWorldModel(
    latent_dim=4, patch_size=2, latent_grid_size=32, embed_dim=64,
    num_blocks=1, num_heads=4, max_frames=8,
    num_actions=29, action_embed_dim=128,
)
B, T = 1, 2
latents = torch.randn(B, T, 4, 32, 32)
actions = torch.randint(0, 29, (B, T))
out = model(latents, actions)
loss = torch.nn.functional.mse_loss(out, latents[:, -1])
loss.backward()
print(f"Model OK — params: {sum(p.numel() for p in model.parameters()):,}")

# Test collate
batch = [
    {"frames": torch.rand(5, 3, 256, 256), "actions": torch.randint(0, 29, (5,))}
    for _ in range(4)
]
collated = collate_stream(batch)
print(f"Collate frames: {list(collated['frames'].shape)}")
print(f"Collate actions: {list(collated['actions'].shape)}")

latent_batch = [
    {"latents": torch.randn(5, 4, 32, 32), "actions": torch.randint(0, 29, (5,))}
    for _ in range(4)
]
collated_latents = collate_stream(latent_batch)
print(f"Collate latents: {list(collated_latents['latents'].shape)}")

print("\nAll Phase 2 tests passed!")
