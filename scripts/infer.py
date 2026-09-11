"""Local playable inference loop (Modal-free).

Usage: python scripts/infer.py --ckpt checkpoints/<run>/best.pt [--config configs/stage1_4ctx.yaml]

Controls: W=forward, S=back, A=left, D=right, Space=jump. ESC quits.
The context window is 16 frames; the engine re-grounds on the latest
decoded frame every 32 steps to bound autoregressive drift.
"""

import argparse
from pathlib import Path

import torch

from mw_jepa.config import load_config
from mw_jepa.inference import InferenceEngine
from mw_jepa.vae import load_vae
from mw_jepa.world_model import MineWorldModel

# Key map: pygame key constant name -> action token id.
KEY_TO_TOKEN = {
    "w": 3,  # forward
    "s": 1,  # back
    "a": 15,  # left
    "d": 16,  # right
    "space": 14,  # jump
}
NOOP_TOKEN = 28
REGROUND_EVERY = 32


def load_engine(ckpt_path: str, config_path: str | None, device: str) -> InferenceEngine:
    ckpt = torch.load(ckpt_path, map_location=device)
    if config_path:
        cfg = load_config(config_path)
    else:
        cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})
    model = MineWorldModel(**model_cfg)
    model.load_state_dict(ckpt["model"])
    vae = load_vae(device=device)
    mean = ckpt.get("mean")
    std = ckpt.get("std")
    if mean is not None:
        mean = torch.as_tensor(mean)
    if std is not None:
        std = torch.as_tensor(std)
    context = cfg.get("context", 16)
    engine = InferenceEngine(model, vae, mean=mean, std=std, context=context, device=device)
    target_size = cfg.get("data", {}).get("target_size", 256)
    seed = torch.zeros(1, 3, target_size, target_size)
    engine.reset(seed)
    return engine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if not Path(args.ckpt).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    import pygame

    engine = load_engine(args.ckpt, args.config, args.device)

    pygame.init()
    screen = pygame.display.set_mode((256, 256))
    pygame.display.set_caption("minecraft-world-jepa (WASD+space, ESC quits)")
    clock = pygame.time.Clock()
    name_to_pygame = {
        "w": pygame.K_w,
        "s": pygame.K_s,
        "a": pygame.K_a,
        "d": pygame.K_d,
        "space": pygame.K_SPACE,
    }

    frame = None
    steps = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        token = NOOP_TOKEN
        for name, tid in KEY_TO_TOKEN.items():
            if keys[name_to_pygame[name]]:
                token = tid
                break

        frame = engine.step(token)
        steps += 1
        if steps % REGROUND_EVERY == 0:
            engine.reset(frame.detach().cpu())

        img = (frame.squeeze(0).detach().cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
        surf = pygame.surfarray.make_surface(img.transpose(1, 0, 2))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(10)

    pygame.quit()


if __name__ == "__main__":
    main()
