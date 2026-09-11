"""Evaluate VAE reconstruction quality on a held-out shard (local single-GPU).

Wraps mw_jepa.vae_experiments.evaluate_vae (PSNR / SSIM / LPIPS / FID).

Usage: python scripts/eval_vae.py --shard 3 --num-frames 500 [--batch-size 32] [--use-finetuned]
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mw_jepa.vae import load_vae
from mw_jepa.data import MinecraftFrameStream
from mw_jepa.vae_experiments import evaluate_vae as run_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=3)
    ap.add_argument("--num-frames", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--use-finetuned", action="store_true")
    ap.add_argument("--ckpt", default="checkpoints/vae-finetune-v1.pt")
    args = ap.parse_args()

    vae = load_vae(device="cuda")
    if args.use_finetuned:
        ckpt_path = Path(args.ckpt)
        if ckpt_path.exists():
            vae.load_state_dict(torch.load(ckpt_path))
            print(f"Loaded fine-tuned VAE from {ckpt_path}")
        else:
            print("No fine-tuned checkpoint found, using base VAE")

    stream = MinecraftFrameStream(
        shard_start=args.shard, shard_end=args.shard + 1, target_size=256
    )
    loader = DataLoader(stream, batch_size=args.batch_size, num_workers=0, drop_last=True)

    results = run_eval(vae, loader, num_frames=args.num_frames)
    for k, v in results.items():
        print(f"{k}: {v:.4f}" if v is not None else f"{k}: N/A")


if __name__ == "__main__":
    main()
