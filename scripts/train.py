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
