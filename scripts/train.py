"""Local single-GPU training. Usage: python scripts/train.py --config configs/stage1_4ctx.yaml [--resume ckpt/best.pt]"""
import argparse
from pathlib import Path
from mw_jepa.config import load_config
from mw_jepa.world_model import MineWorldModel
from mw_jepa.vae import load_vae
from mw_jepa.data import WorldModelStream
from mw_jepa.trainer import Trainer


def resolve_resume_arg(resume: str, ckpt_dir: Path) -> str:
    """Local path passthrough, or hf://<owner>/<repo>/<path> Hub download.

    Hub files land under ckpt_dir/_hub/ (cached by huggingface_hub, so a
    re-resume is free). Auth via HF_TOKEN env var for private repos.
    """
    if Path(resume).exists():
        return resume
    if resume.startswith("hf://"):
        from huggingface_hub import hf_hub_download

        rest = resume[len("hf://"):]
        parts = rest.split("/")
        if len(parts) < 3:
            raise ValueError(f"Bad Hub resume path (want hf://owner/repo/file): {resume}")
        repo_id, filename = "/".join(parts[:2]), "/".join(parts[2:])
        local = hf_hub_download(
            repo_id=repo_id, filename=filename,
            local_dir=ckpt_dir / "_hub", local_dir_use_symlinks=False,
        )
        print(f"Downloaded Hub checkpoint → {local}", flush=True)
        return local
    raise FileNotFoundError(f"Resume checkpoint not found: {resume}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None,
                    help="local path or hf://owner/repo/path (downloaded, cached)")
    ap.add_argument("--reset-optimizer", action="store_true",
                    help="load only model weights (fresh optimizer/scheduler). "
                         "Use for cross-stage resume where the LR schedule restarts.")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the model (benchmark first: must match uncompiled loss)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="cap total optimizer steps (benchmarking / smoke runs)")
    ap.add_argument("--no-hub", action="store_true",
                    help="disable Hub mirror for this run (benchmarks, smoke tests)")
    ap.add_argument("--ckpt-dir", default=None,
                    help="override checkpoint dir (benchmarks: keep stage dirs pristine)")
    ap.add_argument("--shard-start", type=int, default=0)
    ap.add_argument("--shard-end", type=int, default=10)
    args = ap.parse_args()
    cfg = load_config(args.config)
    model = MineWorldModel(**cfg["model"]).cuda()
    if args.compile:
        import torch
        model = torch.compile(model)
        print("torch.compile enabled", flush=True)
    vae = load_vae(device="cuda")
    batch_size = cfg.get("batch_size", 8)
    stream = WorldModelStream(sequence_length=cfg["context"], shard_start=args.shard_start, shard_end=args.shard_end, target_size=cfg["data"]["target_size"])
    # Bar-only step estimate from parquet footers (seconds, no download).
    # Never caps training: the stream always runs to exhaustion.
    hint = None
    try:
        from mw_jepa.data import estimate_sequences
        est = estimate_sequences(args.shard_start, args.shard_end, cfg["context"])
        if est:
            hint = max(1, est // batch_size)
            print(f"Estimated ~{hint} steps/epoch from shard metadata (bar only)", flush=True)
    except Exception as e:
        print(f"Step estimate unavailable ({e}) — bar will be indeterminate", flush=True)
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else Path("checkpoints") / cfg.get("run_name", "run")
    trainer = Trainer(model=model, vae=vae, config=cfg, ckpt_dir=ckpt_dir)
    if args.no_hub:
        trainer.hf_repo_id = None
        print("Hub mirror disabled for this run", flush=True)
    if args.resume:
        trainer.load_checkpoint(resolve_resume_arg(args.resume, ckpt_dir),
                                weights_only=args.reset_optimizer)
    trainer.train(stream, batch_size=batch_size, num_epochs=cfg.get("epochs", 2),
                  max_steps=args.max_steps, total_steps_hint=hint)

if __name__ == "__main__":
    main()
