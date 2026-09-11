"""Overfit a tiny world model on a single repeated trajectory with real actions (local single-GPU).

Pass condition: final loss < 20% of initial loss (proves gradients flow).

Post-training diagnostics:
1. Copy-paste baseline — does trained loss beat "frame t+1 ≈ frame t"?
2. Shuffled action comparison — are actions actually steering predictions?
3. Per-quadrant error map — does action-driven error localise to expected regions?

The trajectory is chosen from shard 0, action distribution is printed.

Usage: python scripts/overfit.py --config configs/stage1_4ctx.yaml
"""
import argparse
from pathlib import Path

CKPT_DIR = Path("checkpoints/overfit_test")


def main():
    import torch
    from torch.utils.data import IterableDataset
    from torchvision.utils import make_grid, save_image
    from datasets import load_dataset

    from mw_jepa.vae import load_vae, encode_frames, decode_latents
    from mw_jepa.data import decode_jpeg, TESS_REPO
    from mw_jepa.action_tokenizer import parse_lumine_action, KEYBOARD_TOKENS
    from mw_jepa.world_model import MineWorldModel
    from mw_jepa.trainer import Trainer
    from mw_jepa.config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage1_4ctx.yaml")
    ap.add_argument("--sequence-length", type=int, default=4)
    ap.add_argument("--num-trajectory-frames", type=int, default=20)
    ap.add_argument("--num-epochs", type=int, default=20)
    ap.add_argument("--shuffle-repeats", type=int, default=20)
    args = ap.parse_args()

    sequence_length = args.sequence_length
    num_trajectory_frames = args.num_trajectory_frames
    num_epochs = args.num_epochs
    shuffle_repeats = args.shuffle_repeats

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: CPU-only action pre-scan (no GPU cost) ──
    print("=" * 60)
    print("OVERFIT TEST: Single trajectory memorisation (real actions)")
    print("=" * 60)
    print("\n  Scanning shard 0 for a trajectory with sufficient action diversity ...")
    token_names = {v: k for k, v in KEYBOARD_TOKENS.items()}

    def find_good_trajectory(shard, min_frames, max_static_ratio=0.80, min_unique_actions=2):
        """Stream dataset parsing actions only (CPU). Returns (action_tokens, frames_bytes) or None."""
        ds = load_dataset(
            TESS_REPO, split="train", streaming=True,
            data_files=f"data/shard_{shard:05d}.parquet",
        )
        action_tokens, frames_bytes = [], []
        last_video_id = None
        action_counts = {}
        for ex in ds:
            video_id = ex["video_id"]
            if last_video_id is None:
                last_video_id = video_id
            if video_id != last_video_id:
                if len(action_tokens) >= min_frames:
                    break
                action_tokens, frames_bytes = [], []
                action_counts = {}
                last_video_id = video_id
            if len(action_tokens) >= min_frames:
                break
            token = parse_lumine_action(ex["action"])
            action_tokens.append(token)
            frames_bytes.append(ex["image"])
            action_counts[token] = action_counts.get(token, 0) + 1

        if len(action_tokens) < min_frames:
            return None, None, None

        esc_count = action_counts.get(0, 0)
        noop_count = action_counts.get(28, 0)
        static_ratio = (esc_count + noop_count) / len(action_tokens)
        if static_ratio > max_static_ratio:
            return None, None, None

        # Require action diversity — at least `min_unique_actions` different
        # non-static action tokens. This ensures shuffled-vs-correct comparison
        # is meaningful (shuffling all-identical actions gives the same sequence).
        unique_non_static = set(t for t in action_tokens if t not in (0, 28))
        if len(unique_non_static) < min_unique_actions:
            return None, None, None

        return action_tokens, frames_bytes, action_counts

    # Scan shards until we find a good trajectory
    action_tokens, frames_bytes, action_counts = None, None, None
    for shard in range(5):  # try shards 0-4
        result = find_good_trajectory(shard, min_frames=num_trajectory_frames)
        if result[0] is not None:
            action_tokens, frames_bytes, action_counts = result
            print(f"  Found suitable trajectory in shard {shard}")
            break
        print(f"  Shard {shard}: too static, skipping ...")

    if action_tokens is None:
        raise RuntimeError(
            "No suitable trajectory found across shards 0-4. "
            "Try running canary Phase 2 "
            "(`python scripts/canary.py --config configs/stage1_4ctx.yaml`) "
            "to inspect the action distribution across shards."
        )

    # Print action distribution for this trajectory
    total_acts = len(action_tokens)
    print(f"\n  Trajectory action distribution ({total_acts} frames):")
    keyboard_acts = sum(c for t, c in action_counts.items() if t <= 21)
    camera_acts = sum(c for t, c in action_counts.items() if 22 <= t <= 27)
    center_acts = action_counts.get(28, 0)
    print(f"    Keyboard (0-21):    {keyboard_acts} ({100*keyboard_acts/total_acts:.0f}%)")
    if camera_acts:
        print(f"    Camera  (22-27):    {camera_acts} ({100*camera_acts/total_acts:.0f}%)")
    print(f"    Center  (28):       {center_acts} ({100*center_acts/total_acts:.0f}%)")
    for tid in sorted(action_counts):
        if tid > 21:
            continue
        label = token_names.get(tid, f"key_{tid}")
        print(f"      [{tid:2d}] {label:<12s} {action_counts[tid]}")

    # Print raw action token sequence
    seq_labels = []
    for t in action_tokens:
        if t <= 21:
            seq_labels.append(token_names.get(t, f"k{t}"))
        elif t <= 27:
            seq_labels.append(f"cam{t}")
        else:
            seq_labels.append("noop")
    print(f"\n  Action sequence ({len(seq_labels)} frames):")
    for row_start in range(0, len(seq_labels), 10):
        row = seq_labels[row_start:row_start + 10]
        print(f"    [{row_start:3d}] " + " ".join(f"{a:<8s}" for a in row))
    print()

    # ── Step 2: GPU work — load VAE, decode frames, encode latents ──
    print("  Loading VAE and encoding frames ...")
    vae = load_vae(device="cuda")

    def validate_frame(frame: torch.Tensor, tol: float = 1e-6) -> tuple[bool, str]:
        """Check frame is not corrupted: non-flat, non-NaN, non-degenerate."""
        if torch.isnan(frame).any():
            return False, "NaN values detected"
        mean = frame.mean().item()
        if mean < tol:
            return False, f"all-black frame (mean={mean:.6f})"
        if (frame > tol).sum() < 10:
            return False, f"frame has only {(frame > tol).sum()} non-zero pixels"
        return True, "OK"

    frames = []
    for idx, fb in enumerate(frames_bytes):
        f = decode_jpeg(fb, 256)
        ok, msg = validate_frame(f)
        if not ok:
            raise RuntimeError(
                f"Frame {idx} in trajectory failed validation: {msg}. "
                "Trajectory contains corrupted data; re-run will scan a different shard."
            )
        frames.append(f)

    frames_tensor = torch.stack(frames).cuda()
    with torch.no_grad():
        latents = encode_frames(vae, frames_tensor).cpu()
    action_tensor = torch.tensor(action_tokens, dtype=torch.long)

    # Build sliding-window sequences
    seq_len = sequence_length + 1
    sequences = []
    for i in range(0, len(frames) - seq_len + 1, seq_len // 2):
        sequences.append({
            "latents": latents[i:i + seq_len],
            "actions": action_tensor[i:i + seq_len],
        })

    print(f"  {len(sequences)} sequences of length {seq_len} (stride {seq_len // 2})")

    class RepeatDataset(IterableDataset):
        def __init__(self, seqs, repeats):
            self.seqs = seqs
            self.repeats = repeats
        def __iter__(self):
            for _ in range(self.repeats):
                for s in self.seqs:
                    yield s

    train_stream = RepeatDataset(sequences, repeats=shuffle_repeats)

    # ── Miniature model ──
    cfg = load_config(args.config)
    cfg["model"].update({
        "embed_dim": 256,
        "num_blocks": 4,
        "num_heads": 4,
    })
    cfg["epochs"] = num_epochs
    cfg["batch_size"] = min(8, len(sequences))

    model = MineWorldModel(**cfg["model"]).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} params")

    # ── Train ──
    trainer = Trainer(model=model, vae=None, config=cfg, ckpt_dir=CKPT_DIR)
    trainer.train(train_stream, batch_size=cfg["batch_size"], num_epochs=num_epochs)

    # ── Post-training diagnostics ──
    print()
    print("=" * 60)
    print("POST-TRAINING DIAGNOSTICS")
    print("=" * 60)

    # Hold-out batch (first sequence, not used during training if drop_last rounds down)
    diag_batch = sequences[0]
    diag_latents = diag_batch["latents"].unsqueeze(0).cuda()  # [1, T, 4, 32, 32]
    diag_actions = diag_batch["actions"].unsqueeze(0).cuda()    # [1, T]
    in_latents = diag_latents[:, :-1]
    target = diag_latents[:, -1]
    in_actions = diag_actions[:, :-1]

    model.eval()
    with torch.no_grad():
        # 1. Copy-paste baseline: predict last input frame as next frame
        last_input = in_latents[:, -1]
        baseline_loss = torch.nn.functional.mse_loss(last_input, target).item()

        # 2. Correct actions
        pred_correct = model(in_latents, in_actions)
        correct_loss = torch.nn.functional.mse_loss(pred_correct, target).item()

        # 3. Shuffled actions (reverse temporal order, preserves distribution)
        shuffled = in_actions.flip(dims=[1])
        with torch.no_grad():
            pred_shuffled = model(in_latents, shuffled)
        shuffled_loss = torch.nn.functional.mse_loss(pred_shuffled, target).item()

        # 4. Decode to pixels for visual comparison
        frame_target = decode_latents(vae, target).cpu()
        frame_correct = decode_latents(vae, pred_correct).cpu()
        frame_shuffled = decode_latents(vae, pred_shuffled).cpu()
        frame_baseline = decode_latents(vae, last_input).cpu()

    print(f"  Copy-paste baseline:  {baseline_loss:.6f}  (predict frame t+1 = frame t)")
    print(f"  Correct actions:      {correct_loss:.6f}")
    print(f"  Shuffled actions:     {shuffled_loss:.6f}")
    print(f"  Action-conditioning delta: {shuffled_loss - correct_loss:+.6f} "
          f"(positive = actions are steering predictions)")
    print()

    # 5. Per-quadrant error breakdown (latent space)
    print("  Per-quadrant latent error (correct vs shuffled):")
    _, C, H, W = target.shape
    q_h, q_w = H // 2, W // 2
    quadrant_names = ["top-left", "top-right", "bottom-left", "bottom-right"]
    quadrant_deltas = []
    for idx, name in enumerate(quadrant_names):
        row, col = idx // 2, idx % 2
        sl = (slice(None), slice(None),
              slice(row * q_h, (row + 1) * q_h),
              slice(col * q_w, (col + 1) * q_w))
        err_correct = torch.nn.functional.mse_loss(pred_correct[sl], target[sl]).item()
        err_shuff = torch.nn.functional.mse_loss(pred_shuffled[sl], target[sl]).item()
        delta = err_shuff - err_correct
        quadrant_deltas.append(delta)
        print(f"    {name:14s}  correct={err_correct:.6f}  shuffled={err_shuff:.6f}  "
              f"delta={delta:+.6f}")

    max_delta = max(quadrant_deltas)
    min_delta = min(quadrant_deltas)
    spatial_spread = max_delta - min_delta
    print(f"  Quadrant delta spread: {spatial_spread:.6f} "
          f"(larger = action effects localise to specific regions)")
    print()

    # 6. Tensor stats for each grid cell (avoids "looks black" ambiguity)
    print("  Pixel-level stats per grid cell (values in [0,1]):")
    for label, tensor in [("target", frame_target), ("correct", frame_correct),
                          ("shuffled", frame_shuffled), ("baseline", frame_baseline)]:
        err = (tensor - frame_target).abs()
        print(f"    {label:12s}  pred  min={tensor.min():.4f}  max={tensor.max():.4f}  mean={tensor.mean():.4f}")
        print(f"    {'':12s}  error min={err.min():.4f}      max={err.max():.4f}      mean={err.mean():.4f}")

    # 7. Save comparison grid to checkpoint dir
    vis_rows = []
    for label, tensor in [("target", frame_target), ("correct", frame_correct),
                          ("shuffled", frame_shuffled), ("baseline", frame_baseline)]:
        err_map = (tensor - frame_target).abs()
        stacked = torch.cat([tensor, err_map], dim=0)
        vis_rows.append(stacked)

    grid = make_grid(torch.cat(vis_rows, dim=0), nrow=2)
    save_image(grid, str(CKPT_DIR / "overfit_diagnostics.png"))
    print(f"  Diagnostic grid → {CKPT_DIR}/overfit_diagnostics.png")
    print("  Columns: prediction | error heatmap")
    print("  Rows:    target | correct actions | shuffled actions | copy-paste baseline")
    print()

    # ── Results ──
    print("=" * 60)
    print("OVERFIT RESULTS")
    print("=" * 60)

    losses = trainer.loss_history
    initial, final = losses[0], losses[-1]
    print(f"  Initial loss:          {initial:.6f}")
    print(f"  Final loss:            {final:.6f}")
    print(f"  Ratio:                 {final / initial:.4f} (goal: < 0.20)")

    spark = "".join(
        "█" if v == max(losses)
        else "▄" if v > (max(losses) + min(losses)) / 2
        else " " for v in losses
    )
    print(f"  Trend:                 [{spark}]")
    print()

    # Pass/fail logic
    gradient_pass = final < initial * 0.20
    conditioning_pass = shuffled_loss > correct_loss * 1.10  # >10% worse under shuffled actions
    baseline_pass = correct_loss < baseline_loss * 0.95      # >5% better than copy-paste

    failures = []
    if not gradient_pass:
        failures.append("loss did not drop sufficiently (gradient issue)")
    if not conditioning_pass:
        failures.append(
            f"shuffled actions only {((shuffled_loss / correct_loss) - 1) * 100:.1f}% worse "
            f"(threshold: 10%) — action conditioning may be dead"
        )
    if not baseline_pass:
        failures.append(
            f"trained loss {correct_loss:.4f} does not beat copy-paste baseline "
            f"{baseline_loss:.4f}"
        )

    if failures:
        print(f"  RESULT: FAIL — {failures[0]}")
        raise RuntimeError("Overfit test failed: " + "; ".join(failures))
    else:
        print("  RESULT: PASS")
        if conditioning_pass:
            print("  [OK]  Actions are steering predictions (shuffled actions increase loss)")
        if baseline_pass:
            print("  [OK]  Model beats the copy-paste baseline")
        print("  [OK]  Training loop verified")

    print("=" * 60)


if __name__ == "__main__":
    main()
