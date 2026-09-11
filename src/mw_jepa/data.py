"""Data pipeline for Phase 2: Streaming from TESS VLA dataset on HuggingFace."""

from pathlib import Path
import io
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset

from mw_jepa.action_tokenizer import parse_lumine_action, CameraQuantizer
from mw_jepa.tensor_contracts import assert_action_tensor, assert_frame_tensor, assert_latent_tensor

TESS_REPO = "TESS-Computer/minecraft-vla-stage1"
TESS_SHARDS = 303
DEFAULT_TARGET_SIZE = 256


def decode_jpeg(image_bytes: bytes, target_size: int = DEFAULT_TARGET_SIZE) -> torch.Tensor:
    """Decode JPEG bytes to [3, H, W] float tensor normalized [0, 1]."""
    import cv2
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if target_size and (img.shape[0] != target_size or img.shape[1] != target_size):
        img = cv2.resize(img, (target_size, target_size))
    frame = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    assert_frame_tensor(frame, size=target_size, name="decoded frame")
    return frame


class MinecraftFrameStream(IterableDataset):
    """Stream individual frames for VAE fine-tuning.
    Yields: [3, H, W] float tensors.
    """

    def __init__(
        self,
        shard_start: int = 0,
        shard_end: int = 2,
        target_size: int = DEFAULT_TARGET_SIZE,
    ):
        self.shard_start = shard_start
        self.shard_end = shard_end
        self.target_size = target_size

    def __iter__(self) -> Iterator[torch.Tensor]:
        num_iter = 0
        while True:
            for shard in range(self.shard_start, self.shard_end):
                ds = load_dataset(
                    TESS_REPO,
                    split="train",
                    streaming=True,
                    data_files=f"data/shard_{shard:05d}.parquet",
                )
                for ex in ds:
                    yield decode_jpeg(ex["image"], self.target_size)
                    num_iter += 1


class MinecraftFrameDataset:
    """Wrapper around MinecraftFrameStream that provides a DataLoader."""

    def __init__(self, shard_start=0, shard_end=2, target_size=DEFAULT_TARGET_SIZE):
        self.stream = MinecraftFrameStream(shard_start, shard_end, target_size)

    def to_dataloader(self, batch_size=64, num_workers=0):
        return DataLoader(
            self.stream,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )


class WorldModelStream(IterableDataset):
    """Stream (latent_seq, action_seq) pairs for world model training.
    Groups consecutive frames from the same trajectory into sequences.

    If latent_dir is provided, loads pre-computed latents from volume
    instead of encoding frames on-the-fly (much faster during training).
    """

    def __init__(
        self,
        sequence_length: int = 16,
        shard_start: int = 0,
        shard_end: int = 10,
        target_size: int = DEFAULT_TARGET_SIZE,
        resume_shard: int = 0,
        resume_row: int = 0,
        latent_dir: str | None = None,
    ):
        self.seq_len = sequence_length + 1  # +1 for target
        self.shard_start = shard_start
        self.shard_end = shard_end
        self.target_size = target_size
        self.resume_shard = resume_shard
        self.resume_row = resume_row
        self.quantizer = CameraQuantizer()
        self.latent_dir = Path(latent_dir) if latent_dir else None

    def _load_precomputed(self, shard: int):
        """Load pre-computed latents for a shard from volume."""
        import torch
        path = self.latent_dir / f"shard_{shard:05d}.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"Pre-computed latents not found at {path}. "
                f"Run precompute_latents() first."
            )
        return torch.load(path, map_location="cpu", weights_only=True)

    def __iter__(self):
        for shard in range(self.shard_start, self.shard_end):
            # Load pre-computed latents if available
            precomputed = None
            if self.latent_dir is not None:
                precomputed = self._load_precomputed(shard)
                # Pre-computed latents are indexed by frame order in shard
                latent_index = 0

            ds = load_dataset(
                TESS_REPO,
                split="train",
                streaming=True,
                data_files=f"data/shard_{shard:05d}.parquet",
            )
            # Buffer for current trajectory. Items are either latents or frames.
            item_kind = "latents" if precomputed is not None else "frames"
            buf_items, buf_actions = [], []
            last_video_id = None
            for ex in ds:
                action_token = parse_lumine_action(ex["action"])
                video_id = ex["video_id"]

                if buf_items and video_id != last_video_id:
                    yield from self._yield_sequences(buf_items, buf_actions, item_kind=item_kind)
                    buf_items, buf_actions = [], []

                if precomputed is not None:
                    latent = precomputed[latent_index]
                    assert_latent_tensor(latent, name=f"precomputed latent shard {shard}")
                    latent_index += 1
                    buf_items.append(latent)
                else:
                    frame = decode_jpeg(ex["image"], self.target_size)
                    buf_items.append(frame)
                buf_actions.append(action_token)
                last_video_id = video_id

            if buf_items:
                yield from self._yield_sequences(buf_items, buf_actions, item_kind=item_kind)

    def _yield_sequences(self, items, actions, *, item_kind: str):
        """Slide window over trajectory and yield sequences.
        items: either list of frames [3,H,W] or list of latents [4,H/8,W/8].
        """
        if len(items) < self.seq_len:
            return
        for i in range(0, len(items) - self.seq_len + 1, self.seq_len // 2):
            seq_items = torch.stack(items[i:i + self.seq_len])
            seq_actions = torch.tensor(actions[i:i + self.seq_len], dtype=torch.long)
            if item_kind == "frames":
                assert_frame_tensor(seq_items, size=self.target_size, name="frame sequence")
            elif item_kind == "latents":
                assert_latent_tensor(seq_items, name="latent sequence")
            else:
                raise ValueError(f"Unknown item kind: {item_kind}")
            assert_action_tensor(seq_actions, sequence_length=self.seq_len)
            yield {
                item_kind: seq_items,  # [T, 3, H, W] or [T, 4, H/8, W/8]
                "actions": seq_actions,  # [T]
            }


class LatentStatsComputer:
    """Compute channel-wise mean/std of VAE latents over streaming data."""

    def __init__(self, vae, shard_start=0, shard_end=2, target_size: int = DEFAULT_TARGET_SIZE):
        self.vae = vae.eval()
        self.shard_start = shard_start
        self.shard_end = shard_end
        self.target_size = target_size

    def compute(self, num_samples: int = 5000) -> dict:
        from mw_jepa.vae import encode_frames

        ds = MinecraftFrameStream(self.shard_start, self.shard_end, target_size=self.target_size)
        loader = DataLoader(ds, batch_size=32, num_workers=0)

        means, stds = [], []
        count = 0
        for batch in loader:
            if count >= num_samples:
                break
            with torch.no_grad():
                z = encode_frames(self.vae, batch.cuda())
            means.append(z.mean(dim=(0, 2, 3)).cpu())
            stds.append(z.std(dim=(0, 2, 3)).cpu())
            count += batch.shape[0]

        all_means = torch.stack(means)
        all_stds = torch.stack(stds)
        return {
            "mean": all_means.mean(dim=0),
            "std": all_stds.mean(dim=0),
        }
