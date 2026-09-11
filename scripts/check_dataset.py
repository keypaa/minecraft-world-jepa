"""Check TESS VLA dataset structure for Phase 2 planning."""
from datasets import load_dataset
import itertools
from collections import Counter
from huggingface_hub import list_repo_files, HfApi

# Count trajectories and shard density
ds = load_dataset("TESS-Computer/minecraft-vla-stage1", split="train", streaming=True)

first_100 = list(itertools.islice(ds, 100))
unique_ids = set(ex["video_id"] for ex in first_100)
print(f"First 100 examples: {len(unique_ids)} unique trajectories")

# Parse action format
sample = first_100[0]["action"]
print(f"Action format: {sample[:120]}")
parts = sample.replace("<|action_start|>", "").replace("<|action_end|>", "").strip()
mouse_part = parts.split(";")[0].strip()
chunks = [c.strip() for c in parts.split(";")[1:]]
print(f"  Mouse: {mouse_part}")
print(f"  Chunks ({len(chunks)}): {chunks}")

# Image size
img = first_100[0]["image"]
print(f"Image per frame: {len(img)} bytes (JPEG)")

# Parquet structure
api = HfApi()
files = list(api.list_repo_files("TESS-Computer/minecraft-vla-stage1", repo_type="dataset"))
parquet_files = sorted(f for f in files if f.endswith(".parquet"))
print(f"\nTotal parquet shards: {len(parquet_files)}")
print(f"Shards 0-4: {parquet_files[:5]}")

import pyarrow.parquet as pq
import io, requests
url = f"https://huggingface.co/datasets/TESS-Computer/minecraft-vla-stage1/resolve/main/{parquet_files[0]}"
resp = requests.get(url, stream=True)
raw = resp.raw.read()
table = pq.read_table(io.BytesIO(raw))
print(f"Shard 0: {table.num_rows} rows, {len(raw)/1e6:.0f}MB")
print(f"Columns: {table.column_names}")

# Count trajectories per shard
video_ids = table.column("video_id").to_pylist()
trajs = set(video_ids)
print(f"Trajectories in shard 0: {len(trajs)}")

# Frame rate: original VPT is 20 FPS, TESS says 5 FPS
# Check frame_idx range
frame_idxs = table.column("frame_idx").to_pylist()
print(f"Frame idx range: {min(frame_idxs)} - {max(frame_idxs)}")
