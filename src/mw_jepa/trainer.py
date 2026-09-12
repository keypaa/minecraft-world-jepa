import time
from pathlib import Path
import os
import queue
import shutil
import threading
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import autocast

from mw_jepa.tensor_contracts import assert_action_tensor, assert_frame_tensor, assert_latent_tensor
from mw_jepa.vae import encode_frames

try:
    from tqdm import tqdm
except ImportError:  # minimal envs: fall back to silent iteration
    tqdm = None


def collate_stream(batch):
    """Collate function for WorldModelStream (handles frames or pre-computed latents)."""
    actions = torch.stack([b["actions"] for b in batch], dim=0)
    if "latents" in batch[0]:
        latents = torch.stack([b["latents"] for b in batch], dim=0)
        return {"latents": latents, "actions": actions}
    frames = torch.stack([b["frames"] for b in batch], dim=0)
    return {"frames": frames, "actions": actions}


class Trainer:
    """Training engine for world model with streaming data."""

    def __init__(self, model, vae, config: dict, ckpt_dir: Path):
        self.model = model
        self.vae = vae.eval() if vae is not None else None
        self.config = config
        self.ckpt_dir = ckpt_dir
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        train_cfg = config.get("training", {})
        data_cfg = config.get("data", {})
        model_cfg = config.get("model", {})
        self.target_size = data_cfg.get("target_size", 256)
        self.latent_grid_size = model_cfg.get("latent_grid_size", self.target_size // 8)
        assert self.target_size // 8 == self.latent_grid_size

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.get("lr", 3e-4),
            weight_decay=train_cfg.get("weight_decay", config.get("weight_decay", 0.05)),
            betas=tuple(train_cfg.get("betas", (0.9, 0.95))),
        )
        # heuristic: epochs*1000 steps; pass explicit T_max via config["training"]["t_max"] if set
        t_max = train_cfg.get("t_max", max(1, config.get("epochs", 10) * 1000))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max)

        self.epoch = 0
        self.step = 0
        self.best_loss = float("inf")
        self.loss_history = []  # tracked per-epoch for diagnostics
        # Crash safety: epoch-only checkpointing loses hours on ~8K-step
        # epochs. Save latest.pt every N steps + optionally mirror to Hub.
        self.save_every_steps = int(train_cfg.get("save_every_steps", 200))
        # Wall-clock backstop: checkpoint at least this often even if steps
        # stall (network/dataloader). 0 disables the time trigger.
        self.save_every_seconds = float(train_cfg.get("save_every_seconds", 300))
        self._last_save_time = None
        self.hf_repo_id = train_cfg.get("hf_repo_id") or os.environ.get("HF_HUB_REPO")
        # Hub push cadence decoupled from local saves: push every Kth
        # periodic save (uploads are ~3.5GB and synchronous). 1 = every save.
        self.hf_push_every = max(1, int(train_cfg.get("hf_push_every", 1)))
        self._periodic_saves = 0
        self.run_name = config.get("run_name", "run")
        self._hub_warned = False
        # Async Hub mirror: uploads run on a daemon worker so the training
        # loop never stalls on network. Snapshots keep Hub == local content
        # (no torn reads if the next save overwrites latest.pt mid-upload).
        # Bounded queue (2 snapshots ≈ 7GB worst case) is the backpressure:
        # a fallen-behind uploader skips, never grows disk unboundedly.
        self._hub_queue: queue.Queue = queue.Queue()
        self._hub_thread = None
        if self.hf_repo_id:
            self._hub_thread = threading.Thread(target=self._hub_worker, daemon=True)
            self._hub_thread.start()

    def train(self, stream, batch_size=8, num_epochs=10, steps_per_epoch=None, max_steps=None):
        self.model.train()
        num_patches = getattr(self.model, "num_patches", 256)

        for epoch in range(self.epoch, num_epochs):
            loader = DataLoader(
                stream,
                batch_size=batch_size,
                num_workers=0,
                pin_memory=True,
                collate_fn=collate_stream,
                drop_last=True,
            )

            epoch_loss = 0.0
            num_batches = 0
            tokens_processed = 0
            epoch_start = time.time()
            if self._last_save_time is None:
                self._last_save_time = epoch_start

            use_bar = tqdm is not None
            pbar = (
                tqdm(
                    loader,
                    desc=f"Epoch {epoch + 1}/{num_epochs}",
                    unit="step",
                    total=steps_per_epoch,
                    mininterval=1.0,
                    leave=True,
                )
                if use_bar
                else loader
            )
            for batch in pbar:
                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break
                actions = batch["actions"].cuda()
                assert_action_tensor(actions, name="batch actions")

                if "latents" in batch:
                    latents = batch["latents"].cuda()
                    assert_latent_tensor(latents, grid_size=self.latent_grid_size, name="batch latents")
                    B, T = latents.shape[:2]
                else:
                    if self.vae is None:
                        raise ValueError("Frame batches require a VAE. Enable precomputed latents or pass a VAE.")
                    frames = batch["frames"].cuda()
                    assert_frame_tensor(frames, size=self.target_size, name="batch frames")
                    B, T = frames.shape[:2]
                    with torch.no_grad():
                        flat_frames = frames.view(B * T, 3, self.target_size, self.target_size)
                        latents = encode_frames(self.vae, flat_frames).detach()
                        latents = latents.view(B, T, 4, self.latent_grid_size, self.latent_grid_size)

                input_latents = latents[:, :-1, :, :, :]
                target_latent = latents[:, -1, :, :, :]
                input_actions = actions[:, :-1]
                assert_action_tensor(input_actions, sequence_length=T - 1, name="input actions")

                tokens_processed += B * (T - 1) * num_patches

                self.optimizer.zero_grad(set_to_none=True)

                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred_latent = self.model(input_latents, input_actions)
                    loss = F.mse_loss(pred_latent, target_latent)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()

                if torch.isnan(loss) or torch.isinf(loss):
                    raise ValueError(f"Loss diverged at step {self.step}: {loss.item()}")

                epoch_loss += loss.item()
                num_batches += 1
                self.step += 1
                if use_bar:
                    elapsed = time.time() - epoch_start
                    pbar.set_postfix(
                        {
                            "loss": f"{epoch_loss / num_batches:.4f}",
                            "tok/s": f"{tokens_processed / elapsed:.0f}" if elapsed > 0 else "n/a",
                            "GiB": (
                                f"{torch.cuda.memory_reserved() / (1024 ** 3):.1f}"
                                if torch.cuda.is_available()
                                else "n/a"
                            ),
                        }
                    )
                if max_steps is not None and self.step >= max_steps:
                    break
                if self.save_every_steps > 0 and self.step % self.save_every_steps == 0:
                    self.save_latest(epoch_loss / num_batches)
                    self._periodic_saves += 1
                    self._last_save_time = time.time()
                    if self._periodic_saves % self.hf_push_every == 0:
                        self.maybe_push_to_hub()
                elif (
                    self.save_every_seconds > 0
                    and self._last_save_time is not None
                    and (time.time() - self._last_save_time) >= self.save_every_seconds
                ):
                    # Step trigger missed (stalls) but wall clock says save.
                    self.save_latest(epoch_loss / num_batches)
                    self._periodic_saves += 1
                    self._last_save_time = time.time()
                    if self._periodic_saves % self.hf_push_every == 0:
                        self.maybe_push_to_hub()

            if use_bar:
                pbar.close()

            if num_batches == 0:
                raise ValueError("Training stream produced no batches")

            epoch_time = time.time() - epoch_start
            steps_per_sec = num_batches / epoch_time if epoch_time > 0 else 0
            tokens_per_sec = tokens_processed / epoch_time if epoch_time > 0 else 0
            gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0

            avg_loss = epoch_loss / num_batches
            self.epoch = epoch
            self.loss_history.append(avg_loss)
            print(
                f"Epoch {epoch + 1}/{num_epochs} — Loss: {avg_loss:.6f} | "
                f"{steps_per_sec:.2f} st/s | {tokens_per_sec:.0f} tok/s | "
                f"GPU: {gpu_mem:.1f} GiB",
                flush=True,
            )
            torch.cuda.reset_peak_memory_stats()
            self.save_checkpoint(avg_loss)
            self.flush_hub()
            if max_steps is not None and self.step >= max_steps:
                break

    def save_checkpoint(self, loss):
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": self.epoch,
            "step": self.step,
            "loss": loss,
            "best_loss": min(loss, self.best_loss),
            "config": self.config,
        }
        path = self.ckpt_dir / f"epoch_{self.epoch:04d}_loss_{loss:.6f}.pt"
        torch.save(ckpt, path)

        if loss < self.best_loss:
            self.best_loss = loss
            torch.save(ckpt, self.ckpt_dir / "best.pt")

        torch.save(ckpt, self.ckpt_dir / "latest.pt")

    def save_latest(self, running_avg_loss: float):
        """Overwrite latest.pt with current state (cheap periodic safety net).

        Unlike save_checkpoint(), this writes no epoch file and never touches
        best.pt — a death costs at most save_every_steps of work.
        """
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": self.epoch,
            "step": self.step,
            "loss": running_avg_loss,
            "best_loss": self.best_loss,
            "config": self.config,
        }
        torch.save(ckpt, self.ckpt_dir / "latest.pt")
        print(f"  [ckpt] step {self.step} → latest.pt (loss {running_avg_loss:.4f})", flush=True)

    def maybe_push_to_hub(self):
        """Enqueue a Hub mirror of latest.pt (async, best-effort, never fatal).

        Enabled by training.hf_repo_id in config or HF_HUB_REPO env var.
        Auth via HF_TOKEN env var (HfApi picks it up automatically).
        Snapshots to latest_push.pt first so the upload is byte-identical to
        the local checkpoint even if the next save lands mid-upload.
        If the uploader is backlogged (>2 pending), the push is skipped with
        a warning — local latest.pt is always the source of truth.
        """
        if not self.hf_repo_id or self._hub_thread is None:
            return
        if self._hub_queue.qsize() >= 2:
            if not self._hub_warned:
                print("  [hub] uploader backlogged, skipping push (local ckpt unaffected)", flush=True)
                self._hub_warned = True
            return
        try:
            snapshot = self.ckpt_dir / f"latest_push_{self.step:07d}.pt"
            shutil.copy2(self.ckpt_dir / "latest.pt", snapshot)
            self._hub_queue.put(snapshot)
            print(f"  [hub] queued step {self.step} for upload", flush=True)
        except Exception as e:
            print(f"  [hub] snapshot failed (continuing locally): {e}", flush=True)

    def _hub_worker(self):
        """Daemon: upload enqueued snapshots to latest.pt on the Hub."""
        from huggingface_hub import HfApi

        api = HfApi()
        while True:
            snapshot = self._hub_queue.get()
            try:
                api.upload_file(
                    path_or_fileobj=str(snapshot),
                    path_in_repo=f"{self.run_name}/latest.pt",
                    repo_id=self.hf_repo_id,
                    repo_type="model",
                )
                print(f"  [hub] mirrored → {self.hf_repo_id}:{self.run_name}/latest.pt", flush=True)
            except Exception as e:
                print(f"  [hub] upload failed (local ckpt unaffected): {e}", flush=True)
            finally:
                try:
                    os.remove(snapshot)
                except OSError:
                    pass
                self._hub_queue.task_done()

    def flush_hub(self):
        """Block until all queued uploads finish. Call at epoch end."""
        if self._hub_thread is not None:
            self._hub_queue.join()

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location="cuda", weights_only=False)  # trusted local checkpoint
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.epoch = ckpt["epoch"]
        self.step = ckpt["step"]
        if ckpt.get("best_loss") is not None:
            self.best_loss = ckpt["best_loss"]
        print(f"Resumed from epoch {self.epoch} (step {self.step}, loss {ckpt['loss']:.6f})")


def get_training_schedule():
    return [
        {"context": 4,  "epochs": 2,  "lr": 3e-4,  "batch_size": 64,  "lr_reset": True},
        {"context": 8,  "epochs": 3,  "lr": 2e-4,  "batch_size": 32,  "lr_reset": True},
        {"context": 16, "epochs": 5,  "lr": 1e-4,  "batch_size": 16,  "lr_reset": True},
        {"context": 32, "epochs": 10, "lr": 5e-5,  "batch_size": 8,   "lr_reset": True},
    ]
