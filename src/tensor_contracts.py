"""Shared tensor contracts for the Minecraft world model pipeline."""

from __future__ import annotations

import torch


def assert_frame_tensor(x: torch.Tensor, *, size: int | None = None, name: str = "frames") -> None:
    """Validate RGB frame tensors in [0, 1].

    Accepts either [C, H, W], [T, C, H, W], or [B, T, C, H, W].
    """
    if x.dim() not in (3, 4, 5):
        raise ValueError(f"{name} must be rank 3, 4, or 5; got shape {tuple(x.shape)}")
    channel_dim = -3
    if x.shape[channel_dim] != 3:
        raise ValueError(f"{name} must have 3 RGB channels; got shape {tuple(x.shape)}")
    if size is not None and (x.shape[-2] != size or x.shape[-1] != size):
        raise ValueError(f"{name} must be {size}x{size}; got shape {tuple(x.shape)}")
    if torch.is_floating_point(x) and (x.detach().amin() < -1e-4 or x.detach().amax() > 1.0001):
        raise ValueError(f"{name} must be normalized to [0, 1]")


def assert_latent_tensor(
    x: torch.Tensor,
    *,
    grid_size: int | None = None,
    name: str = "latents",
) -> None:
    """Validate VAE latent tensors.

    Accepts either [C, H, W], [T, C, H, W], or [B, T, C, H, W].
    """
    if x.dim() not in (3, 4, 5):
        raise ValueError(f"{name} must be rank 3, 4, or 5; got shape {tuple(x.shape)}")
    channel_dim = -3
    if x.shape[channel_dim] != 4:
        raise ValueError(f"{name} must have 4 latent channels; got shape {tuple(x.shape)}")
    if grid_size is not None and (x.shape[-2] != grid_size or x.shape[-1] != grid_size):
        raise ValueError(f"{name} must use a {grid_size}x{grid_size} latent grid; got shape {tuple(x.shape)}")


def assert_action_tensor(
    x: torch.Tensor,
    *,
    sequence_length: int | None = None,
    num_actions: int = 29,
    name: str = "actions",
) -> None:
    """Validate integer action-token tensors."""
    if x.dtype != torch.long:
        raise ValueError(f"{name} must use dtype torch.long; got {x.dtype}")
    if sequence_length is not None and x.shape[-1] != sequence_length:
        raise ValueError(f"{name} must have sequence length {sequence_length}; got shape {tuple(x.shape)}")
    if x.numel() and (x.detach().amin() < 0 or x.detach().amax() >= num_actions):
        raise ValueError(f"{name} must contain token ids in [0, {num_actions - 1}]")
