"""Local inference engine. Context 16, re-ground every 32 frames."""

import torch
from mw_jepa.vae import encode_frames, decode_latents


class InferenceEngine:
    def __init__(self, model, vae, mean=None, std=None, context=16, device="cuda"):
        self.model = model.eval().to(device)
        self.vae = vae.eval().to(device)
        self.mean, self.std = mean, std
        self.context = context
        self.device = torch.device(device)
        self.buf = []

    def reset(self, seed_frame: torch.Tensor):
        with torch.no_grad():
            z = encode_frames(self.vae, seed_frame.to(self.device))
        if self.mean is not None:
            z = (z - self.mean.to(self.device)) / (self.std.to(self.device) + 1e-6)
        self.buf = [z.squeeze(0)]

    def step(self, action_token: int):
        import torch as _t

        hist = _t.stack(self.buf[-self.context :])
        if hist.shape[0] < self.context:
            hist = _t.cat([hist[0:1].expand(self.context - hist.shape[0], -1, -1, -1), hist], dim=0)
        if self.mean is not None:
            pass
        acts = _t.full((1, self.context), action_token, dtype=_t.long, device=self.device)
        with _t.no_grad():
            nxt = self.model(hist.unsqueeze(0), acts)
            raw = nxt
            if self.mean is not None:
                raw = nxt * (self.std.to(self.device) + 1e-6) + self.mean.to(self.device)
            frame = decode_latents(self.vae, raw)
        self.buf.append(nxt.squeeze(0))
        return frame
