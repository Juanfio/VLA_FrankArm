"""
policy/models.py
----------------
All neural network components.

1. PhaseClassifier       – small transformer over force/torque history
2. SubPolicy             – Diffusion Policy for a single phase
3. HierarchicalPolicy    – phase classifier + 3 sub-policies (proposed model)
4. MonolithicPolicy      – single Diffusion Policy baseline
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ─────────────────────────────────────────────────────────────────────────────
# Shared primitives
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalPosEmb(nn.Module):
    """Standard sinusoidal embedding for diffusion timestep."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half   = self.dim // 2
        emb    = math.log(10000) / (half - 1)
        emb    = torch.exp(torch.arange(half, device=device) * -emb)
        emb    = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, cond_dim: int):
        super().__init__()
        self.fc1   = nn.Linear(in_dim, out_dim)
        self.fc2   = nn.Linear(out_dim, out_dim)
        self.cond  = nn.Linear(cond_dim, out_dim)
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.skip  = (nn.Linear(in_dim, out_dim)
                      if in_dim != out_dim else nn.Identity())

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = F.mish(self.norm1(self.fc1(x))) + self.cond(c)
        h = F.mish(self.norm2(self.fc2(h)))
        return h + self.skip(x)


# ─────────────────────────────────────────────────────────────────────────────
# Encoders
# ─────────────────────────────────────────────────────────────────────────────

class ImageEncoder(nn.Module):
    """Small CNN to encode 64x64 RGB images into a feature vector."""
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2),  nn.ReLU(),   # 31
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),   # 14
            nn.Conv2d(64, 128, 4, stride=2),nn.ReLU(),   # 6
            nn.Conv2d(128, 64, 3),          nn.ReLU(),   # 4
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, out_dim),
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        # rgb: (B, 3, H, W)  values in [0, 1]
        return self.net(rgb)


class DepthEncoder(nn.Module):
    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2),nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride=2),nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 6 * 6, out_dim),
        )

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.net(depth)


class ProprioEncoder(nn.Module):
    def __init__(self, in_dim: int = 21, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ForceEncoder(nn.Module):
    def __init__(self, in_dim: int = 6, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Phase Classifier
# ─────────────────────────────────────────────────────────────────────────────

class PhaseClassifier(nn.Module):
    """
    Classifies current phase (0/1/2) from a window of force/torque readings.

    Input:  force history  (B, T, 6)
    Output: logits         (B, 3)
    """
    def __init__(self, window: int = 20, hidden: int = 64):
        super().__init__()
        self.window = window
        # 1D conv over time acts like a lightweight temporal encoder
        self.temporal = nn.Sequential(
            nn.Conv1d(6, hidden, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # (B, hidden, 1)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden, 32), nn.ReLU(),
            nn.Linear(32, 3),       # 3 phases
        )

    def forward(self, force_window: torch.Tensor) -> torch.Tensor:
        # force_window: (B, T, 6) → permute to (B, 6, T) for Conv1d
        x = rearrange(force_window, "b t c -> b c t")
        x = self.temporal(x)
        return self.head(x)

    def predict(self, force_window: torch.Tensor) -> int:
        """Convenience wrapper; returns integer phase."""
        with torch.no_grad():
            logits = self.forward(force_window)
            return int(logits.argmax(dim=-1).item())


# ─────────────────────────────────────────────────────────────────────────────
# 2. Diffusion noise head (shared by sub-policies and monolithic)
# ─────────────────────────────────────────────────────────────────────────────

class DiffusionNoiseNet(nn.Module):
    """
    Predicts the noise added at diffusion step t given:
      - noisy action x_t   (B, action_dim)
      - condition vector c  (B, cond_dim)
      - diffusion step t    (B,)
    """
    def __init__(self, action_dim: int, cond_dim: int,
                 hidden: int = 256, depth: int = 4):
        super().__init__()
        self.time_emb  = nn.Sequential(
            SinusoidalPosEmb(hidden),
            nn.Linear(hidden, hidden), nn.Mish(),
        )
        dims = [action_dim] + [hidden] * depth
        self.blocks = nn.ModuleList([
            ResidualBlock(dims[i], dims[i + 1], cond_dim + hidden)
            for i in range(len(dims) - 1)
        ])
        self.out = nn.Linear(hidden, action_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        t_emb    = self.time_emb(t)
        cond_cat = torch.cat([cond, t_emb], dim=-1)
        for block in self.blocks:
            x = block(x, cond_cat)
        return self.out(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Sub-policy (one per phase)
# ─────────────────────────────────────────────────────────────────────────────

# Modality sets per phase:
#   0 APPROACH  → RGB + depth + proprio
#   1 GRASP     → proprio + force
#   2 INSERT    → proprio + force  (force signal is critical for insertion)
PHASE_MODALITIES = {
    0: ["rgb", "depth", "proprio"],
    1: ["proprio", "force"],
    2: ["proprio", "force"],
}


class SubPolicy(nn.Module):
    """
    Diffusion Policy conditioned on the modalities relevant to one phase.

    action_dim: number of continuous action dimensions to predict
                (here 7 = [dx,dy,dz,droll,dpitch,dyaw,gripper])
    """

    N_DIFFUSION_STEPS = 50

    def __init__(self, phase_id: int, action_dim: int = 7):
        super().__init__()
        self.phase_id = phase_id
        self.action_dim = action_dim
        self.modalities = PHASE_MODALITIES[phase_id]

        # build encoders for this phase's modalities
        cond_dim = 0
        if "rgb" in self.modalities:
            self.rgb_enc   = ImageEncoder(out_dim=128); cond_dim += 128
        if "depth" in self.modalities:
            self.depth_enc = DepthEncoder(out_dim=64);  cond_dim += 64
        if "proprio" in self.modalities:
            self.prop_enc  = ProprioEncoder(out_dim=64);cond_dim += 64
        if "force" in self.modalities:
            self.force_enc = ForceEncoder(out_dim=32);  cond_dim += 32

        self.noise_net = DiffusionNoiseNet(action_dim, cond_dim)

        # linear noise schedule
        betas = torch.linspace(1e-4, 0.02, self.N_DIFFUSION_STEPS)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas",     betas)
        self.register_buffer("alphas",    alphas)
        self.register_buffer("alpha_bar", alpha_bar)

    def _encode(self, obs: dict) -> torch.Tensor:
        parts = []
        if "rgb" in self.modalities:
            rgb = obs["rgb"].float() / 255.0
            if rgb.ndim == 3:
                rgb = rgb.permute(2, 0, 1).unsqueeze(0)
            elif rgb.ndim == 4:
                rgb = rgb.permute(0, 3, 1, 2)
            parts.append(self.rgb_enc(rgb))
        if "depth" in self.modalities:
            d = obs["depth"].float()
            if d.ndim == 3:
                d = d.permute(2, 0, 1).unsqueeze(0)
            elif d.ndim == 4:
                d = d.permute(0, 3, 1, 2)
            parts.append(self.depth_enc(d))
        if "proprio" in self.modalities:
            parts.append(self.prop_enc(obs["proprio"].float()))
        if "force" in self.modalities:
            parts.append(self.force_enc(obs["force"].float()))
        return torch.cat(parts, dim=-1)

    def loss(self, obs: dict, actions: torch.Tensor) -> torch.Tensor:
        """Training loss: predict noise at random diffusion step."""
        B = actions.shape[0]
        cond  = self._encode(obs)
        t_idx = torch.randint(0, self.N_DIFFUSION_STEPS, (B,),
                              device=actions.device)
        ab    = self.alpha_bar[t_idx].unsqueeze(1)
        noise = torch.randn_like(actions)
        x_t   = ab.sqrt() * actions + (1 - ab).sqrt() * noise
        pred  = self.noise_net(x_t, t_idx.float(), cond)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, obs: dict, n_steps: int = 20) -> np.ndarray:
        """DDPM sampling; returns action as numpy array."""
        cond = self._encode(obs)
        B    = cond.shape[0]
        x    = torch.randn(B, self.action_dim, device=cond.device)

        step_ids = torch.linspace(self.N_DIFFUSION_STEPS - 1, 0,
                                  n_steps, dtype=torch.long)
        for t_scalar in step_ids:
            t     = t_scalar.expand(B).float().to(cond.device)
            t_idx = t_scalar.item()
            eps   = self.noise_net(x, t, cond)
            alpha = self.alphas[t_idx]
            ab    = self.alpha_bar[t_idx]
            x     = (x - (1 - alpha) / (1 - ab).sqrt() * eps) / alpha.sqrt()
            if t_idx > 0:
                x += self.betas[t_idx].sqrt() * torch.randn_like(x)

        return x.cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Hierarchical Policy (proposed)
# ─────────────────────────────────────────────────────────────────────────────

class HierarchicalPolicy(nn.Module):
    """
    PhaseClassifier gates between three SubPolicies.
    At inference time, maintains a rolling force window for the classifier.
    """

    def __init__(self, action_dim: int = 7, classifier_window: int = 20):
        super().__init__()
        self.classifier = PhaseClassifier(window=classifier_window)
        self.sub_policies = nn.ModuleList([
            SubPolicy(phase_id=i, action_dim=action_dim) for i in range(3)
        ])
        self.classifier_window = classifier_window
        self._force_history: list = []

    def reset_episode(self):
        self._force_history = []

    def _update_force_history(self, force: np.ndarray):
        self._force_history.append(force.copy())
        if len(self._force_history) > self.classifier_window:
            self._force_history.pop(0)

    def _get_force_window_tensor(self, device) -> torch.Tensor:
        W  = self.classifier_window
        fh = self._force_history
        # pad with zeros if history shorter than window
        padded = ([np.zeros(6)] * (W - len(fh))) + fh
        arr    = np.stack(padded, axis=0)            # (W, 6)
        return torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(device)

    def act(self, obs: dict, device: str = "cpu") -> tuple[np.ndarray, int]:
        """
        Returns (action_7d, predicted_phase).
        obs values are numpy arrays.
        """
        force = obs["force"]
        self._update_force_history(force)

        fw    = self._get_force_window_tensor(device)
        phase = self.classifier.predict(fw)

        sub   = self.sub_policies[phase]

        # wrap obs as tensors with batch dim
        obs_t = {}
        for k, v in obs.items():
            if k in ("rgb", "depth", "proprio", "force"):
                obs_t[k] = torch.tensor(v, dtype=torch.float32
                                        ).unsqueeze(0).to(device)

        action = sub.sample(obs_t, n_steps=10).squeeze(0)
        return action, phase

    def forward(self, obs: dict, phase_labels: torch.Tensor,
                actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Training forward pass.
        Returns dict with 'classifier_loss', 'policy_loss_ph0/1/2', 'total'.
        """
        # force window for classifier (B, T, 6) — pre-assembled in dataset
        clf_logits = self.classifier(obs["force_window"])
        clf_loss   = F.cross_entropy(clf_logits, phase_labels.long())

        policy_losses = {}
        for ph in range(3):
            mask = (phase_labels == ph)
            if mask.sum() == 0:
                policy_losses[f"ph{ph}"] = torch.tensor(0.0)
                continue
            obs_ph = {k: v[mask] for k, v in obs.items()
                      if k in ("rgb", "depth", "proprio", "force")}
            loss_ph = self.sub_policies[ph].loss(obs_ph, actions[mask])
            policy_losses[f"ph{ph}"] = loss_ph

        policy_loss = sum(policy_losses.values()) / 3.0
        total       = clf_loss + policy_loss

        return {"classifier_loss": clf_loss,
                "policy_loss": policy_loss,
                "total": total,
                **{f"policy_loss_{k}": v
                   for k, v in policy_losses.items()}}


# ─────────────────────────────────────────────────────────────────────────────
# 5. Monolithic Baseline
# ─────────────────────────────────────────────────────────────────────────────

class MonolithicPolicy(nn.Module):
    """
    Single Diffusion Policy using ALL modalities (baseline for comparison).
    """
    N_DIFFUSION_STEPS = 50

    def __init__(self, action_dim: int = 7):
        super().__init__()
        self.action_dim = action_dim
        self.rgb_enc    = ImageEncoder(out_dim=128)
        self.depth_enc  = DepthEncoder(out_dim=64)
        self.prop_enc   = ProprioEncoder(out_dim=64)
        self.force_enc  = ForceEncoder(out_dim=32)
        cond_dim        = 128 + 64 + 64 + 32

        self.noise_net  = DiffusionNoiseNet(action_dim, cond_dim)

        betas     = torch.linspace(1e-4, 0.02, self.N_DIFFUSION_STEPS)
        alphas    = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas",     betas)
        self.register_buffer("alphas",    alphas)
        self.register_buffer("alpha_bar", alpha_bar)

    def _encode(self, obs: dict) -> torch.Tensor:
        rgb = obs["rgb"].float() / 255.0
        if rgb.ndim == 4:
            rgb = rgb.permute(0, 3, 1, 2)
        d = obs["depth"].float()
        if d.ndim == 4:
            d = d.permute(0, 3, 1, 2)
        return torch.cat([
            self.rgb_enc(rgb),
            self.depth_enc(d),
            self.prop_enc(obs["proprio"].float()),
            self.force_enc(obs["force"].float()),
        ], dim=-1)

    def loss(self, obs: dict, actions: torch.Tensor) -> torch.Tensor:
        B = actions.shape[0]
        cond  = self._encode(obs)
        t_idx = torch.randint(0, self.N_DIFFUSION_STEPS, (B,),
                              device=actions.device)
        ab    = self.alpha_bar[t_idx].unsqueeze(1)
        noise = torch.randn_like(actions)
        x_t   = ab.sqrt() * actions + (1 - ab).sqrt() * noise
        pred  = self.noise_net(x_t, t_idx.float(), cond)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, obs: dict, n_steps: int = 20) -> np.ndarray:
        cond = self._encode(obs)
        B    = cond.shape[0]
        x    = torch.randn(B, self.action_dim, device=cond.device)
        step_ids = torch.linspace(self.N_DIFFUSION_STEPS - 1, 0,
                                  n_steps, dtype=torch.long)
        for t_scalar in step_ids:
            t     = t_scalar.expand(B).float().to(cond.device)
            t_idx = t_scalar.item()
            eps   = self.noise_net(x, t, cond)
            alpha = self.alphas[t_idx]
            ab    = self.alpha_bar[t_idx]
            x     = (x - (1 - alpha) / (1 - ab).sqrt() * eps) / alpha.sqrt()
            if t_idx > 0:
                x += self.betas[t_idx].sqrt() * torch.randn_like(x)
        return x.cpu().numpy()
