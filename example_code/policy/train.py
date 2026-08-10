"""
policy/train.py
---------------
Training script for both HierarchicalPolicy and MonolithicPolicy.

Usage:
    # Train hierarchical policy on source-domain demos
    python policy/train.py --model hierarchical --demos data/demos_source.hdf5

    # Train monolithic baseline
    python policy/train.py --model monolithic --demos data/demos_source.hdf5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from policy.models import HierarchicalPolicy, MonolithicPolicy

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
FORCE_WINDOW = 20   # timesteps of force history for phase classifier


class DemoDataset(Dataset):
    """
    Loads all demonstrations from an HDF5 file into memory and returns
    individual timesteps (not full episodes).
    """

    def __init__(self, hdf5_path: str):
        self.samples: list[dict] = []

        with h5py.File(hdf5_path, "r") as f:
            demo_keys = sorted(f.keys())
            for key in demo_keys:
                grp     = f[key]
                rgb     = grp["rgb"][:]         # (T, H, W, 3)
                depth   = grp["depth"][:]       # (T, H, W, 1)
                proprio = grp["proprio"][:]     # (T, 21)
                force   = grp["force"][:]       # (T, 6)
                actions = grp["actions"][:]     # (T, 8)
                phases  = grp["phases"][:]      # (T,)

                T = rgb.shape[0]
                for t in range(T):
                    # build force history window ending at t
                    start = max(0, t - FORCE_WINDOW + 1)
                    window = force[start:t + 1]  # (<=FORCE_WINDOW, 6)
                    # pad if needed
                    if window.shape[0] < FORCE_WINDOW:
                        pad = np.zeros(
                            (FORCE_WINDOW - window.shape[0], 6),
                            dtype=np.float32
                        )
                        window = np.concatenate([pad, window], axis=0)

                    self.samples.append({
                        "rgb":          rgb[t].astype(np.uint8),
                        "depth":        depth[t].astype(np.float32),
                        "proprio":      proprio[t].astype(np.float32),
                        "force":        force[t].astype(np.float32),
                        "force_window": window.astype(np.float32),
                        "action":       actions[t, :7].astype(np.float32),
                        "phase":        int(phases[t]),
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def collate_fn(batch: list[dict]) -> dict:
    out: dict = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        if k == "phase":
            out[k] = torch.tensor(vals, dtype=torch.long)
        else:
            out[k] = torch.from_numpy(np.stack(vals))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Training loops
# ─────────────────────────────────────────────────────────────────────────────

def train_hierarchical(args):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = DemoDataset(args.demos)
    n_val   = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=0)
    val_loader   = DataLoader(val_set, batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=0)

    model = HierarchicalPolicy().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr,
                               weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * len(train_loader))

    best_val = float("inf")
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # ── train ─────────────────────────────────────────────────────────
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} train",
                          leave=False):
            batch  = {k: v.to(device) for k, v in batch.items()}
            losses = model(
                obs          = batch,
                phase_labels = batch["phase"],
                actions      = batch["action"],
            )
            loss = losses["total"]
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            train_losses.append(loss.item())

        # ── validate ──────────────────────────────────────────────────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch  = {k: v.to(device) for k, v in batch.items()}
                losses = model(
                    obs          = batch,
                    phase_labels = batch["phase"],
                    actions      = batch["action"],
                )
                val_losses.append(losses["total"].item())

        tr = np.mean(train_losses)
        vl = np.mean(val_losses)
        print(f"Epoch {epoch:03d}  train={tr:.4f}  val={vl:.4f}")

        if vl < best_val:
            best_val = vl
            ckpt = Path(args.ckpt_dir) / "hierarchical_best.pt"
            torch.save(model.state_dict(), ckpt)
            print(f"  → saved checkpoint ({ckpt})")

    print(f"Training done. Best val loss: {best_val:.4f}")


def train_monolithic(args):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = DemoDataset(args.demos)
    n_val   = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=0)
    val_loader   = DataLoader(val_set, batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=0)

    model = MonolithicPolicy().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr,
                               weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * len(train_loader))

    best_val = float("inf")
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} train",
                          leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            obs   = {k: batch[k] for k in
                     ("rgb", "depth", "proprio", "force")}
            loss  = model.loss(obs, batch["action"])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                obs   = {k: batch[k] for k in
                         ("rgb", "depth", "proprio", "force")}
                loss  = model.loss(obs, batch["action"])
                val_losses.append(loss.item())

        tr = np.mean(train_losses)
        vl = np.mean(val_losses)
        print(f"Epoch {epoch:03d}  train={tr:.4f}  val={vl:.4f}")

        if vl < best_val:
            best_val = vl
            ckpt = Path(args.ckpt_dir) / "monolithic_best.pt"
            torch.save(model.state_dict(), ckpt)
            print(f"  → saved checkpoint ({ckpt})")

    print(f"Training done. Best val loss: {best_val:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      type=str, default="hierarchical",
                        choices=["hierarchical", "monolithic"])
    parser.add_argument("--demos",      type=str,
                        default="data/demos_source.hdf5")
    parser.add_argument("--ckpt_dir",   type=str, default="checkpoints")
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr",         type=float, default=1e-4)
    args = parser.parse_args()

    if args.model == "hierarchical":
        train_hierarchical(args)
    else:
        train_monolithic(args)
