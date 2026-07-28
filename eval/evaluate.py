"""
eval/evaluate.py
----------------
Evaluates trained policies in PyBullet across source and target domains.
Produces the results table for the paper.

Usage:
    python eval/evaluate.py \
        --hierarchical_ckpt checkpoints/hierarchical_best.pt \
        --monolithic_ckpt   checkpoints/monolithic_best.pt \
        --n_episodes 50
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from tqdm import tqdm

from sim.pybullet_env import PegInsertionEnv
from policy.models import HierarchicalPolicy, MonolithicPolicy


# ─────────────────────────────────────────────────────────────────────────────

def obs_to_tensor(obs: dict, device: str = "cpu") -> dict:
    """Convert numpy obs dict to torch tensors with batch dim=1."""
    out = {}
    for k in ("rgb", "depth", "proprio", "force"):
        out[k] = torch.tensor(obs[k], dtype=torch.float32).unsqueeze(0).to(device)
    return out


def evaluate_hierarchical(model: HierarchicalPolicy,
                           domain: str,
                           n_episodes: int,
                           device: str,
                           max_steps: int = 300) -> dict:
    env = PegInsertionEnv(render=False, domain=domain, max_steps=max_steps)
    model.eval()

    successes        = 0
    phase_successes  = {0: 0, 1: 0, 2: 0}   # how many eps reach each phase
    phase_correct    = []                     # classifier accuracy
    inference_times  = []

    for ep in tqdm(range(n_episodes), desc=f"[{domain}] hierarchical"):
        obs = env.reset()
        model.reset_episode()
        done = False
        max_phase_reached  = 0
        ep_phase_correct   = []

        while not done:
            t0     = time.perf_counter()
            action, pred_phase = model.act(obs, device=device)
            inference_times.append(time.perf_counter() - t0)

            true_phase = obs["phase"]
            ep_phase_correct.append(int(pred_phase == true_phase))
            max_phase_reached = max(max_phase_reached, true_phase)

            # build full 8-d action for env
            full_action         = np.zeros(8, dtype=np.float32)
            full_action[:7]     = action
            full_action[7]      = float(pred_phase)

            obs, _, done, info  = env.step(full_action)

        for ph in range(max_phase_reached + 1):
            phase_successes[ph] += 1

        if info["success"]:
            successes += 1

        phase_correct.extend(ep_phase_correct)

    env.close()
    return {
        "success_rate":    successes / n_episodes,
        "phase_acc":       np.mean(phase_correct),
        "phase_reach": {ph: v / n_episodes
                        for ph, v in phase_successes.items()},
        "mean_inference_ms": np.mean(inference_times) * 1000,
    }


def evaluate_monolithic(model: MonolithicPolicy,
                         domain: str,
                         n_episodes: int,
                         device: str,
                         max_steps: int = 300) -> dict:
    env = PegInsertionEnv(render=False, domain=domain, max_steps=max_steps)
    model.eval()

    successes       = 0
    phase_successes = {0: 0, 1: 0, 2: 0}
    inference_times = []

    for ep in tqdm(range(n_episodes), desc=f"[{domain}] monolithic"):
        obs  = env.reset()
        done = False
        max_phase_reached = 0

        while not done:
            obs_t = obs_to_tensor(obs, device)

            t0     = time.perf_counter()
            action = model.sample(obs_t, n_steps=10).squeeze(0)
            inference_times.append(time.perf_counter() - t0)

            max_phase_reached = max(max_phase_reached, obs["phase"])

            full_action     = np.zeros(8, dtype=np.float32)
            full_action[:7] = action
            full_action[7]  = float(obs["phase"])

            obs, _, done, info = env.step(full_action)

        for ph in range(max_phase_reached + 1):
            phase_successes[ph] += 1

        if info["success"]:
            successes += 1

    env.close()
    return {
        "success_rate":    successes / n_episodes,
        "phase_reach": {ph: v / n_episodes
                        for ph, v in phase_successes.items()},
        "mean_inference_ms": np.mean(inference_times) * 1000,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Perturbation experiment
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_with_perturbation(model, model_type: str,
                                n_episodes: int, device: str) -> dict:
    """
    At step 50, apply a 5mm lateral perturbation to the peg position
    (not seen during training). Tests robustness to distribution shift.
    """
    import pybullet as p

    env = PegInsertionEnv(render=False, domain="source", max_steps=300)
    if model_type == "hierarchical":
        model.reset_episode()

    successes = 0

    for ep in tqdm(range(n_episodes), desc="perturbation eval"):
        obs  = env.reset()
        done = False
        step = 0
        perturbed = False

        if model_type == "hierarchical":
            model.reset_episode()

        while not done:
            # apply perturbation at step 50
            if step == 50 and not perturbed:
                peg_pos, peg_orn = p.getBasePositionAndOrientation(
                    env.peg_id, physicsClientId=env.client)
                new_pos = list(peg_pos)
                new_pos[1] += 0.005   # 5 mm lateral shift
                p.resetBasePositionAndOrientation(
                    env.peg_id, new_pos, peg_orn,
                    physicsClientId=env.client)
                perturbed = True

            if model_type == "hierarchical":
                action, pred_phase = model.act(obs, device=device)
                full_action        = np.zeros(8, dtype=np.float32)
                full_action[:7]    = action
                full_action[7]     = float(pred_phase)
            else:
                obs_t = obs_to_tensor(obs, device)
                action          = model.sample(obs_t, n_steps=10).squeeze(0)
                full_action     = np.zeros(8, dtype=np.float32)
                full_action[:7] = action
                full_action[7]  = float(obs["phase"])

            obs, _, done, info = env.step(full_action)
            step += 1

        if info["success"]:
            successes += 1

    env.close()
    return {"perturbed_success_rate": successes / n_episodes}


# ─────────────────────────────────────────────────────────────────────────────
# Main: produce results table
# ─────────────────────────────────────────────────────────────────────────────

def print_table(results: dict):
    print("\n" + "=" * 65)
    print(f"{'Condition':<30} {'Hierarch.':>10} {'Monolith.':>10}")
    print("=" * 65)

    keys = [
        ("Source success rate",   "source",     "success_rate"),
        ("Target success rate",   "target",     "success_rate"),
        ("Perturbed success rate","perturb",    "perturbed_success_rate"),
        ("Phase classifier acc.", "source",     "phase_acc"),
        ("Inference time (ms)",   "source",     "mean_inference_ms"),
    ]

    for label, domain, metric in keys:
        h_val = results["hierarchical"].get(domain, {}).get(metric, "—")
        m_val = results["monolithic"].get(domain, {}).get(metric, "—")
        h_str = f"{h_val:.3f}" if isinstance(h_val, float) else str(h_val)
        m_str = f"{m_val:.3f}" if isinstance(m_val, float) else str(m_val)
        print(f"{label:<30} {h_str:>10} {m_str:>10}")

    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hierarchical_ckpt", type=str,
                        default="checkpoints/hierarchical_best.pt")
    parser.add_argument("--monolithic_ckpt",   type=str,
                        default="checkpoints/monolithic_best.pt")
    parser.add_argument("--n_episodes",        type=int, default=50)
    parser.add_argument("--device",            type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device

    # load models
    h_model = HierarchicalPolicy()
    h_model.load_state_dict(torch.load(args.hierarchical_ckpt,
                                        map_location=device))
    h_model.to(device).eval()

    m_model = MonolithicPolicy()
    m_model.load_state_dict(torch.load(args.monolithic_ckpt,
                                        map_location=device))
    m_model.to(device).eval()

    results = {
        "hierarchical": {},
        "monolithic":   {},
    }

    for domain in ["source", "target"]:
        results["hierarchical"][domain] = evaluate_hierarchical(
            h_model, domain, args.n_episodes, device)
        results["monolithic"][domain]   = evaluate_monolithic(
            m_model, domain, args.n_episodes, device)

    # perturbation
    ph = evaluate_with_perturbation(h_model, "hierarchical",
                                     args.n_episodes, device)
    pm = evaluate_with_perturbation(m_model, "monolithic",
                                     args.n_episodes, device)
    results["hierarchical"]["perturb"] = ph
    results["monolithic"]["perturb"]   = pm

    print_table(results)
