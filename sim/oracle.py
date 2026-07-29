"""
sim/oracle.py
-------------
Scripted oracle policy for collecting expert demonstrations.

The oracle knows the ground-truth peg and hole positions (privileged info
that the learned policy will NOT have access to) and outputs the correct
phase-labelled actions at each step.

Usage:
    python sim/oracle.py --n_demos 100 --domain source --out data/demos_source.hdf5
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from sim.pybullet_env import PegInsertionEnv


# ─────────────────────────────────────────────────────────────────────────────
# Oracle logic
# ─────────────────────────────────────────────────────────────────────────────
class OraclePolicy:
    """
    Three-phase scripted controller.
    Returns action = [dx, dy, dz, 0, 0, 0, gripper, phase_id]
    """

    APPROACH_HEIGHT = 0.85      # z above peg to approach
    GRASP_HEIGHT    = 0.65      # z to descend to for grasping
    LIFT_HEIGHT     = 0.90      # z to lift peg to before moving to hole
    INSERT_HEIGHT   = 0.645      # z to lower peg into hole

    POS_THRESH = 0.015          # position tolerance to advance phase

    def reset(self, peg_pos: np.ndarray, hole_pos: np.ndarray):
        self.peg_pos  = peg_pos.copy()
        self.hole_pos = hole_pos.copy()
        self._sub_phase = 0     # internal sub-states within each phase

    def act(self, obs: dict) -> np.ndarray:
        """Return action array (8,)."""
        ee_pos = obs["proprio"][14:17]   # slice ee_pos from proprio
        phase  = obs["phase"]

        # ── PHASE 0: APPROACH ────────────────────────────────────────────────
        if phase == 0:
            if self._sub_phase == 0:
                # move above peg (x,y aligned, z high)
                target = np.array([self.peg_pos[0], self.peg_pos[1],
                                   self.APPROACH_HEIGHT])
                delta, done = self._delta_to(ee_pos, target)
                if done:
                    self._sub_phase = 1
                return self._action(delta, gripper_open=True, phase=0)

            elif self._sub_phase == 1:
                # descend to grasp height
                target = np.array([self.peg_pos[0], self.peg_pos[1],
                                   self.GRASP_HEIGHT])
                delta, done = self._delta_to(ee_pos, target)
                if done:
                    self._sub_phase = 2
                return self._action(delta, gripper_open=True, phase=0)

        # ── PHASE 1: GRASP ───────────────────────────────────────────────────
        if phase == 1 or self._sub_phase == 2:
            if self._sub_phase == 2:
                # close gripper
                self._sub_phase = 3
                return self._action(np.zeros(3), gripper_open=False, phase=1)

            elif self._sub_phase == 3:
                # lift
                target = np.array([self.peg_pos[0], self.peg_pos[1],
                                   self.LIFT_HEIGHT])
                delta, done = self._delta_to(ee_pos, target)
                if done:
                    self._sub_phase = 4
                return self._action(delta, gripper_open=False, phase=1)

        # ── PHASE 2: INSERT ──────────────────────────────────────────────────
        if phase == 2 or self._sub_phase >= 4:
            if self._sub_phase == 4:
                # move over hole
                target = np.array([self.hole_pos[0], self.hole_pos[1],
                                   self.LIFT_HEIGHT])
                delta, done = self._delta_to(ee_pos, target)
                if done:
                    self._sub_phase = 5
                return self._action(delta, gripper_open=False, phase=2)

            elif self._sub_phase == 5:
                # lower into hole
                target = np.array([self.hole_pos[0], self.hole_pos[1],
                                   self.INSERT_HEIGHT])
                delta, done = self._delta_to(ee_pos, target)
                if done:
                    self._sub_phase = 6
                return self._action(delta, gripper_open=False, phase=2)

            elif self._sub_phase == 6:
                # release
                return self._action(np.zeros(3), gripper_open=True, phase=2)

        # fallback: stay still
        return self._action(np.zeros(3), gripper_open=True, phase=phase)

    # ── helpers ───────────────────────────────────────────────────────────────

    def advance_phase(self, obs: dict) -> int:
        """Determine correct phase from sub_phase counter."""
        if self._sub_phase < 2:
            return 0
        elif self._sub_phase < 4:
            return 1
        else:
            return 2

    def _delta_to(self, current: np.ndarray,
                  target: np.ndarray) -> tuple[np.ndarray, bool]:
        diff  = target - current
        dist  = np.linalg.norm(diff)
        done  = dist < self.POS_THRESH
        # normalise to unit sphere then the env scales by 0.02 m/step
        delta = diff / (np.linalg.norm(diff) + 1e-8)
        return delta, done

    @staticmethod
    def _action(delta_pos: np.ndarray, gripper_open: bool,
                phase: int) -> np.ndarray:
        action = np.zeros(8, dtype=np.float32)
        action[:3]  = delta_pos
        action[6]   = 1.0 if gripper_open else 0.0
        action[7]   = float(phase)
        return action


# ─────────────────────────────────────────────────────────────────────────────
# Data collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_demos(n_demos: int, domain: str, out_path: str,
                  render: bool = False) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    env    = PegInsertionEnv(render=render, domain=domain, max_steps=300)
    oracle = OraclePolicy()

    rgb_list, depth_list, proprio_list, force_list = [], [], [], []
    action_list, phase_list, success_list = [], [], []

    collected = 0
    attempts  = 0

    while collected < n_demos:
        attempts += 1
        obs = env.reset()

        # extract privileged ground-truth positions from environment
        import pybullet as p
        peg_pos  = obs["peg_pos"]
        hole_pos = env._hole_pos
        oracle.reset(peg_pos, hole_pos)

        ep_rgb, ep_depth, ep_proprio, ep_force = [], [], [], []
        ep_action, ep_phase = [], []

        done = False
        while not done:
            # override phase in obs with oracle's phase counter
            obs["phase"] = oracle.advance_phase(obs)
            action = oracle.act(obs)

            ep_rgb.append(obs["rgb"])
            ep_depth.append(obs["depth"])
            ep_proprio.append(obs["proprio"])
            ep_force.append(obs["force"])
            ep_action.append(action)
            ep_phase.append(obs["phase"])

            obs, _, done, info = env.step(action)

        if info["success"]:
            rgb_list.append(np.stack(ep_rgb))
            depth_list.append(np.stack(ep_depth))
            proprio_list.append(np.stack(ep_proprio))
            force_list.append(np.stack(ep_force))
            action_list.append(np.stack(ep_action))
            phase_list.append(np.array(ep_phase, dtype=np.int32))
            success_list.append(True)
            collected += 1
            print(f"[{domain}] Demo {collected}/{n_demos} "
                  f"(attempt {attempts}, length {len(ep_rgb)})")

    env.close()
    _save_hdf5(out_path, rgb_list, depth_list, proprio_list,
               force_list, action_list, phase_list)
    print(f"Saved {collected} demos to {out_path} "
          f"(success rate {collected/attempts:.1%})")


def _save_hdf5(path, rgb, depth, proprio, force, actions, phases):
    with h5py.File(path, "w") as f:
        for i, (r, d, pr, ft, a, ph) in enumerate(
                zip(rgb, depth, proprio, force, actions, phases)):
            grp = f.create_group(f"demo_{i:04d}")
            grp.create_dataset("rgb",     data=r,  compression="gzip")
            grp.create_dataset("depth",   data=d,  compression="gzip")
            grp.create_dataset("proprio", data=pr, compression="gzip")
            grp.create_dataset("force",   data=ft, compression="gzip")
            grp.create_dataset("actions", data=a,  compression="gzip")
            grp.create_dataset("phases",  data=ph, compression="gzip")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_demos", type=int, default=100)
    parser.add_argument("--domain",  type=str, default="source",
                        choices=["source", "target"])
    parser.add_argument("--out",     type=str,
                        default="data/demos_source.hdf5")
    parser.add_argument("--render",  action="store_true")
    args = parser.parse_args()

    collect_demos(args.n_demos, args.domain, args.out, args.render)
