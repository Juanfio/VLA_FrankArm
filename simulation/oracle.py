from pathlib import Path
import numpy as np
import h5py
import argparse

# Allow running from project root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from simulation.miniarm_env import MiniArmEnv

class Oracle():
    def reset(self, sphere_pos: np.ndarray):
        '''Adds as attributes to the Oracle object the position of the sphere.
           Store here the target positions. Those that you do not want to change, but rather reached.'''
        self.sphere_pos = sphere_pos.copy()
    
    def act(self, obs: dict):
        current_ee_pos = obs["ee_pos"]
        target = self.sphere_pos
        action = self._delta_to(current_ee_pos, target)
        return action

    def _delta_to(self, current: np.ndarray, target: np.ndarray):
        "Compute the delta position to then move"
        diff = target - current
        dist = np.linalg.norm(diff)
        # normalise the action. delta only provides the direction of adjustment.
        delta = diff / (np.linalg.norm(diff) + 1e-8) # + 1e-8 controls an error arising from np.linalg.norm(diff) = 0.
        return delta

def collect_demos(n_demos: int, out_path: str, domain: str = "source"):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    env = MiniArmEnv(max_steps=50)
    oracle = Oracle()
    
    rgb_list, depth_list, proprio_list, ee_pos_list, ee_orn_list, actions_list = [], [], [], [], [], []
    sphere_pos_list = []

    collected = 0

    while collected < n_demos:
        obs = env.reset()
        oracle.reset(obs["sphere_pos"])

        rgb, depth, proprio, ee_pos, ee_orn, actions = [], [], [], [], [], []


        done = False
        while not done:
            action = oracle.act(obs) # generates the action as a delta between current end-effector position and sphere position.
            
            rgb.append(obs["rgb"])
            depth.append(obs["depth"])
            proprio.append(obs["proprio"])
            ee_pos.append(obs["ee_pos"])
            ee_orn.append(obs["ee_orn"])
            actions.append(action)

            obs, _, done, info = env.step(action)

        if info["success"]:
            
            rgb_list.append(np.stack(rgb))
            depth_list.append(np.stack(depth))
            proprio_list.append(np.stack(proprio))
            ee_pos_list.append(np.stack(ee_pos))
            ee_orn_list.append(np.stack(ee_orn))
            actions_list.append(np.stack(actions))
            sphere_pos_list.append(obs["sphere_pos"])

            collected += 1
        
        print(f"Current demo: {collected}\t\tSteps: {info['step']}")
    
    env.close()
    
    _save_hdf5(out_path, rgb_list, depth_list, proprio_list, ee_pos_list, ee_orn_list, actions_list, sphere_pos_list)


def _save_hdf5(out_path: str, rgb: list, depth: list, proprio: list, ee_pos: list, ee_orn: list, actions: list, sphere_pos: list):
    print(out_path)
    with h5py.File(out_path, "w") as f:
        for i, (r,d,p,ep,eo,a,sp) in enumerate(zip(rgb, depth, proprio, ee_pos, ee_orn, actions, sphere_pos)):
            grp = f.create_group(f"demo_{i:04d}")
            grp.create_dataset("rgb", data=r, compression="gzip")
            grp.create_dataset("depth", data=d, compression="gzip")
            grp.create_dataset("proprio", data=p, compression="gzip")
            grp.create_dataset("ee_pos", data=ep, compression="gzip")
            grp.create_dataset("ee_orn", data=eo, compression="gzip")
            grp.create_dataset("actions", data=a, compression="gzip")
            grp.create_dataset("sphere_pos", data=sp, compression="gzip")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_demos",      type=int, default=100)
    parser.add_argument("--out_path",     type=str,
                        default="data/demos.hdf5")
    args = parser.parse_args()

    collect_demos(args.n_demos, Path.cwd().parent / args.out_path)