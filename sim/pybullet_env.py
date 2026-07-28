"""
sim/pybullet_env.py
-------------------
Pick-and-place-with-insertion environment in PyBullet.

Task:
  1. APPROACH  – move end-effector above the peg object
  2. GRASP     – lower, close gripper, lift
  3. INSERT    – move to hole, lower peg into hole

Observations returned per step:
  - rgb image (64x64x3)
  - depth image (64x64x1)
  - proprioception: [joint_positions(7), joint_velocities(7), ee_pos(3), ee_orn(4)]
  - force_torque: [fx, fy, fz, tx, ty, tz]

The phase label (0/1/2) is returned for supervised phase-classifier training.
"""

import math
import time
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data
import pybullet_data as pd


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
IMG_W, IMG_H = 64, 64
CTRL_HZ      = 20          # control frequency (Hz); one step = 1/CTRL_HZ s
GRAVITY      = -9.81

# Franka Panda joint indices that are actuated (7 arm joints)
PANDA_ARM_JOINTS = [0, 1, 2, 3, 4, 5, 6]
PANDA_FINGER_JOINTS = [9, 10]           # prismatic finger joints
EE_LINK_INDEX   = 11                    # end-effector link

# Domain-randomisation ranges (source vs target domain)
DR_SOURCE = dict(
    table_color_range  = (0.6, 0.7),    # grey-ish
    peg_color_range    = (0.8, 0.9),    # light
    object_mass_range  = (0.05, 0.06),
    friction_range     = (0.4, 0.5),
    light_direction    = ([1, -1, -1], [1, -1, -1]),
)
DR_TARGET = dict(
    table_color_range  = (0.2, 0.9),    # anything
    peg_color_range    = (0.1, 0.95),
    object_mass_range  = (0.03, 0.12),
    friction_range     = (0.2, 0.8),
    light_direction    = ([-1, -1, -1], [1, 1, -1]),
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build a simple peg URDF on-the-fly
# ─────────────────────────────────────────────────────────────────────────────
PEG_URDF_TEMPLATE = """<?xml version="1.0"?>
<robot name="peg">
  <link name="base">
    <visual>
      <geometry><cylinder radius="0.015" length="0.08"/></geometry>
      <origin xyz="0 0 0.04"/>
      <material name="peg_mat">
        <color rgba="{r} {g} {b} 1"/>
      </material>
    </visual>
    <collision>
      <geometry><cylinder radius="0.015" length="0.08"/></geometry>
      <origin xyz="0 0 0.04"/>
    </collision>
    <inertial>
      <mass value="{mass}"/>
      <inertia ixx="1e-5" iyy="1e-5" izz="1e-5" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
</robot>
"""

HOLE_URDF_TEMPLATE = """<?xml version="1.0"?>
<robot name="hole">
  <link name="base">
    <visual>
      <geometry><box size="0.06 0.06 0.04"/></geometry>
      <origin xyz="0 0 0.02"/>
      <material name="hole_mat"><color rgba="0.3 0.3 0.8 1"/></material>
    </visual>
    <collision>
      <geometry><box size="0.06 0.06 0.04"/></geometry>
      <origin xyz="0 0 0.02"/>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="1e-4" iyy="1e-4" izz="1e-4" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
</robot>
"""


def _write_tmp_urdf(content: str, name: str) -> str:
    path = Path(f"/tmp/{name}.urdf")
    path.write_text(content)
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────
class PegInsertionEnv:
    """
    Minimal pick-and-insert environment.

    Parameters
    ----------
    render      : bool  – open GUI window (slow)
    domain      : str   – 'source' or 'target' (controls domain randomisation)
    max_steps   : int   – episode length in control steps
    """

    PHASE_APPROACH = 0
    PHASE_GRASP    = 1
    PHASE_INSERT   = 2

    def __init__(self, render: bool = False, domain: str = "source",
                 max_steps: int = 300):
        self.render    = render
        self.domain    = domain
        self.max_steps = max_steps
        self._dr       = DR_SOURCE if domain == "source" else DR_TARGET

        mode = p.GUI if render else p.DIRECT
        self.client = p.connect(mode)
        p.setAdditionalSearchPath(pd.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, GRAVITY, physicsClientId=self.client)
        p.setTimeStep(1.0 / 240, physicsClientId=self.client)   # 240 Hz physics

        self._step_count = 0
        self._phase      = self.PHASE_APPROACH

        self.robot_id  = None
        self.peg_id    = None
        self.hole_id   = None
        self.table_id  = None

    # ── public API ───────────────────────────────────────────────────────────

    def reset(self) -> dict:
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, GRAVITY, physicsClientId=self.client)
        p.setTimeStep(1.0 / 240, physicsClientId=self.client)

        self._step_count = 0
        self._phase      = self.PHASE_APPROACH
        self._grasped    = False

        self._load_scene()
        self._set_robot_home()

        # warm-up physics
        for _ in range(50):
            p.stepSimulation(physicsClientId=self.client)

        return self._get_obs()

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        """
        action : np.ndarray shape (8,)
            [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper_open(0/1), phase_id(0/1/2)]
            (for data collection the oracle fills this; for policy rollout same)

        Returns obs, reward, done, info
        """
        self._apply_action(action)

        # step physics 12 times per control step (240/20 = 12)
        for _ in range(12):
            p.stepSimulation(physicsClientId=self.client)
            if self.render:
                time.sleep(1.0 / 240)

        self._step_count += 1
        self._phase = int(np.clip(round(action[7]), 0, 2))

        obs    = self._get_obs()
        reward = self._compute_reward()
        done   = self._is_done() or self._step_count >= self.max_steps
        info   = {"phase": self._phase, "step": self._step_count,
                  "success": self._check_insertion_success()}

        return obs, reward, done, info

    def close(self):
        p.disconnect(self.client)

    # ── scene loading ────────────────────────────────────────────────────────

    def _load_scene(self):
        dr = self._dr

        # table
        tc = np.random.uniform(*dr["table_color_range"])
        self.table_id = p.loadURDF(
            "table/table.urdf",
            basePosition=[0.5, 0, 0],
            physicsClientId=self.client
        )
        p.changeVisualShape(self.table_id, -1,
                            rgbaColor=[tc, tc, tc, 1],
                            physicsClientId=self.client)

        # peg object
        pc  = np.random.uniform(*dr["peg_color_range"])
        mass = np.random.uniform(*dr["object_mass_range"])
        fric = np.random.uniform(*dr["friction_range"])
        peg_urdf = _write_tmp_urdf(
            PEG_URDF_TEMPLATE.format(r=pc, g=pc * 0.5, b=0.2, mass=mass),
            "peg"
        )
        peg_x = np.random.uniform(0.35, 0.45)
        peg_y = np.random.uniform(-0.1, 0.1)
        self.peg_id = p.loadURDF(
            peg_urdf,
            basePosition=[peg_x, peg_y, 0.63],
            physicsClientId=self.client
        )
        p.changeDynamics(self.peg_id, -1, lateralFriction=fric,
                         physicsClientId=self.client)

        # hole / receptacle (fixed)
        hole_urdf = _write_tmp_urdf(HOLE_URDF_TEMPLATE, "hole")
        self.hole_id = p.loadURDF(
            hole_urdf,
            basePosition=[0.55, 0.20, 0.625],
            useFixedBase=True,
            physicsClientId=self.client
        )

        # Franka Panda
        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0, 0, 0.625],
            useFixedBase=True,
            physicsClientId=self.client
        )

        # store peg initial position for reward computation
        self._peg_start = np.array([peg_x, peg_y, 0.63])
        self._hole_pos  = np.array([0.55, 0.20, 0.665])  # top of hole

    def _set_robot_home(self):
        """Reset joints to a neutral above-table pose."""
        home_q = [0, -0.4, 0, -2.0, 0, 1.6, 0.8]
        for i, q in zip(PANDA_ARM_JOINTS, home_q):
            p.resetJointState(self.robot_id, i, q,
                              physicsClientId=self.client)
        # open gripper
        for j in PANDA_FINGER_JOINTS:
            p.resetJointState(self.robot_id, j, 0.04,
                              physicsClientId=self.client)

    # ── action application ───────────────────────────────────────────────────

    def _apply_action(self, action: np.ndarray):
        ee_pos, ee_orn = self._get_ee_pose()
        delta_pos = action[:3] * 0.02        # scale to max 2 cm per step
        target_pos = ee_pos + delta_pos

        target_pos = np.clip(target_pos, [0.2, -0.4, 0.63], [0.8, 0.4, 1.0])

        gripper_open = action[6] > 0.5
        finger_q     = 0.04 if gripper_open else 0.0

        # IK
        joint_poses = p.calculateInverseKinematics(
            self.robot_id, EE_LINK_INDEX,
            target_pos, ee_orn,
            physicsClientId=self.client
        )

        for i, jp in zip(PANDA_ARM_JOINTS, joint_poses[:7]):
            p.setJointMotorControl2(
                self.robot_id, i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=jp,
                force=100,
                physicsClientId=self.client
            )

        for j in PANDA_FINGER_JOINTS:
            p.setJointMotorControl2(
                self.robot_id, j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=finger_q,
                force=20,
                physicsClientId=self.client
            )

    # ── observations ─────────────────────────────────────────────────────────

    def _get_obs(self) -> dict:
        rgb, depth = self._render_camera()
        proprio     = self._get_proprio()
        ft          = self._get_force_torque()
        peg_pos     = np.array(
            p.getBasePositionAndOrientation(self.peg_id,
                                            physicsClientId=self.client)[0]
        )

        return {
            "rgb":        rgb,          # (H, W, 3) uint8
            "depth":      depth,        # (H, W, 1) float32
            "proprio":    proprio,      # (21,) float32
            "force":      ft,           # (6,) float32
            "peg_pos":    peg_pos,      # for oracle / reward only
            "phase":      self._phase,
        }

    def _render_camera(self):
        view_mat = p.computeViewMatrix(
            cameraEyePosition   = [0.5, -0.5, 1.2],
            cameraTargetPosition= [0.5,  0.1, 0.65],
            cameraUpVector      = [0, 0, 1],
            physicsClientId     = self.client
        )
        proj_mat = p.computeProjectionMatrixFOV(
            fov=60, aspect=1.0, nearVal=0.1, farVal=2.0,
            physicsClientId=self.client
        )
        _, _, rgb, depth_buf, _ = p.getCameraImage(
            IMG_W, IMG_H,
            viewMatrix      = view_mat,
            projectionMatrix= proj_mat,
            renderer        = p.ER_TINY_RENDERER,
            physicsClientId = self.client
        )
        rgb_arr   = np.array(rgb,  dtype=np.uint8)[:, :, :3]
        depth_arr = np.array(depth_buf, dtype=np.float32)[..., None]
        return rgb_arr, depth_arr

    def _get_proprio(self) -> np.ndarray:
        joint_states = p.getJointStates(
            self.robot_id, PANDA_ARM_JOINTS,
            physicsClientId=self.client
        )
        q   = np.array([s[0] for s in joint_states], dtype=np.float32)
        dq  = np.array([s[1] for s in joint_states], dtype=np.float32)
        ee_pos, ee_orn = self._get_ee_pose()
        return np.concatenate([q, dq, ee_pos, ee_orn]).astype(np.float32)

    def _get_force_torque(self) -> np.ndarray:
        """Simulated force/torque at the wrist (joint 6)."""
        state = p.getJointState(
            self.robot_id, 6,
            physicsClientId=self.client
        )
        # state[2] = reaction forces [Fx,Fy,Fz,Mx,My,Mz]
        ft = np.array(state[2], dtype=np.float32)
        # add small noise to simulate real sensor
        ft += np.random.normal(0, 0.02, size=6).astype(np.float32)
        return ft

    def _get_ee_pose(self):
        state = p.getLinkState(
            self.robot_id, EE_LINK_INDEX,
            physicsClientId=self.client
        )
        return np.array(state[0]), state[1]   # pos, orn (quaternion)

    # ── reward & termination ─────────────────────────────────────────────────

    def _compute_reward(self) -> float:
        peg_pos = np.array(
            p.getBasePositionAndOrientation(self.peg_id,
                                            physicsClientId=self.client)[0]
        )
        dist_to_hole = np.linalg.norm(peg_pos[:2] - self._hole_pos[:2])
        height_diff  = peg_pos[2] - self._hole_pos[2]

        r = -dist_to_hole * 2.0
        if height_diff < 0.01:  # peg below hole rim
            r += 1.0
        return float(r)

    def _is_done(self) -> bool:
        return self._check_insertion_success()

    def _check_insertion_success(self) -> bool:
        peg_pos = np.array(
            p.getBasePositionAndOrientation(self.peg_id,
                                            physicsClientId=self.client)[0]
        )
        xy_dist = np.linalg.norm(peg_pos[:2] - self._hole_pos[:2])
        inserted = peg_pos[2] < self._hole_pos[2] - 0.02
        return bool(xy_dist < 0.02 and inserted)
