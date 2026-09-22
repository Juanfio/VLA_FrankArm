import pybullet as p
import pybullet_data
import numpy as np


PANDA_ARM_JOINTS = [0, 1, 2, 3, 4, 5, 6]
PANDA_FINGER_JOINTS = [9, 10]


class MiniArmEnv():
    '''
    TODO: A function that adjusts the size of delta_step depending on the distance between end-effector and sphere.
            -- Large steps when it's far. Small steps when it's close.

    
    -------------------
    arm_env = MiniArmEnv()
    arm_env.reset()
    dt_pos = np.array([0.0,0.0,0.01]) # delta perturbation in the position.
    obs, reward, done, info = arm_env.step(action=dt_pos)
    '''
    def __init__(self, max_steps: int = 300, image_size: int = 224, render: bool = True):
        self.max_steps = max_steps
        self.physics_engine_update = 240 # 240 Hz.
        self.success_threshold = 0.25
        self._step_count = 0
        self.delta_step = 0.05 # update 5cm towards the direction of the sphere per step.
        self.image_size = image_size # both width and height.

        mode = p.GUI if render else p.DIRECT
        self.client = p.connect(mode)

        # load scene
        self._load_scene()

        # get end effector index.
        self.end_effector = self._get_end_effector(robotId=self.robotId)

    def reset(self):
        home_positions = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        finger_home = [0.04, 0.04]  # open gripper

        for j, angle in zip(PANDA_ARM_JOINTS, home_positions):
            p.resetJointState(self.robotId, j, angle, targetVelocity=0)

        for j, angle in zip(PANDA_FINGER_JOINTS, finger_home):
            p.resetJointState(self.robotId, j, angle, targetVelocity=0)

        # reset sphere position
        p.resetBasePositionAndOrientation(bodyUniqueId=self.sphereId,
                                          posObj=[np.random.uniform(-0.4, 0.4),
                                                  np.random.uniform(-0.4, 0.4),
                                                  np.random.uniform(0.4, 0.4)],
                                          ornObj=[0.0, 0.0, 0.0, 1.0])
        
        self.sphereId_position = np.array(p.getBasePositionAndOrientation(self.sphereId)[0])
        
        p.stepSimulation()
        return self._get_obs()

    def step(self, action):
        
        self._apply_action(action)

        p.setTimeStep(1 / self.physics_engine_update)

        for _ in range(18):
            # TODO: add a break. If the EE reached the target position it stops. Use self._check_ee_close_sphere() every N steps.
            p.stepSimulation()

        self._step_count += 1        
                  
        obs = self._get_obs()
        reward = None
        done = self._is_done() or self._step_count >= self.max_steps
        info = {"step": self._step_count, "success": self._check_position_success()}

        return obs, reward, done, info

    def close(self):
        p.disconnect(self.client)


    # ── scene loading ────────────────────────────────────────────────────────
    def _load_scene(self):
        "Load plane, frank arm and sphere"
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        p.loadURDF("plane.urdf")
        # upload the franka arm.
        self.robotId = p.loadURDF("franka_panda/panda.urdf",
                                basePosition=[0,0,0],
                                useFixedBase=True)

        # create a sphere
        radius = 0.1
        visual_shape = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                        radius=radius,
                                        rgbaColor=[0, 1, 0, 1])

        collision_shape = p.createCollisionShape(shapeType=p.GEOM_SPHERE,
                                                radius=radius)
                                                
        self.sphereId = p.createMultiBody(baseMass=0,
                                    baseCollisionShapeIndex=collision_shape,
                                    baseVisualShapeIndex=visual_shape,
                                    basePosition=[1,1,1])

    # ── action application ───────────────────────────────────────────────────

    def _apply_action(self, action):
        ee_pos, ee_orn = self._get_ee_pose() # reach task: the end effector must reach the position of the sphere.
        delta_pos = action * self.delta_step # delta (x,y,z) in meters (MKS system).
        target_pos = ee_pos + delta_pos
        
        # inverse kinematics
        joint_angles = p.calculateInverseKinematics(
            bodyUniqueId=self.robotId,
            endEffectorLinkIndex=self.end_effector,
            targetPosition=target_pos, # TODO: Add the ee_orn???
            maxNumIterations=400,
            residualThreshold=1e-7
        )

        # move the arm
        p.setJointMotorControlArray(
            bodyIndex=self.robotId,
            jointIndices=PANDA_ARM_JOINTS + PANDA_FINGER_JOINTS,
            controlMode=p.POSITION_CONTROL,
            targetPositions = joint_angles,
            forces = [500]*len(PANDA_ARM_JOINTS + PANDA_FINGER_JOINTS)
        )


    # ── observations ─────────────────────────────────────────────────────────

    def _get_obs(self):
        rgb_arr, depth_arr = self._render_camera()
        proprio = self._get_proprio()
        ee_pos, ee_orn = self._get_ee_pose() # position of robotic arm end effector in the URDF link frame (the one placed at a meaningful reference point by the URDF designer)
        sphere_pos = self.sphereId_position
        
        return {
            "rgb": rgb_arr,
            "depth": depth_arr,
            "proprio": proprio,
            "ee_pos": ee_pos,
            "ee_orn": ee_orn,
            "sphere_pos": sphere_pos
        }

    def _render_camera(self):
        view_mat = p.computeViewMatrix(
            cameraEyePosition    = [1,0,2], 
            cameraTargetPosition = [0,0,1],
            cameraUpVector       = [0,1,0.5],
            physicsClientId     = self.client
            )

        proj_mat = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=1.0,
            nearVal=0.1,
            farVal=10
            )
                                        
        _, _, rgb, depth, _ = p.getCameraImage(
            width=self.image_size,
            height=self.image_size,
            viewMatrix = view_mat,
            projectionMatrix=proj_mat,
            physicsClientId = self.client
        )
        rgb_arr   = np.array(rgb,  dtype=np.uint8)[:, :, :3]
        depth_arr = np.array(depth, dtype=np.float32)[..., None]

        return rgb_arr, depth_arr

    def _get_proprio(self) -> np.ndarray:
        '''
        * joint_states has length = PANDA_ARM_JOINTS
        * s[0] position value of the joint "s"
        * s[1] velocity value of the joint "s"
        '''
        joint_states = p.getJointStates(
            self.robotId, PANDA_ARM_JOINTS,
            physicsClientId=self.client
        )
        q   = np.array([s[0] for s in joint_states], dtype=np.float32) # joint angle (rad): how much a joint is rotated. Internal configuration.
        dq  = np.array([s[1] for s in joint_states], dtype=np.float32) # velocity
        ee_pos, ee_orn = self._get_ee_pose() # ee_pos: position (x,y,z) of the end effector.
        return np.concatenate([q, dq, ee_pos, ee_orn]).astype(np.float32)
    
    def _get_ee_pose(self):
        state = p.getLinkState(
            self.robotId, self.end_effector,
            physicsClientId=self.client
        )
        return np.array(state[4]), state[5]   # pos, orn (quaternion) in URDF frame of reference.

    # ── reward & termination ─────────────────────────────────────────────────
    def _check_position_success(self):
        ee_pos, _ = self._get_ee_pose()
        return np.sqrt(np.dot(ee_pos - self.sphereId_position, ee_pos - self.sphereId_position)) < self.success_threshold

    def _is_done(self):
        if self._check_position_success():
            print(f"End-effector is close (< {self.success_threshold} m) to sphere position")
        return self._check_position_success()

    # ── &&&&&& ─────────────────────────────────────────────────
    @staticmethod
    def _get_end_effector(robotId: int):
        end_effector_index = None

        for i in range(p.getNumJoints(robotId)):
            link_name = p.getJointInfo(robotId, i)[12].decode("utf-8")

            if link_name == "panda_hand":
                end_effector_index = i
                break
        return end_effector_index