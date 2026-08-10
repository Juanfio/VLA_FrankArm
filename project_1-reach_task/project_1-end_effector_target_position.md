# Hands-On Curriculum: Sim-to-Real Robot Arm Control

Goal: rebuild, piece by piece and in your own words/code, a system that trains
a policy (from simulated demonstrations) to control a robotic arm — the same
kind of pipeline we've been dissecting (env → oracle → dataset → model →
training → evaluation), but broken into bite-sized tasks so you build real
intuition instead of copy-pasting.

How to use this: do the tracks roughly in order. Each task has a **goal**,
**concepts**, a **checklist**, and a **"you know it works when"** signal.
Don't look at the original files while doing a task — write it from scratch,
then compare afterward. Ask me for a code review after each task; I'll point
out bugs and design issues rather than rewrite it for you.

---

## TRACK A — PyBullet Fundamentals

### A1. Hello, physics
**Goal:** connect to PyBullet, drop a cube on a plane, watch it fall and settle.
**Concepts:** `p.connect`, `p.GUI` vs `p.DIRECT`, `p.setGravity`, `p.loadURDF`,
`p.stepSimulation`, the physics-timestep vs wall-clock distinction.
**Checklist:**
- Connect with `p.GUI` so you can *see* it.
- Load `plane.urdf` from `pybullet_data`.
- Load `r2d2.urdf` (comes bundled) or a simple box a meter above the ground.
- Step simulation ~500 times with a small `time.sleep` so you can watch it fall.
- Print the object's position every 100 steps using `p.getBasePositionAndOrientation`.
**You know it works when:** you can watch the object fall, bounce a little, and
settle, and your printed z-position converges to a resting value.

### A2. Load and inspect a robot arm
**Goal:** load the Franka Panda (or KUKA `kuka_iiwa/model.urdf`, also bundled)
and understand its joint structure before trying to control it.
**Concepts:** links vs joints, `p.getNumJoints`, `p.getJointInfo`, joint types
(revolute, prismatic, fixed), why finger joints are separate from arm joints.
**Checklist:**
- Load the arm with `useFixedBase=True` (so it doesn't fall over).
- Loop over `range(p.getNumJoints(robot_id))`, print each joint's index, name,
  and type (`p.getJointInfo` returns a tuple — look up what each field means).
- Identify which joint indices are the actuated arm joints vs. gripper/fingers
  vs. fixed/decorative joints.
**You know it works when:** you can explain, in your own words, why the code
we've been reading hardcodes `PANDA_ARM_JOINTS = [0,1,2,3,4,5,6]` and
`PANDA_FINGER_JOINTS = [9, 10]` — i.e. you found those numbers yourself by
inspecting the URDF, not by trusting the comment.

### A3. Move the arm — joint control
**Goal:** command individual joints to specific angles and watch the arm move.
**Concepts:** `p.setJointMotorControl2`, `POSITION_CONTROL`, the `force`
parameter (max torque — what happens if it's too low?).
**Checklist:**
- Pick one arm joint, command it to a target angle with `POSITION_CONTROL`.
- Step the simulation and watch it move there.
- Try setting `force` very low (e.g. `1`) vs. high (`200`) — observe the
  difference when gravity/other joints resist the motion.
- Now command **all 7 arm joints** at once to some home pose.
**You know it works when:** you can pose the arm into a specific
configuration on command, and you can explain what `force` controls (it's not
speed — what is it?).

### A4. Move the end-effector — inverse kinematics
**Goal:** instead of specifying joint angles, specify a target XYZ for the
end-effector and let IK figure out the joints.
**Concepts:** `p.calculateInverseKinematics`, why IK can have multiple valid
solutions or fail to converge, `p.getLinkState` for reading EE pose.
**Checklist:**
- Find the end-effector link index (see A2).
- Read the current EE position with `getLinkState`.
- Pick a target position 10cm away, call `calculateInverseKinematics`, apply
  the returned joint angles via `POSITION_CONTROL`.
- Step until it (approximately) arrives; print the resulting EE position and
  compare to your target.
**You know it works when:** you can move the EE to an arbitrary reachable
point just by specifying XYZ, without thinking about individual joints.
**Stretch:** try a target *outside* the arm's reachable workspace and observe
what IK gives you instead of an error.

### A5. Build a minimal `reset()` / `step()` environment
**Goal:** wrap everything above into a tiny Gym-style class.
**Concepts:** environment API design, what state needs to be reset vs.
reloaded each episode, the difference between control frequency and physics
frequency (why do we substep?).
**Checklist:**
- Write a class `MiniArmEnv` with `reset()` (rebuild scene, robot home pose)
  and `step(action)` where `action` is an XYZ delta.
- Inside `step`, scale the delta, call IK, apply joint targets, then call
  `p.stepSimulation()` **multiple times** per `step()` call (pick a control
  Hz vs physics Hz ratio and justify your choice).
- Return `(obs, reward, done, info)` even if `obs` is just EE position for now.
**You know it works when:** you can write a tiny scripted loop (e.g. "move
+x for 20 steps") purely by calling `env.step(action)` repeatedly, with no
PyBullet calls outside the class.

### A6. Add a camera and read pixels
**Goal:** render an RGB/depth image from a virtual camera in the scene.
**Concepts:** view matrix vs. projection matrix, `computeViewMatrix`,
`computeProjectionMatrixFOV`, `getCameraImage`, image array shapes.
**Checklist:**
- Place a camera looking down at your scene (pick eye position, target,
  up-vector yourself — don't copy the numbers from `pybullet_env.py`).
- Call `getCameraImage` and inspect the returned RGB array's shape/dtype.
- Save one frame as a PNG (`PIL.Image.fromarray`) so you can actually look
  at it.
**You know it works when:** you have a saved image file that looks like your
scene from the angle you chose.

### A7. Force/torque sensing
**Goal:** read joint reaction forces and understand what they represent.
**Concepts:** `p.getJointState`, the force/torque tuple it returns, why
raw physics forces need noise added to resemble a real sensor.
**Checklist:**
- Read force/torque at a joint while the arm is at rest vs. while it's
  pushing against something (e.g. lower it into the table/plane collision).
- Compare the readings.
**You know it works when:** you can describe, from your own observation, why
force spikes when the arm makes contact — this is the signal your phase
classifier will later rely on.

---

## TRACK B — PyTorch Fundamentals

Do these somewhat in parallel with Track A — they don't depend on PyBullet.

### B1. Tensors and a manual training loop
**Goal:** fit `y = 2x + 1` with pure tensors, no `nn.Module`, no optimizer.
**Concepts:** `requires_grad`, `.backward()`, manual gradient descent,
why we zero gradients.
**Checklist:**
- Generate noisy synthetic `(x, y)` data.
- Initialize `w`, `b` as tensors with `requires_grad=True`.
- Write the loop: forward pass → MSE loss → `loss.backward()` → manually
  update `w -= lr * w.grad` → zero grads.
**You know it works when:** your learned `w`, `b` converge close to `2`, `1`.

### B2. `nn.Module`, `nn.Sequential`, and a real optimizer
**Goal:** redo B1 (or a slightly harder nonlinear function) using proper
PyTorch idioms.
**Concepts:** subclassing `nn.Module`, `forward()`, `nn.Linear`,
`torch.optim`, why `optimizer.zero_grad()` still exists.
**Checklist:**
- Build a tiny 2-layer MLP as an `nn.Module`.
- Use `torch.optim.Adam`.
- Fit a nonlinear function, e.g. `y = sin(x) + noise`.
**You know it works when:** your loss curve goes down and predictions
visually match the sine curve when plotted.

### B3. Encoding different modalities
**Goal:** build small encoders for different input shapes, mirroring
`ImageEncoder` / `ProprioEncoder`, but on toy data.
**Concepts:** `nn.Conv2d` and how spatial dims shrink with stride, flattening,
concatenating heterogeneous feature vectors.
**Checklist:**
- Make a fake "image" encoder: random `(B, 3, 32, 32)` tensor → small CNN →
  `(B, 64)` vector. Manually compute (or just print) the shape after each
  conv layer so you understand why the flatten size is what it is.
- Make a fake "state vector" encoder: random `(B, 10)` tensor → MLP →
  `(B, 32)` vector.
- Concatenate the two encoder outputs into one `(B, 96)` conditioning vector.
**You know it works when:** you can explain, without checking, why an input
image of `64x64` becomes `4x4` after 3 stride-2 convs and why the `Linear`
layer after `Flatten` needs `64 * 4 * 4` as its input size (this trips
everyone up at least once — good to get the arithmetic wrong and debug it).

### B4. A classifier with temporal input
**Goal:** rebuild something like `PhaseClassifier` on toy sequential data.
**Concepts:** `nn.Conv1d` over a time axis, `rearrange`/`permute` for axis
ordering, `AdaptiveAvgPool1d`, cross-entropy loss.
**Checklist:**
- Generate toy sequences: 3 classes of synthetic "signals" over 20 timesteps
  x 6 channels (e.g. class 0 = flat near-zero, class 1 = a spike partway
  through, class 2 = high sustained value) with noise.
- Build a small Conv1d classifier, train it to distinguish the 3 classes.
- Check accuracy on held-out synthetic sequences.
**You know it works when:** your classifier gets high accuracy on clearly
separable synthetic classes, and you can explain what would happen to
accuracy if you shrank the window from 20 steps to 2.

### B5. Regression baseline for actions
**Goal:** before touching diffusion, build the "naive" way to predict actions
— direct regression — and understand its failure mode.
**Concepts:** MSE regression head, mode-averaging with multimodal targets.
**Checklist:**
- Create a synthetic dataset where, for a given input `x`, the "expert
  action" is **bimodal** — e.g. with 50% probability `y = x + 5`, else
  `y = x - 5`, plus noise.
- Train a plain MLP regressor with MSE loss to predict `y` from `x`.
- Plot: does the model predict something near `x+5`, near `x-5`, or right in
  the useless middle (~`x+0`)?
**You know it works when:** you can *see* the mode-averaging failure with
your own eyes — this is the exact problem diffusion policies exist to solve,
and this exercise is what makes that motivation click instead of being an
abstract claim.

### B6. Minimal diffusion model (1D toy version)
**Goal:** implement DDPM training + sampling on a simple 1D or 2D dataset
(*not* robot actions yet) — e.g. learn to generate points sampled from two
separate Gaussian blobs.
**Concepts:** noise schedule (`betas`, `alphas`, `alpha_bar`), the forward
noising formula, noise-prediction loss, reverse sampling loop.
**Checklist:**
- Define a linear beta schedule and precompute `alphas`/`alpha_bar` (same
  formulas used in `models.py`).
- Build a tiny MLP `noise_net(x_t, t, cond=None)` (condition can be omitted
  for this toy version, or condition on which blob to generate).
- Training loop: sample real point → sample random `t` → noise it → predict
  noise → MSE loss.
- Sampling loop: start from pure noise, iterate backward through timesteps,
  denoise.
- Plot generated samples vs. the real two-blob dataset.
**You know it works when:** your generated samples visually cluster into the
two blobs (not one blended blob in the middle) — i.e. you've reproduced,
on a toy problem, exactly the thing that fixes the failure from B5.
**This is the single most important task in the whole curriculum** — once you
can build this from scratch and explain every line, `DiffusionNoiseNet` /
`SubPolicy.loss` / `SubPolicy.sample` in the original code will be almost
line-for-line familiar.

### B7. Conditioning the diffusion model
**Goal:** extend B6 so generation depends on an input condition, not just
"generate from the dataset in general."
**Concepts:** conditional generation, concatenating condition + time
embedding, why conditioning changes *which* mode gets generated.
**Checklist:**
- Modify the toy dataset from B5 so the "which mode" choice depends
  deterministically (or probabilistically) on some input `c` (e.g. `c=0` →
  always the "+5" mode, `c=1` → always the "-5" mode, `c=2` → 50/50).
- Add a `cond` input to your noise-prediction network, feed `c` in (embed it
  if it's categorical, or just concatenate if continuous).
- Verify: sampling with `c=0` reliably gives you the "+5" mode, `c=1` gives
  "-5", `c=2` gives a genuine mix across many samples.
**You know it works when:** conditioning actually steers generation —
this *is* what "observation-conditioned action generation" means in the
robot policy, just with `cond` = encoded RGB/proprio/force instead of a
toy scalar.

---

## TRACK C — Integration: Build Your Own Mini Pipeline

Now combine both tracks into your own (deliberately simplified) version of
the system we've been reading. Build a **much smaller task** than full peg
insertion first — this keeps iteration fast and debugging tractable.

### C1. Simplify the task drastically
**Goal:** define a toy task you can solve end-to-end in a day, not a week.
Suggestion: **"reach" task** — no grasping, no insertion, just move the EE
to a randomly-placed visible target sphere and stop when close.
**Checklist:**
- Extend your `MiniArmEnv` from A5: spawn a colored sphere at a random
  position each `reset()`, add a camera (A6), add a success check
  (distance < threshold).
- No gripper, no phases yet — just position control.
**You know it works when:** `reset()`/`step()` work and you can manually
drive the EE to the sphere with hardcoded actions.

### C2. Write your own scripted oracle
**Goal:** a simple proportional controller that always reaches the target.
**Concepts:** privileged information, proportional control, threshold-based
"done" detection (recall we debugged exactly this kind of logic in
`OraclePolicy`).
**Checklist:**
- Oracle reads the *true* target position (privileged) and outputs
  `action = normalize(target - ee_pos)`.
- Run it in a loop, confirm it reliably reaches the target within your
  `max_steps`.
- **Deliberately introduce and then find a bug** — e.g. advance to "done"
  after only 1 step near the target instead of requiring it to stay close —
  and see it fail intermittently, the same class of bug we found in the
  gripper-dwell issue. This is good practice for the debugging skill, not
  just the coding skill.
**You know it works when:** you can report a success rate over N attempts
(should be near 100% for a well-tuned reach task).

### C3. Collect a dataset
**Goal:** save oracle rollouts to HDF5.
**Concepts:** episode-based storage, why HDF5 groups per-episode, saving
multiple modalities together.
**Checklist:**
- Loop: reset, roll out oracle, save `(rgb, proprio, action)` per timestep.
- Save into `demo_0000/`, `demo_0001/`, ... groups, one HDF5 file.
- Print a running success rate as you collect (don't wait 90 minutes to find
  out something's wrong — instrument this from the start, unlike the
  original `collect_demos`).
**You know it works when:** you have an `.hdf5` file with N successful demo
groups, and you can re-open it and print shapes to verify.

### C4. Build your dataset class
**Goal:** load the HDF5 file into per-timestep training samples.
**Concepts:** flattening episodes into samples (and now that you understand
the train/val leakage issue we discussed — **split by episode, not by
timestep**, as a deliberate improvement over the original).
**Checklist:**
- Write `ReachDataset(Dataset)` loading from your HDF5 file.
- Split into train/val **by demo index**, not by flattened sample index.
- Write a `collate_fn`.
**You know it works when:** `DataLoader` gives you correctly-shaped batches,
and you can print which demo indices ended up in val vs train to confirm no
leakage.

### C5. Baseline: direct regression policy
**Goal:** train the "naive" policy first, get a working end-to-end loop
before adding diffusion complexity.
**Checklist:**
- Encoders (from B3) → concatenate → small MLP head → predict action
  directly (MSE loss against oracle's action).
- Train it, checkpoint on val loss.
- Write a **rollout evaluation script** (the piece missing from the original
  code!) that loads the checkpoint, runs it in closed-loop in your env, and
  reports success rate — not just val loss.
**You know it works when:** you have an actual success-rate number for your
trained policy, obtained by *running* it, not just its loss value.

### C6. Upgrade to a diffusion policy
**Goal:** swap the MLP regression head for the diffusion approach from B6/B7.
**Checklist:**
- Reuse your B6/B7 diffusion machinery, conditioned on the encoder output
  from C5.
- Retrain, re-run your rollout evaluation script from C5.
- Compare success rate and qualitative behavior against the regression
  baseline — for a simple unimodal reach task, do you expect diffusion to
  help much? (Good question to think about *before* running it — then check
  if your prediction was right.)
**You know it works when:** you can state, with your own evidence, whether
diffusion helped on this particular toy task, and why that result does or
doesn't match your B5 intuition about multimodality.

### C7. Add a second phase + gating
**Goal:** extend the task so there's a real reason for phase structure —
e.g. "reach toward target, then reach toward a second target" — and add a
simple classifier that gates between two sub-policies.
**Checklist:**
- Two-phase oracle, phase labels saved to HDF5.
- A small classifier (from B4) trained to predict phase from observations.
- At rollout time, use the classifier's prediction (not ground truth) to
  route to the correct sub-policy — and specifically check what happens when
  the classifier is *wrong* (does the wrong sub-policy behave reasonably or
  catastrophically?).
**You know it works when:** you can describe a concrete failure case you
observed where classifier error propagated into task failure — this is
the real, hands-on version of the "train/inference mismatch" issue we
discussed on the original `HierarchicalPolicy`.

### C8. Domain randomization + a "domain gap" experiment
**Goal:** reproduce the source/target idea on your toy task.
**Checklist:**
- Add two randomization regimes (narrow vs. wide ranges) for color/lighting
  of your scene, like `DR_SOURCE`/`DR_TARGET`.
- Train only on source-domain demos.
- Evaluate (rollout success rate) on source vs. target domain.
**You know it works when:** you can report a real number for the
sim-to-(harder-)sim performance drop — the actual research question this
whole codebase was built to study, now something you've measured yourself
on a system you built from scratch.

---

## How to work with me on this

For each task: write your attempt first, then share it with me and ask for
a **review**, not a rewrite. I'll point out bugs, bad defaults, and design
smells (the way we did with the gripper-dwell bug and the solid-hole URDF) —
that debugging back-and-forth is where most of the real learning happens,
more than the initial writing.

Suggested pace: Track A and B in parallel, roughly one task every 1–2
sessions. Track C only after finishing both — it's where everything
compounds, and it's the part that will feel like "I actually understand
this system" rather than "I've read about this system."
