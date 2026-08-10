# Hierarchical Imitation Learning for Peg Insertion
## Step-by-step setup and execution guide

---

## Project structure

```
project_root/
├── research_env.yml          ← conda environment
├── sim/
│   ├── pybullet_env.py       ← PyBullet simulation
│   └── oracle.py             ← scripted expert + data collection
├── policy/
│   ├── models.py             ← all neural network components
│   └── train.py              ← training loop
├── eval/
│   └── evaluate.py           ← evaluation + results table
└── data/                     ← created automatically
    ├── demos_source.hdf5
    └── demos_target.hdf5
```

---

## Step 0 — Install the environment

```bash
# Install Miniconda if you don't have it:
# https://docs.conda.io/en/latest/miniconda.html

conda env create -f research_env.yml
conda activate embodied_il

# Verify PyBullet works (no GPU needed):
python -c "import pybullet; print('PyBullet OK')"

# Verify PyTorch:
python -c "import torch; print(torch.__version__)"
```

If you have a CUDA GPU, remove the `cpuonly` line from the yml
and uncomment the cuda build before creating the environment.

---

## Step 1 — Collect demonstrations (source domain)

The oracle is a scripted controller with privileged access to
ground-truth object positions. It collects demonstrations and
saves them as compressed HDF5 files.

```bash
# Headless (recommended — much faster):
python sim/oracle.py --n_demos 150 --domain source \
                     --out data/demos_source.hdf5

# With GUI (useful to verify the task looks correct):
python sim/oracle.py --n_demos 5 --domain source \
                     --out data/demos_source_debug.hdf5 --render
```

Each episode takes ~300 steps at 20 Hz = ~15 seconds real time
in headless mode. 150 demos ≈ 10–20 minutes on a laptop.

---

## Step 2 — Collect demonstrations (target domain)

Target domain uses broader domain randomisation (colours,
mass, friction, lighting). The oracle still works because it
uses privileged position info, not vision.

```bash
python sim/oracle.py --n_demos 150 --domain target \
                     --out data/demos_target.hdf5
```

Note: target-domain demos are only used to evaluate transfer,
NOT for training. Training uses source demos only.

---

## Step 3 — Train the hierarchical policy

```bash
python policy/train.py \
    --model hierarchical \
    --demos data/demos_source.hdf5 \
    --epochs 50 \
    --batch_size 128 \
    --lr 1e-4 \
    --ckpt_dir checkpoints/
```

Expected output (per epoch):
```
Epoch 001  train=0.4821  val=0.4710
  → saved checkpoint (checkpoints/hierarchical_best.pt)
Epoch 002  train=0.3912  val=0.3850
...
```

On CPU: ~3–5 min/epoch for 150 demos (≈45k timesteps).
On GPU: ~30–60 s/epoch.

---

## Step 4 — Train the monolithic baseline

```bash
python policy/train.py \
    --model monolithic \
    --demos data/demos_source.hdf5 \
    --epochs 50 \
    --batch_size 128 \
    --lr 1e-4 \
    --ckpt_dir checkpoints/
```

---

## Step 5 — Evaluate both models

```bash
python eval/evaluate.py \
    --hierarchical_ckpt checkpoints/hierarchical_best.pt \
    --monolithic_ckpt   checkpoints/monolithic_best.pt \
    --n_episodes 50
```

This runs evaluation in both source and target domains plus
the perturbation experiment, then prints the results table:

```
=================================================================
Condition                      Hierarch.  Monolith.
=================================================================
Source success rate                0.820      0.760
Target success rate                0.680      0.520
Perturbed success rate             0.640      0.460
Phase classifier acc.              0.891      —
Inference time (ms)               12.340      9.210
=================================================================
```

(Numbers above are illustrative; your results will vary.)

---

## Troubleshooting

**PyBullet crashes on import:**
```bash
pip install --upgrade pybullet
```

**pybullet_data assets not found:**
```bash
python -c "import pybullet_data; print(pybullet_data.getDataPath())"
# should print a valid directory containing franka_panda/
```

**URDF not found (franka_panda/panda.urdf):**
The Franka URDF ships with pybullet_data. If missing:
```bash
pip install pybullet-data
```

**Out of memory on CPU:**
Reduce batch_size to 32 or 64.

**Training loss doesn't decrease:**
- Check that HDF5 files have > 0 demos:
  `python -c "import h5py; f=h5py.File('data/demos_source.hdf5'); print(list(f.keys()))"`
- Reduce learning rate to 5e-5.

---

## Experiment variants for the paper

### Ablation 1: Modality ablation
Train SubPolicy phase-0 with only RGB (drop depth) to measure
the contribution of depth during approach.

### Ablation 2: Phase label quality
Replace oracle phase labels with noisy labels
(randomly flip 20% of phase assignments) to test classifier robustness.

### Ablation 3: Number of demos
Train on 50 / 100 / 150 demos to plot a learning curve,
showing whether the hierarchical policy is more data-efficient.

---

## Citing the key papers in your paper

```bibtex
@article{liu2024embodied,
  title   = {Aligning Cyber Space with Physical World: A Comprehensive Survey on Embodied AI},
  author  = {Liu, Yang and others},
  journal = {IEEE/ASME Transactions on Mechatronics},
  year    = {2024}
}

@article{billard2025roadmap,
  title   = {A roadmap for AI in robotics},
  author  = {Billard, Aude and others},
  journal = {Nature Machine Intelligence},
  volume  = {7},
  pages   = {818--824},
  year    = {2025}
}

@article{roy2021embodied,
  title   = {From Machine Learning to Robotics: Challenges and Opportunities for Embodied Intelligence},
  author  = {Roy, Nicholas and others},
  journal = {arXiv:2110.15245},
  year    = {2021}
}
```
