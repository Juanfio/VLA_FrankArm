# Simulation projects

* Robotic's projects with simulated data (PyBullet) over multiple different tasks (e.g., project 1 is a reach task.)
* Projects aim to address the sim-to-real problem, i.e., policies learned in simulated environments often failed when deployed on real-world physical robots.

# Project structure

#### Project 1 - Reach Task

* track_A_pybullet_fundamentals.ipynb.
    - examples showing how to build the simulated environment (with a Franka Arm).
* track_B_pytorch_fundamentals.ipynb.
    - examples showing how to build the NN controller (with a Franka Arm).
* track_B_pybullet_pytorch_integration.ipynb.
    - Simulate Franka Arm reaching a sphere.
    - Collect demonstrations.
    - Model predicting arm movement from visual and proprioceptive inputs.
* /data:
    - Simulated franka arm reaching a sphere in .hdf5 format.

# References

1. Billard, A. et al. A roadmap for AI in robotics. Nature Machine Intelligence 7, 818–824 (2025).
2. Liu, Y. et al. Aligning Cyber Space with Physical World: A Comprehensive Survey on Embodied AI. arXiv:2407.06886 (2024).
3. Roy, N. et al. From Machine Learning to Robotics: Challenges and Opportunities for Embodied Intelligence. arXiv:2110.15245 (2021).