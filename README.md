# Simulation projects

* Robotic's projects with simulated data (PyBullet) over multiple different tasks (e.g., project 1 is a reach task.)
* Projects aim to address the sim-to-real problem, i.e., policies learned in simulated environments often failed when deployed on real-world physical robots.

# Project structure

#### Project 1 - Reach Task

* /simulation:
    * miniarm_env.py: Environment with Franka Arm and sphere.
    * oracle.py: Simulate Frank Arm movement towards the sphere with privileged information; i.e., actions of the arm are taken in the direction of the sphere.
* /notebooks:
    * track_A_pybullet_fundamentals.ipynb.
        - examples showing how to build the simulated environment (with a Franka Arm).
    * track_B_pytorch_fundamentals.ipynb.
        - examples showing how to build the NN controller (with a Franka Arm).
    * track_C_pybullet_pytorch_integration.ipynb.
        - Simulate Franka Arm reaching a sphere.
        - Collect demonstrations.
        - Model predicting arm movement from visual (rgb) and proprioceptive inputs. Model description:
            * ImageEncoder(): Conv2d is used to encode the rgb data generated using p.getCameraImage() from PyBullet.
            * ProprioEncoder(): Linear layers are used to encode the proprioceptive data containing the angles and velocities of the Franka Arm joints (MiniArmEnv()_get_proprio())
            * ActionPrediction(): Encodes the rgb and proprioceptive inputs and outputs an action (delta from current end-effector position).
            * The model reaches the sphere position by generating small increments (deltas) from current end-effector position towards the sphere. It uses the image (rgb) and the proprio data to guide its actions.
* /data:
    - Simulated franka arm reaching a sphere in .hdf5 format.
* environment:
    env_robotics.yml




# References

1. Billard, A. et al. A roadmap for AI in robotics. Nature Machine Intelligence 7, 818–824 (2025).
2. Liu, Y. et al. Aligning Cyber Space with Physical World: A Comprehensive Survey on Embodied AI. arXiv:2407.06886 (2024).
3. Roy, N. et al. From Machine Learning to Robotics: Challenges and Opportunities for Embodied Intelligence. arXiv:2110.15245 (2021).