# Mujoco Panda

> [!WARNING]
> **Archived -- superseded by [`mujoco_robot`](https://github.com/justagist/mujoco_robot).**
>
> This repo targets the deprecated [`mujoco_py`](https://github.com/openai/mujoco-py) bindings and is Panda-specific. It is no longer maintained.
>
> The successor, **[`mujoco_robot`](https://github.com/justagist/mujoco_robot)**, is a ground-up rewrite on the official [`mujoco`](https://github.com/google-deepmind/mujoco) bindings: generic (any robot via `robot_descriptions`), with a cleaner state/control API, Jacobian/`mink`-based IK, native FT sensors, rich contact info, and step hooks. It's a MuJoCo counterpart to [`pybullet_robot`](https://github.com/justagist/pybullet_robot).
>
> Use `mujoco_robot` for new work. This repo stays up for reference only.

- Franka Emika Panda Robot model definitions for Mujoco.
- MuJoCo-based robot simulator.
- Python 3 API for controlling and monitoring the simulated Franka Emika Panda Robot.
- Low-level controllers: direct position and torque control.
- Higher-level controller: task-space hybrid force motion controller (torque-based).

Robot models are in [mujoco_panda/models](mujoco_panda/models).

## Requirements

To use all functionalities of the provided library, the following dependencies have to be met.

- [mujoco_py](https://github.com/openai/mujoco-py)
- numpy (`pip install numpy`)
- scipy (`pip install scipy`)
- quaternion (`pip install numpy-quaternion`)
- tkinter (`apt-get install python3-tk) (only for visualised debugging`)

## Setup Instructions

Once mujoco_py is correctly installed, this library can be installed by running:

`python setup.py install`



## Usage

See `examples` for controller and API usage. See [MujocoRobot](mujoco_panda/mujoco_robot.py) and [PandaArm](mujoco_panda/panda_robot.py) for full documentation.
