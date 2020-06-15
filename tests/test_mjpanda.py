
from mujoco_panda import PandaArm

if __name__ == "__main__":
    p = PandaArm()

    while True:
        p.step()
