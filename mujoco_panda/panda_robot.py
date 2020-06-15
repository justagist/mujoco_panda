import os
from .mujoco_robot import MujocoRobot

MODEL_PATH = os.environ['MJ_PANDA_PATH']+'/mujoco_panda/models/franka_panda.xml'

class PandaArm(MujocoRobot):

    def __init__(self, model_path=MODEL_PATH, render=True):

        super().__init__(model_path, render=render)
        
