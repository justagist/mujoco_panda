# from mujoco_py import load_model_from_path, MjSim, MjViewer
import mujoco_py as mjp

class MujocoRobot(object):

    def __init__(self, model_path, render = True):

        self._model = mjp.load_model_from_path(model_path)
        self._sim = mjp.MjSim(self._model)
        self._viewer = mjp.MjViewer(self._sim) if render else None

    def step(self, render=True):
        self._sim.step()
        if render:
            self.render()

    def render(self):
        if self._viewer is not None:
            self._viewer.render()
