# from mujoco_py import load_model_from_path, MjSim, MjViewer
import mujoco_py as mjp
import threading

class MujocoRobot(object):

    def __init__(self, model_path, render = True):
        """
        Constructor

        :param model_path: path to model xml file for robot
        :type model_path: str
        :param render: create a visualiser instance in mujoco; defaults to True
        :type render: bool, optional
        """
        self._model = mjp.load_model_from_path(model_path)
        self._sim = mjp.MjSim(self._model)
        self._viewer = mjp.MjViewer(self._sim) if render else None

        self._asynch_thread_active = False

    def start_asynchronous_run(self):
        """
        Start a separate thread running the step simulation 
        for the robot. Rendering still has to be called in the 
        main thread.
        """
        def continuous_run():
            self._asynch_thread_active = True
            while self._asynch_thread_active:
                self.step(render=False)
        
        self._asynch_sim_thread = threading.Thread(target = continuous_run)
        self._asynch_sim_thread.start()

    def stop_asynchronous_run(self):
        self._asynch_thread_active = False
        self._asynch_sim_thread.join()

    def __del__(self):
        if hasattr(self,'_asynch_sim_thread') and self._asynch_sim_thread.isAlive():
            self._asynch_thread_active = False
            self._asynch_sim_thread.join()

    def step(self, render=True):
        self._sim.step()
        if render:
            self.render()

    def render(self):
        if self._viewer is not None:
            self._viewer.render()
