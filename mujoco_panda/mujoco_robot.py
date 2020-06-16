# from mujoco_py import load_model_from_path, MjSim, MjViewer
import mujoco_py as mjp
import numpy as np
import threading
import logging

LOG_LEVEL = "DEBUG"
logging.basicConfig(format='\n%(levelname)s: %(message)s\n', level=LOG_LEVEL)

class MujocoRobot(object):

    def __init__(self, model_path, render = True, config = None):
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

        self._has_gripper = False # by default, assume no gripper is attached

        self._movable_joints = self.get_movable_joints()

        self._all_joints = range(self._model.nv)

        self._all_joint_names = [self.model.joint_id2name(j) for j in self._all_joints]

        self._all_joint_dict = dict(zip(self._all_joint_names, self._all_joints))

        if config is not None and config['ee_link_name'] is not None:
            self._ee_link_name = config['ee_link_name']
            self._ee_link_idx = self._model.body_id2name(self._ee_link_name)
        else:
            self._ee_link_idx, self._ee_link_name = self._use_last_defined_link()

        self._asynch_thread_active = False

        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(LOG_LEVEL)

    def _use_last_defined_link(self):
        return self._model.nbody-1, self._model.body_id2name(self._model.nbody-1)

    @property
    def sim(self):
        return self._sim

    @property
    def viewer(self):
        return self._viewer
    
    @property
    def model(self):
        return self._model

    def has_body(self, bodies):
        if isinstance(bodies, str):
            bodies = [bodies]
        for body in bodies:
            if not body in self._model.body_names:
                return False
        return True

    def get_movable_joints(self):
        # self._model.joint_names
        trntype = self._model.actuator_trntype # transmission type (0 == joint)
        trnid = self._model.actuator_trnid # transmission id (get joint actuated)

        mvbl_jnts = []
        for i in range(trnid.shape[0]):
            if trntype[i] == 0 and trnid[i,0] not in mvbl_jnts:
                mvbl_jnts.append(trnid[i,0])
        
        return sorted(mvbl_jnts)

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

    def set_joint_positions(self, cmd, joints=None):
        """
        Set target joint positions. Use for position controlling.

        @param cmd  : joint position values
        @type cmd   : [float] * self._nu

        """
        cmd = np.asarray(cmd).flatten()
        joints = np.r_[:cmd.shape[0]] if joints is None else np.r_[joints]

        assert cmd.shape[0] == joints.shape[0]
        
        self._sim.data.ctrl[joints] = cmd

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
