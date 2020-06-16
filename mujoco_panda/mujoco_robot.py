import mujoco_py as mjp
import numpy as np
import threading
import logging
import time
import quaternion
from threading import Lock

LOG_LEVEL = "DEBUG"
logging.basicConfig(format='\n%(levelname)s: %(message)s\n', level=LOG_LEVEL)

class MujocoRobot(object):

    def __init__(self, model_path, render = True, config = None, gravity_compensation = True):
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

        self._nu = len(self._movable_joints)

        self._all_joints = range(self._model.nv)

        self._nq = len(self._all_joints)

        self._all_joint_names = [self.model.joint_id2name(j) for j in self._all_joints]

        self._all_joint_dict = dict(zip(self._all_joint_names, self._all_joints))

        self._enable_gravity_compensation = gravity_compensation
        self._additional_force = np.zeros(self._nq)

        if config is not None and config['ee_name'] is not None:
            self.set_as_ee(config['ee_name'])
        else:
            self._ee_idx, self._ee_name = self._use_last_defined_link()
            self._ee_is_a_site = False

        self._mutex = Lock()
        self._asynch_thread_active = False

        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(LOG_LEVEL)

    def set_as_ee(self, body_name):
        self._ee_name = body_name

        if body_name in self._model.site_names:
            self._ee_is_a_site = True
            self._ee_idx = self._model.site_name2id("ee_site")
            self._logger.debug("End-effector is a site in model: {}".format(body_name))
        else:
            self._ee_is_a_site = False
            self._ee_idx = self._model.body_name2id(self._ee_name)

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
        """
        Check if the provided bodies exist in model.

        :param bodies: list of body names
        :type bodies: [str]
        :return: True if all bodies present in model
        :rtype: bool
        """
        if isinstance(bodies, str):
            bodies = [bodies]
        for body in bodies:
            if not body in self._model.body_names:
                return False
        return True

    def body_jacobian(self, body_id = None, joint_indices = None):
        """
        return body jacobian at current step

        :param body_id: id of body whose jacobian is to be computed, defaults to end-effector (set in config)
        :type body_id: int, optional
        :param joint_indices: list of joint indices, defaults to all movable joints. Final jacobian will be of 
            shape 6 x len(joint_indices)
        :type joint_indices: [int], optional
        :return: 6xN body jacobian
        :rtype: np.ndarray
        """
        if body_id is None:
            body_id = self._ee_idx
            if self._ee_is_a_site:
                return self.site_jacobian(body_id,joint_indices)

        if joint_indices is None:
            joint_indices = self._movable_joints
        
        jacp = self._sim.data.body_jacp[body_id,:].reshape(3,-1)
        jacr = self._sim.data.body_jacr[body_id,:].reshape(3,-1)

        return np.vstack([jacp[:,joint_indices],jacr[:,joint_indices]])

    def site_jacobian(self, site_id, joint_indices = None):
        """
        Return jacobian computed for a site defined in model

        :param site_id: index of the site
        :type site_id: int
        :param joint_indices: list of joint indices, defaults to all movable joints. Final jacobian will be of
            shape 6 x len(joint_indices)
        :type joint_indices: [int]
        :return: 6xN jacobian
        :rtype: np.ndarray
        """
        if joint_indices is None:
            joint_indices = self._movable_joints
        jacp = self._sim.data.site_jacp[site_id,:].reshape(3,-1)
        jacr = self._sim.data.site_jacr[site_id,:].reshape(3,-1)
        return np.vstack([jacp[:,joint_indices],jacr[:,joint_indices]])


    def ee_pose(self):
        """
        Return end-effector pose at current sim step

        :return: EE position (x,y,z), EE quaternion (w,x,y,z)
        :rtype: np.ndarray, np.ndarray
        """
        if self._ee_is_a_site:
            return self.site_pose(self._ee_idx)
        return self.body_pose(self._ee_idx)
    
    def body_pose(self, body_id):
        """
        Return pose of specified body at current sim step

        :return: position (x,y,z), quaternion (w,x,y,z)
        :rtype: np.ndarray, np.ndarray
        """
        return self._sim.data.body_xpos[body_id], self._sim.data.body_xquat[body_id]

    def site_pose(self, site_id):
        """
        Return pose of specified site at current sim step

        :return: position (x,y,z), quaternion (w,x,y,z)
        :rtype: np.ndarray, np.ndarray
        """
        return self._sim.data.site_xpos[site_id], quaternion.from_rotation_matrix(self._sim.data.site_xmat[site_id].reshape(3,3))

    def ee_velocity(self):
        """
        Return end-effector velocity at current sim step

        :return: EE linear velocity (x,y,z), EE angular velocity (x,y,z)
        :rtype: np.ndarray, np.ndarray
        """
        if self._ee_is_a_site:
            return self.site_velocity(self._ee_idx)
        return self.body_velocity(self._ee_idx)

    def body_velocity(self, body_id):
        """
        Return velocity of specified body at current sim step

        :return: linear velocity (x,y,z), angular velocity (x,y,z)
        :rtype: np.ndarray, np.ndarray
        """
        return self._sim.data.body_xvelp[body_id], self._sim.data.body_xvelr[body_id]

    def site_velocity(self, site_id):
        """
        Return velocity of specified site at current sim step

        :return: linear velocity (x,y,z), angular velocity (x,y,z)
        :rtype: np.ndarray, np.ndarray
        """
        return self._sim.data.site_xvelp[site_id], self._sim.data.site_xvelr[site_id]

    def get_movable_joints(self):
        """
        Return list of movable (actuated) joints in the given model

        :return: list of indices of controllable joints
        :rtype: [int] (size: self._nu)
        """
        trntype = self._model.actuator_trntype # transmission type (0 == joint)
        trnid = self._model.actuator_trnid # transmission id (get joint actuated)

        mvbl_jnts = []
        for i in range(trnid.shape[0]):
            if trntype[i] == 0 and trnid[i,0] not in mvbl_jnts:
                mvbl_jnts.append(trnid[i,0])
        
        return sorted(mvbl_jnts)

    def start_asynchronous_run(self, rate = 50):
        """
        Start a separate thread running the step simulation 
        for the robot. Rendering still has to be called in the 
        main thread.
        """
        def continuous_run():
            self._asynch_thread_active = True
            while self._asynch_thread_active:
                now = time.time()
                self.step(render=False)
                elapsed = time.time() - now
                sleep_time = (1./rate) - elapsed
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
        
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
        """
        The actual step function to forward the simulation

        :param render: flag to forward the renderer as well, defaults to True
        :type render: bool, optional
        """
        self._mutex.acquire()
        # pre-step computations and value assignments
        if self._enable_gravity_compensation:
            self._additional_force += self._sim.data.qfrc_bias
            self._sim.data.qfrc_applied[:] = self._additional_force
            
        self._sim.step()

        # post-step changes
        self._additional_force *= 0
        
        if render:
            self.render()

        self._mutex.release()

    def render(self):
        """
        Separate function to render the visualiser. Only required if using 
        asynchronous simulation.
        """
        if self._viewer is not None:
            self._viewer.render()
