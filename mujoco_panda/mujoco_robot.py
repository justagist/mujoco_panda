import copy
import time
import logging
import threading
import quaternion
import numpy as np
import mujoco_py as mjp
from threading import Lock

LOG_LEVEL = "DEBUG"

class ContactInfo(object):
    def __init__(self, pos, ori, ft):
        self.pos = pos.copy()
        self.ori_mat = ori.copy()
        self.quat = quaternion.as_float_array(quaternion.from_rotation_matrix(self.ori_mat))
        self.ft = ft.copy()
    
    def __str__(self):
        return "pos: {}, quat: {}, ft: {}".format(self.pos, self.quat, self.ft)

class MujocoRobot(object):
    """
        Constructor

        :param model_path: path to model xml file for robot
        :type model_path: str
        :param from_model: PyMjModel instance of robot. If provided, will ignore the `model_path` param.
        :type from_model: mjp.PyMjModel
        :param render: create a visualiser instance in mujoco; defaults to True.
        :type render: bool, optional
        :param prestep_callables: dictionary of callable iterms to be run before running
            sim.run(). Format: {'name_for_callable': [callable_handle, [list_of_arguments_for_callable]]}
        :type prestep_callables: {'str': [callable, [*args]]}
        :param poststep_callables: dictionary of callable iterms to be run after running
            sim.run(). Format: {'name_for_callable': [callable_handle, [list_of_arguments_for_callable]]}
        :type poststep_callables: {'str': [callable, [*args]]}
        :param config: optional values for setting end-effector and ft sensor locations. 
            Format: {'ee_name': "name_of_body_or_site_in_model",
                     'ft_site': "name_of_site_with_force_torque_sensor"}
        :type config: {str: str}
    """

    def __init__(self, model_path=None, render=True, config=None, prestep_callables={}, poststep_callables={}, from_model=False):

        logging.basicConfig(format='\n{}: %(levelname)s: %(message)s\n'.format(
            self.__class__.__name__), level=LOG_LEVEL)
        self._logger = logging.getLogger(__name__)

        if isinstance(from_model,mjp.cymj.PyMjModel):
            self._model = from_model
        else:
            self._model = mjp.load_model_from_path(model_path)

        self._sim = mjp.MjSim(self._model)
        self._viewer = mjp.MjViewer(self._sim) if render else None

        self._has_gripper = False  # by default, assume no gripper is attached

        # seprate joints that are controllable, movable, etc.
        self._define_joint_ids()

        self._nu = len(self.controllable_joints)  # number of actuators

        self._nq = len(self.qpos_joints)  # total number of joints

        self._all_joint_names = [
            self.model.joint_id2name(j) for j in self.qpos_joints]

        self._all_joint_dict = dict(
            zip(self._all_joint_names, self.qpos_joints))

        if 'ee_name' in config:
            self.set_as_ee(config['ee_name'])
        else:
            self._ee_idx, self._ee_name = self._use_last_defined_link()
            self._ee_is_a_site = False

        if 'ft_site_name' in config:
            self._ft_site_name = config['ft_site_name']
        else:
            self._ft_site_name = False

        self._mutex = Lock()
        self._asynch_thread_active = False


        self._forwarded = False

        self._pre_step_callables = prestep_callables
        self._post_step_callables = poststep_callables

        self._first_step_not_done = True

    def set_as_ee(self, body_name):
        """
        Set provided body or site as the end-effector of the robot.

        :param body_name: name of body or site in mujoco model
        :type body_name: str
        """
        self._ee_name = body_name

        if body_name in self._model.site_names:
            self._ee_is_a_site = True
            self._ee_idx = self._model.site_name2id(body_name)
            self._logger.debug(
                "End-effector is a site in model: {}".format(body_name))
        else:
            self._ee_is_a_site = False
            self._ee_idx = self._model.body_name2id(self._ee_name)

    def _use_last_defined_link(self):
        return self._model.nbody-1, self._model.body_id2name(self._model.nbody-1)

    def add_pre_step_callable(self, f_dict):
        """
        Add new values to prestep_callable dictionary. See contructor params for details.

        :param f_dict: {"name_of_callable": [callable_handle,[list_of_arguments_for_callable]]}
        :type f_dict: {str:[callable,[*args]]}
        """
        for key in list(f_dict.keys()):
            if key not in self._pre_step_callables:
                self._pre_step_callables.update(f_dict)

    def add_post_step_callable(self, f_dict):
        """
        Add new values to poststep_callable dictionary. See contructor params for details.

        :param f_dict: {"name_of_callable": [callable_handle,[list_of_arguments_for_callable]]}
        :type f_dict: {str:[callable,[*args]]}
        """
        for key in list(f_dict.keys()):
            if key not in self._post_step_callables:
                self._post_step_callables.update(f_dict)

    @property
    def sim(self):
        """
        Get mujoco_py.sim object associated with this instance

        :return: mujoco_py.sim object associated with this instance
        :rtype: mujoco_py.MjSim
        """
        return self._sim

    @property
    def viewer(self):
        """
        :return: mujoco_py.MjViewer object associated with this instance
        :rtype: mujoco_py.MjViewer
        """
        return self._viewer

    @property
    def model(self):
        """
        :return: mujoco_py.cymj.PyMjModel object associated with this instance
        :rtype: mujoco_py.cymj.PyMjModel
        """
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

    def get_ft_reading(self, in_global_frame=True):
        """
        Return sensordata values. Assumes no other sensor is present.

        :return: Force torque sensor readings from the defined force torque sensor.
        :rtype: np.ndarray (3,), np.ndarray (3,)
        """
        if self._model.sensor_type[0] == 4 and self._model.sensor_type[1] == 5:
            if not self._forwarded:
                self.forward_sim()
            sensordata = -self._sim.data.sensordata.copy() # change sign to make force relative to parent body
            if in_global_frame:
                if self._ft_site_name:
                    new_sensordata = np.zeros(6)
                    _, site_ori = self.site_pose(
                        self._model.site_name2id(self._ft_site_name))
                    rotation_mat = quaternion.as_rotation_matrix(
                        np.quaternion(*site_ori))
                    new_sensordata[:3] = np.dot(
                        rotation_mat, np.asarray(sensordata[:3]))
                    new_sensordata[3:] = np.dot(
                        rotation_mat, np.asarray(sensordata[3:]))
                    sensordata = new_sensordata.copy()
                else:
                    self._logger.warning("Could not transform ft sensor values. Please\
                        provide ft site name in config file.")
            return sensordata[:3], sensordata[3:]
        else:
            self._logger.debug(
                "Could not find FT sensor as the first sensor in model!")
            return np.zeros(6)

    def get_contact_info(self):
        """
        Get details about physical contacts between bodies.
        Includes contact point positions, orientations, contact forces.

        :return: list of ContactInfo objects
        :rtype: [ContactInfo]
        """
        self._mutex.acquire()
        mjp.functions.mj_rnePostConstraint(
            self._sim.model, self._sim.data)

        nc = self._sim.data.ncon

        c_array = np.zeros(6, dtype=np.float64)
        contact_list = []
        
        for i in range(nc):
            contact = self._sim.data.contact[i]
            c_array.fill(0)
            mjp.functions.mj_contactForce(
                self._sim.model, self._sim.data, i, c_array)

            ori = np.flip(contact.frame.reshape(3, 3).T, 1)

            # # for mujoco the normal force is along x
            # # so for the convention we flip X and Z
            # c_array = np.hstack([np.flip(c_array[:3]),
            #                      np.flip(c_array[3:])])

            contact_list.append(copy.deepcopy(ContactInfo(
                contact.pos.copy(), ori.copy(), c_array.copy())))

        assert nc == len(contact_list)
        
        self._mutex.release()

        return contact_list

    def body_jacobian(self, body_id=None, joint_indices=None, recompute=True):
        """
        return body jacobian at current step

        :param body_id: id of body whose jacobian is to be computed, defaults to end-effector (set in config)
        :type body_id: int, optional
        :param joint_indices: list of joint indices, defaults to all movable joints. Final jacobian will be of 
            shape 6 x len(joint_indices)
        :type joint_indices: [int], optional
        :param recompute: if set to True, will perform forward kinematics computation for the step and provide updated
            results; defaults to True
        :type recompute: bool
        :return: 6xN body jacobian
        :rtype: np.ndarray
        """
        if body_id is None:
            body_id = self._ee_idx
            if self._ee_is_a_site:
                return self.site_jacobian(body_id, joint_indices)

        if joint_indices is None:
            joint_indices = self.controllable_joints

        if recompute and not self._forwarded:
            self.forward_sim()

        jacp = self._sim.data.body_jacp[body_id, :].reshape(3, -1)
        jacr = self._sim.data.body_jacr[body_id, :].reshape(3, -1)

        return np.vstack([jacp[:, joint_indices], jacr[:, joint_indices]])

    def site_jacobian(self, site_id, joint_indices=None, recompute=True):
        """
        Return jacobian computed for a site defined in model

        :param site_id: index of the site
        :type site_id: int
        :param joint_indices: list of joint indices, defaults to all movable joints. Final jacobian will be of
            shape 6 x len(joint_indices)
        :type joint_indices: [int]
        :param recompute: if set to True, will perform forward kinematics computation for the step and provide updated
            results; defaults to True
        :type recompute: bool
        :return: 6xN jacobian
        :rtype: np.ndarray
        """
        if joint_indices is None:
            joint_indices = self.controllable_joints

        if recompute and not self._forwarded:
            self.forward_sim()

        jacp = self._sim.data.site_jacp[site_id, :].reshape(3, -1)
        jacr = self._sim.data.site_jacr[site_id, :].reshape(3, -1)
        return np.vstack([jacp[:, joint_indices], jacr[:, joint_indices]])

    def ee_pose(self):
        """
        Return end-effector pose at current sim step

        :return: EE position (x,y,z), EE quaternion (w,x,y,z)
        :rtype: np.ndarray, np.ndarray
        """
        if self._ee_is_a_site:
            return self.site_pose(self._ee_idx)
        return self.body_pose(self._ee_idx)

    def body_pose(self, body_id, recompute=True):
        """
        Return pose of specified body at current sim step

        :param body_id: id or name of the body whose world pose has to be obtained.
        :type body_id: int / str
        :param recompute: if set to True, will perform forward kinematics computation for the step and provide updated
            results; defaults to True
        :type recompute: bool
        :return: position (x,y,z), quaternion (w,x,y,z)
        :rtype: np.ndarray, np.ndarray
        """
        if isinstance(body_id, str):
            body_id = self._model.body_name2id(body_id)
        if recompute and not self._forwarded:
            self.forward_sim()
        return self._sim.data.body_xpos[body_id].copy(), self._sim.data.body_xquat[body_id].copy()

    def site_pose(self, site_id, recompute=True):
        """
        Return pose of specified site at current sim step

        :param site_id: id or name of the site whose world pose has to be obtained.
        :type site_id: int / str
        :param recompute: if set to True, will perform forward kinematics computation for the step and provide updated
            results; defaults to True
        :type recompute: bool
        :return: position (x,y,z), quaternion (w,x,y,z)
        :rtype: np.ndarray, np.ndarray
        """
        if isinstance(site_id, str):
            site_id = self._model.site_name2id(site_id)
        if recompute and not self._forwarded:
            self.forward_sim()
        return self._sim.data.site_xpos[site_id].copy(), quaternion.as_float_array(quaternion.from_rotation_matrix(self._sim.data.site_xmat[site_id].copy().reshape(3, 3)))

    def ee_velocity(self):
        """
        Return end-effector velocity at current sim step

        :return: EE linear velocity (x,y,z), EE angular velocity (x,y,z)
        :rtype: np.ndarray, np.ndarray
        """
        if self._ee_is_a_site:
            return self.site_velocity(self._ee_idx)
        return self.body_velocity(self._ee_idx)

    def body_velocity(self, body_id, recompute=True):
        """
        Return velocity of specified body at current sim step

        :param body_id: id or name of the body whose cartesian velocity has to be obtained.
        :type body_id: int / str
        :param recompute: if set to True, will perform forward kinematics computation for the step and provide updated
            results; defaults to True
        :type recompute: bool
        :return: linear velocity (x,y,z), angular velocity (x,y,z)
        :rtype: np.ndarray, np.ndarray
        """
        if recompute and not self._forwarded:
            self.forward_sim()
        return self._sim.data.body_xvelp[body_id].copy(), self._sim.data.body_xvelr[body_id].copy()

    def site_velocity(self, site_id, recompute=True):
        """
        Return velocity of specified site at current sim step

        :param site_id: id or name of the site whose cartesian velocity has to be obtained.
        :type site_id: int / str
        :param recompute: if set to True, will perform forward kinematics computation for the step and provide updated
            results; defaults to True
        :type recompute: bool
        :return: linear velocity (x,y,z), angular velocity (x,y,z)
        :rtype: np.ndarray, np.ndarray
        """
        if recompute and not self._forwarded:
            self.forward_sim()
        return self._sim.data.site_xvelp[site_id].copy(), self._sim.data.site_xvelr[site_id].copy()

    def _define_joint_ids(self):
        # transmission type (0 == joint)
        trntype = self._model.actuator_trntype
        # transmission id (get joint actuated)
        trnid = self._model.actuator_trnid

        ctrl_joints = []
        for i in range(trnid.shape[0]):
            if trntype[i] == 0 and trnid[i, 0] not in ctrl_joints:
                ctrl_joints.append(trnid[i, 0])

        self.controllable_joints = sorted(ctrl_joints)
        self.movable_joints = self._model.jnt_dofadr
        self.qpos_joints = self._model.jnt_qposadr

    def start_asynchronous_run(self, rate=None):
        """
        Start a separate thread running the step simulation 
        for the robot. Rendering still has to be called in the 
        main thread.

        :param rate: rate of thread loop, defaults to the simulation timestep defined
            in the model
        :type rate: float
        """
        if rate is None:
            rate = 1.0/self._model.opt.timestep

        def continuous_run():
            self._asynch_thread_active = True
            while self._asynch_thread_active:
                now = time.time()
                self.step(render=False)
                elapsed = time.time() - now
                sleep_time = (1./rate) - elapsed
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

        self._asynch_sim_thread = threading.Thread(target=continuous_run)
        self._asynch_sim_thread.start()

    def joint_positions(self, joints=None):
        """
        Get positions of robot joints

        :param joints: list of joints whose positions are to be obtained, defaults to all present joints
        :type joints: [int], optional
        :return: joint positions
        :rtype: np.ndarray
        """
        if joints is None:
            joints = self.qpos_joints
        return self._sim.data.qpos[joints]

    def joint_velocities(self, joints=None):
        """
        Get velocities of robot joints

        :param joints: list of joints whose velocities are to be obtained, defaults to all present joints
        :type joints: [int], optional
        :return: joint velocities
        :rtype: np.ndarray
        """
        if joints is None:
            joints = self.movable_joints
        return self._sim.data.qvel[joints]

    def joint_accelerations(self, joints=None):
        """
        Get accelerations of robot joints

        :param joints: list of joints whose accelerations are to be obtained, defaults to all present joints
        :type joints: [int], optional
        :return: joint accelerations
        :rtype: np.ndarray
        """
        if joints is None:
            joints = self.movable_joints
        return self._sim.data.qacc[joints]

    def stop_asynchronous_run(self):
        """
        Stop asynchronous run thread. See :func:`start_asynchronous_run`.
        """
        self._asynch_thread_active = False
        self._asynch_sim_thread.join()

    def set_actuator_ctrl(self, cmd, actuator_ids=None):
        """
        Set controller values. Each cmd should be an appropriate
        value for the controller type of actuator specified.

        @param cmd  : actuator ctrl values
        @type cmd   : [float] * self._nu
        """
        cmd = np.asarray(cmd).flatten()
        actuator_ids = np.r_[:cmd.shape[0]
                             ] if actuator_ids is None else np.r_[actuator_ids]

        assert cmd.shape[0] == actuator_ids.shape[0]

        self._sim.data.ctrl[actuator_ids] = cmd

    def hard_set_joint_positions(self, values, indices=None):
        """
        Hard set robot joints to the specified values. Used mainly when
        using torque actuators in model.

        :param values: joint positions
        :type values: np.ndarray / [float]
        :param indices: indices of joints to use, defaults to the first :math:`n` joints in model,
            where :math:`n` is the number of values in :param:`joints`.
        :type indices: [int], optional
        """
        if indices is None:
            indices = range(np.asarray(values).shape[0])

        self._sim.data.qpos[indices] = values

    def __del__(self):
        if hasattr(self, '_asynch_sim_thread') and self._asynch_sim_thread.isAlive():
            self._asynch_thread_active = False
            self._asynch_sim_thread.join()

    def step(self, render=True):
        """
        The actual step function to forward the simulation. Not required or recommended if 
        using asynchronous run.

        :param render: flag to forward the renderer as well, defaults to True
        :type render: bool, optional
        """

        for f_id in self._pre_step_callables:
            self._pre_step_callables[f_id][0](
                *self._pre_step_callables[f_id][1])


        self._sim.step()

        # if self._first_step_not_done:
        #     self._first_step_not_done = False
        #     self._sim.data.sensordata[:] = np.zeros_like(self._sim.data.sensordata)

        self._forwarded = False

        for f_id in self._post_step_callables:
            self._post_step_callables[f_id][0](
                *self._post_step_callables[f_id][1])

        if render:
            self.render()

    def forward_sim(self):
        """
        Perform no-motion forward simulation without integrating over time, i.e. sim step is 
        not increased.
        """
        self._sim.forward()
        self._forwarded = True

    def render(self):
        """
        Separate function to render the visualiser. Required for visualising in main thread if using 
        asynchronous simulation.
        """
        if self._viewer is not None:
            self._viewer.render()
