import os
import numpy as np
import mujoco_py as mjp
from collections import deque
from .mujoco_robot import MujocoRobot
from .gravity_robot import GravityRobot

MODEL_PATH = os.environ['MJ_PANDA_PATH'] + \
    '/mujoco_panda/models/franka_panda.xml'

DEFAULT_CONFIG = {
    'ft_site_name': 'ee_site'
}


class PandaArm(MujocoRobot):
    """
        Contructor

        :param model_path: path to model xml file for Panda, defaults to MODEL_PATH
        :type model_path: str, optional
        :param render: if set to True, will create a visualiser instance, defaults to True
        :type render: bool, optional
        :param config: see :py:class:`MujocoRobot`, defaults to DEFAULT_CONFIG
        :type config: dict, optional
        :param compensate_gravity: if set to True, will perform gravity compensation. Only required when using torque actuators
            and gravity is enabled. Defaults to True.
        :type compensate_gravity: bool, optional
        :param grav_comp_model_path: path to xml file for gravity compensation robot, defaults to :param:`model_path`.
        :type grav_comp_model_path: str, optional
        :param smooth_ft_sensor: (experimental) if set to True, :py:meth:`MujocoRobot.get_ft_reading` returns
            smoothed (low-pass-filter) force torque value **in world frame**. Defaults to False
        :type smooth_ft_sensor: bool, optional
    """

    def __init__(self, model_path=MODEL_PATH, render=True, config=DEFAULT_CONFIG, compensate_gravity=True, grav_comp_model_path=None, smooth_ft_sensor=False, **kwargs):

        super(PandaArm, self).__init__(model_path,
                                       render=render, config=config, **kwargs)

        self._compensate_gravity = compensate_gravity

        self._grav_comp_robot = None
        if self._compensate_gravity:
            grav_comp_model_path = grav_comp_model_path if grav_comp_model_path is not None else model_path
            self._grav_comp_robot = GravityRobot(grav_comp_model_path)

            assert self._grav_comp_robot.model.nv == self._model.nv

            self.add_pre_step_callable(
                {'grav_comp': [self._grav_compensator_handle, {}]})

            def _resetter():
                self._ignore_grav_comp = False
            self.add_post_step_callable({'grav_resetter': [_resetter, {}]})

        self._has_gripper = self.has_body(
            ['panda_hand', 'panda_leftfinger', 'panda_rightfinger'])

        self._logger.info("PandaArm: Robot instance initialised with{} gripper".format(
            "out" if not self._has_gripper else ""))

        if self._has_gripper:
            self.set_as_ee("panda_hand")
        else:
            self.set_as_ee("ee_site")

        self._group_actuator_joints()  # identifies position and torque actuators

        self._neutral_pose = [0., -0.785, 0, -2.356, 0, 1.571, 0.785]
        self._ignore_grav_comp = False

        self._smooth_ft = smooth_ft_sensor

        if self._smooth_ft:

            # change length of smoothing buffer if required
            self._smooth_ft_buffer = deque(maxlen=50)
            self.add_pre_step_callable(
                {'ft_smoother': [self._smoother_handle, [True]]})

    @property
    def has_gripper(self):
        """
        :return: True if gripper is present in model
        :rtype: bool
        """
        return self._has_gripper

    def _group_actuator_joints(self):
        arm_joint_names = ["panda_joint{}".format(i) for i in range(1, 8)]

        self.actuated_arm_joints = []
        self.actuated_arm_joint_names = []

        self.unactuated_arm_joints = []
        self.unactuated_arm_joint_names = []

        for jnt in arm_joint_names:
            jnt_id = self._model.joint_name2id(jnt)
            if jnt_id in self.controllable_joints:
                self.actuated_arm_joints.append(jnt_id)
                self.actuated_arm_joint_names.append(jnt)
            else:
                self.unactuated_arm_joints.append(jnt_id)
                self.unactuated_arm_joint_names.append(jnt)

        self._logger.debug("Movable arm joints: {}".format(
            str(self.actuated_arm_joint_names)))

        if len(self.unactuated_arm_joint_names) > 0:
            self._logger.info("These arm joints are not actuated: {}".format(
                str(self.unactuated_arm_joint_names)))

        self.actuated_gripper_joints = []
        self.actuated_gripper_joint_names = []

        self.unactuated_gripper_joints = []
        self.unactuated_gripper_joint_names = []

        if self._has_gripper:
            gripper_joint_names = [
                "panda_finger_joint1", "panda_finger_joint2"]

            for jnt in gripper_joint_names:
                jnt_id = self._model.joint_name2id(jnt)
                if jnt_id in self.controllable_joints:
                    self.actuated_gripper_joints.append(jnt_id)
                    self.actuated_gripper_joint_names.append(jnt)
                else:
                    self.unactuated_gripper_joints.append(jnt_id)
                    self.unactuated_gripper_joint_names.append(jnt)

            self._logger.debug("Movable gripper joints: {}".format(
                str(self.actuated_gripper_joint_names)))
            if len(self.unactuated_gripper_joint_names) > 0:
                self._logger.info("These gripper joints are not actuated: {}".format(
                    str(self.unactuated_gripper_joint_names)))

        self._pos_actuators, self._torque_actuators = self._identify_actuators(
            self.actuated_arm_joints+self.actuated_gripper_joints)

        self._logger.debug("Position actuators: {}\nTorque actuators: {}".format(
            str(self._pos_actuators), str(self._torque_actuators)))

    def _identify_actuators(self, joint_ids):

        pos_actuator_ids = []
        torque_actuator_ids = []

        self._joint_to_actuator_map = {}

        for idx, jnt in enumerate(self._model.actuator_trnid[:, 0].tolist()):
            assert jnt == joint_ids[idx]
            actuator_name = self._model.actuator_id2name(idx)
            controller_type = actuator_name.split('_')[1]
            if controller_type == 'torque' or controller_type == 'direct':
                torque_actuator_ids.append(idx)
                assert jnt not in self._joint_to_actuator_map, "Joint {} already has an actuator assigned!".format(
                    self._model.joint_id2name(jnt))
                self._joint_to_actuator_map[jnt] = idx
            elif controller_type == 'position' or controller_type == 'pos':
                pos_actuator_ids.append(idx)
                assert jnt not in self._joint_to_actuator_map, "Joint {} already has an actuator assigned!".format(
                    self._model.joint_id2name(jnt))
                self._joint_to_actuator_map[jnt] = idx
            else:
                self._logger.warn(
                    "Unknown actuator type for '{}'. Ignoring. This actuator will not be controllable via PandaArm api.".format(actuator_name))

        return pos_actuator_ids, torque_actuator_ids

    def get_ft_reading(self, pr=False, *args, **kwargs):
        """
        Overriding the parent class method for FT smoothing. FT smoothing has to be
        enabled while initialising PandaArm instance.

        :return: force, torque
        :rtype: np.ndarray, np.ndarray
        """
        if not self._smooth_ft:
            return super(PandaArm, self).get_ft_reading(*args, **kwargs)
        else:
            vals = np.mean(np.asarray(self._smooth_ft_buffer), 0)
            return vals[:3].copy(), vals[3:].copy()

    def jacobian(self, body_id=None):
        """
        Return body jacobian of end-effector based on the arm joints.

        :param body_id: index of end-effector to be used, defaults to "panda_hand" or "ee_site"
        :type body_id: int, optional
        :return: 6x7 body jacobian
        :rtype: np.ndarray
        """
        return self.body_jacobian(body_id=body_id, joint_indices=self.actuated_arm_joints)

    def set_neutral_pose(self, hard=True):
        """
        Set robot to neutral pose (tuck pose)

        :param hard: if False, uses position control (requires position actuators in joints). Defaults to True,
            hard set joint positions in simulator.
        :type hard: bool, optional
        """
        if hard:
            self.hard_set_joint_positions(self._neutral_pose)
        else:
            # position control, requires position actuators.
            self.set_joint_positions(self._neutral_pose)

    def get_actuator_ids(self, joint_list):
        """
        Get actuator ids for the provided list of joints

        :param joint_list: list of joint indices
        :type joint_list: [int]
        :raises ValueError: if joint does not have a valid actuator attached.
        :return: list of actuator ids
        :rtype: np.ndarray
        """
        try:
            return np.asarray([self._joint_to_actuator_map[i] for i in joint_list])
        except KeyError as e:
            raise ValueError(
                "Joint {} does not have a valid actuator".format(e))

    def set_joint_commands(self, cmd=None, joints=None, *args, **kwargs):
        """
        Uses available actuator to control the specified joints. Automatically selects
        and controls the actuator for joints of the provided ids.

        :param cmd: joint commands
        :type cmd: np.ndarray or [float]
        :param joints: ids of joints to command, defaults to all controllable joints
        :type joints: [int], optional
        """
        if cmd is None:
            self._ignore_grav_comp = False
            return

        if joints is None:
            joints = self.actuated_arm_joints[:len(cmd)]

        act_ids = self.get_actuator_ids(joints)
        cmd = np.asarray(cmd)

        torque_ids = np.intersect1d(act_ids, self._torque_actuators)
        pos_ids = np.intersect1d(act_ids, self._pos_actuators)

        if len(torque_ids) > 0:
            self.set_joint_torques(
                cmd[torque_ids], torque_ids, *args, **kwargs)

        if len(pos_ids) > 0:
            self.set_joint_positions(cmd[pos_ids], pos_ids, *args, **kwargs)

    def set_joint_torques(self, cmd, joint_ids=None, compensate_dynamics=False):
        """
        Set joint torques for direct torque control. Use for torque controlling.
        Works for joints whose actuators are of type 'motor'.

        :param cmd: torque values
        :type cmd: [float] (size: self._nu)
        :param joint_ids: ids of joints to control
        :type joint_ids: [int]
        :param compensate_dynamics: if set to True, compensates for external dynamics using inverse
            dynamics computation. Not recommended when performing contact tasks.
        :type compensate_dynamics: bool
        """

        cmd = np.asarray(cmd).flatten()
        joint_ids = np.r_[:cmd.shape[0]
                          ] if joint_ids is None else np.r_[joint_ids]

        act_ids = self.get_actuator_ids(joint_ids)

        torque_ids = np.intersect1d(act_ids, self._torque_actuators)

        assert cmd[torque_ids].shape[0] == torque_ids.shape[0]

        # Cancel other dynamics
        if compensate_dynamics:
            self._ignore_grav_comp = True
            acc_des = np.zeros(self._sim.model.nv)
            acc_des[torque_ids] = cmd[torque_ids]
            self._sim.data.qacc[:] = acc_des
            mjp.functions.mj_inverse(self._model, self._sim.data)
            cmd = self._sim.data.qfrc_inverse[torque_ids].copy()
        elif not compensate_dynamics and self._compensate_gravity:
            self._ignore_grav_comp = True
            cmd[torque_ids] += self.gravity_compensation_torques()[torque_ids]

        if len(torque_ids) > 0:
            self.set_actuator_ctrl(cmd[torque_ids], torque_ids)

    def set_joint_positions(self, cmd=None, joint_ids=None, *args, **kwargs):
        """
        Set joint positions for position control.
        Works for joints whose actuators are of type 'motor'.

        :param cmd: torque values
        :type cmd: [float] (size: self._nu)
        :param joint_ids: ids of joints to control
        :type joint_ids: [int]
        """
        self._ignore_grav_comp = False
        if cmd is None:
            return

        cmd = np.asarray(cmd).flatten()
        joint_ids = np.r_[:cmd.shape[0]
                          ] if joint_ids is None else np.r_[joint_ids]

        act_ids = self.get_actuator_ids(joint_ids)

        pos_ids = np.intersect1d(act_ids, self._pos_actuators)

        assert cmd[pos_ids].shape[0] == pos_ids.shape[0]

        if len(pos_ids) > 0:
            self.set_actuator_ctrl(cmd[pos_ids], pos_ids)

    def gravity_compensation_torques(self):
        """
        Get the gravity compensation torque at current sim timestep

        :return: gravity compensation torques in all movable joints
        :rtype: np.ndarray
        """
        self._grav_comp_robot.sim.data.qpos[self._grav_comp_robot._all_joints] = self._sim.data.qpos[self.qpos_joints].copy(
        )
        self._grav_comp_robot.sim.data.qvel[self._grav_comp_robot._controllable_joints] = 0.
        self._grav_comp_robot.sim.data.qacc[self._grav_comp_robot._controllable_joints] = 0.
        mjp.functions.mj_inverse(
            self._grav_comp_robot.model, self._grav_comp_robot.sim.data)

        return self._grav_comp_robot.sim.data.qfrc_inverse.copy()

    def _grav_compensator_handle(self):
        if self._compensate_gravity and not self._ignore_grav_comp:
            self._sim.data.ctrl[self._torque_actuators] = self.gravity_compensation_torques()[
                self._torque_actuators]

    def _smoother_handle(self, *args, **kwargs):
        self._smooth_ft_buffer.append(
            np.append(*(super(self).get_ft_reading(*args, **kwargs))))

    @classmethod
    def fullRobotWithTorqueActuators(cls, **kwargs):
        """
        Create an instance of this class using the model of the full
        robot (arm + gripper) and torque actuators at arm joints.
        """
        model_path = os.environ['MJ_PANDA_PATH'] + \
            '/mujoco_panda/models/franka_panda.xml'
        return cls(model_path=model_path, **kwargs)

    @classmethod
    def fullRobotWithPositionActuators(cls, **kwargs):
        """
        Create an instance of this class using the model of the full
        robot (arm + gripper) and position actuators at arm joints.
        """
        model_path = os.environ['MJ_PANDA_PATH'] + \
            '/mujoco_panda/models/franka_panda_pos.xml'
        return cls(model_path=model_path, **kwargs)

    @classmethod
    def withTorqueActuators(cls, **kwargs):
        """
        Create an instance of this class using the model of the
        robot arm without gripper, and torque actuators at arm joints.
        """
        model_path = os.environ['MJ_PANDA_PATH'] + \
            '/mujoco_panda/models/franka_panda_no_gripper.xml'
        return cls(model_path=model_path, **kwargs)

    @classmethod
    def withPositionActuators(cls, **kwargs):
        """
        Create an instance of this class using the model of the
        robot arm without gripper, and position actuators at arm joints.
        """
        model_path = os.environ['MJ_PANDA_PATH'] + \
            '/mujoco_panda/models/franka_panda_pos_no_gripper.xml'
        return cls(model_path=model_path, **kwargs)
