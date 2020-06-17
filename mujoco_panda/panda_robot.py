import os
import mujoco_py as mjp
import numpy as np
from .mujoco_robot import MujocoRobot

MODEL_PATH = os.environ['MJ_PANDA_PATH'] + \
    '/mujoco_panda/models/franka_panda.xml'


class PandaArm(MujocoRobot):

    def __init__(self, model_path=MODEL_PATH, render=True, **kwargs):

        super(PandaArm, self).__init__(model_path, render=render, **kwargs)

        self._has_gripper = self.has_body(
            ['panda_hand', 'panda_leftfinger', 'panda_rightfinger'])

        self._logger.info("PandaArm: Robot instance initialised with{} gripper".format(
            "out" if not self._has_gripper else ""))

        if self._has_gripper:
            self.set_as_ee("panda_hand")
        else:
            self.set_as_ee("ee_site")

        self._group_actuator_joints()

        self._neutral_pose = [0., -0.785, 0, -2.356, 0, 1.571, 0.785]

    @property
    def has_gripper(self):
        return self._has_gripper

    def _group_actuator_joints(self):
        arm_joint_names = ["panda_joint{}".format(i) for i in range(1, 8)]

        self.actuated_arm_joints = []
        self.actuated_arm_joint_names = []

        self.unactuated_arm_joints = []
        self.unactuated_arm_joint_names = []

        for jnt in arm_joint_names:
            jnt_id = self._model.joint_name2id(jnt)
            if jnt_id in self._controllable_joints:
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
                if jnt_id in self._controllable_joints:
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

    def jacobian(self, body_id=None):
        """
        Return body jacobian of end-effector based on the arm joints.

        :param body_id: index of end-effector to be used, defaults to "panda_hand" or "ee_site"
        :type body_id: int, optional
        :return: 6x7 body jacobian
        :rtype: np.ndarray
        """
        return self.body_jacobian(body_id=body_id, joint_indices=self.actuated_arm_joints)

    def set_neutral_pose(self, hard = True):
        if hard:
            self.hard_set_joint_positions(self._neutral_pose)
        else:
            self.set_joint_positions(self._neutral_pose)

    def get_actuator_ids(self, joint_list):

        try:
            return np.asarray([self._joint_to_actuator_map[i] for i in joint_list])
        except KeyError as e:
            raise ValueError(
                "Joint {} does not have a valid actuator".format(e))

    def set_joint_commands(self, cmd, joints=None, *args, **kwargs):
        """
        Uses available actuator to control the specified joints. Automatically selects
        and controls the actuator for joints of the provided ids.

        :param cmd: commands
        :type cmd: np.ndarray or [float]
        :param joints: ids of joints to command, defaults to all controllable joints
        :type joints: [int], optional
        """
        if joints is None:
            joints = self._controllable_joints[:len(cmd)]

        assert len(cmd) == len(joints)

        act_ids = self.get_actuator_ids(joints)
        cmd = np.asarray(cmd)

        torque_ids = np.intersect1d(act_ids, self._torque_actuators)
        pos_ids = np.intersect1d(act_ids, self._pos_actuators)

        if len(torque_ids) > 0:
            self.set_joint_torques(cmd[torque_ids], torque_ids)

        if len(pos_ids) > 0:
            self.set_joint_positions(cmd[pos_ids], pos_ids)

    def set_joint_torques(self, cmd, joint_ids=None, compensate_gravity=True):
        """
        Set joint torques for direct torque control. Use for torque controlling.
        Works for joints whose actuators are of type 'motor'.

        ..note: For better performance, this method cancels all other dynamics 
        acting on the robot, and applies the provided cmd as joint force.

        @param cmd  : torque values
        @type cmd   : [float] * self._nu

        """
        cmd = np.asarray(cmd).flatten()
        joint_ids = np.r_[:cmd.shape[0]
                             ] if joint_ids is None else np.r_[joint_ids]

        act_ids = self.get_actuator_ids(joint_ids)

        torque_ids = np.intersect1d(act_ids, self._torque_actuators)

        assert cmd[torque_ids].shape[0] == torque_ids.shape[0]

        # Cancel other dynamics
        acc_des = np.zeros(self._sim.model.nv)
        acc_des[torque_ids] = cmd[torque_ids]
        self._sim.data.qacc[:] = acc_des
        mjp.functions.mj_inverse(self._model, self._sim.data)
        cmd = self._sim.data.qfrc_inverse[torque_ids].copy()

        if len(torque_ids) > 0:
            self.set_actuator_ctrl(cmd[torque_ids], torque_ids)

    def set_joint_positions(self, cmd, joint_ids = None):
        """
        Set joint positions for position control.
        Works for joints whose actuators are of type 'motor'.

        ..note: For better performance, this method cancels all other dynamics 
        acting on the robot, and applies the provided cmd as joint force.

        @param cmd  : torque values
        @type cmd   : [float] * self._nu

        """
        cmd = np.asarray(cmd).flatten()
        joint_ids = np.r_[:cmd.shape[0]
                          ] if joint_ids is None else np.r_[joint_ids]

        act_ids = self.get_actuator_ids(joint_ids)

        pos_ids = np.intersect1d(act_ids, self._torque_actuators)

        assert cmd[pos_ids].shape[0] == pos_ids.shape[0]

        if len(pos_ids) > 0:
            self.set_actuator_ctrl(cmd[pos_ids], pos_ids)

