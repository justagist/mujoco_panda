from mujoco_panda.controllers.controller_base import ControllerBase
from mujoco_panda.utils.tf import quatdiff_in_euler
from .configs import BASIC_HYB_CONFIG
import numpy as np


class OSHybridForceMotionController(ControllerBase):

    def __init__(self, robot_object, config=BASIC_HYB_CONFIG, *args, **kwargs):

        super(OSHybridForceMotionController,
              self).__init__(robot_object, config)

        self._goal_pos, self._goal_ori = self._robot.ee_pose()
        self._goal_vel, self._goal_omg = np.zeros(3), np.zeros(3)
        self._goal_force, self._goal_torque = np.zeros(
            3), np.zeros(3)

        # gains for position
        self._kp_p = np.diag(self._config['kp_p'])
        self._kd_p = np.diag(self._config['kd_p'])

        # gains for orientation
        self._kp_o = np.diag(self._config['kp_o'])
        self._kd_o = np.diag(self._config['kd_o'])

        # gains for force
        self._kp_f = np.diag(self._config['kp_f'])
        self._kd_f = np.diag(self._config['kd_f'])

        # gains for torque
        self._kp_t = np.diag(self._config['kp_t'])
        self._kd_t = np.diag(self._config['kd_t'])

        self._force_dir = np.diag(self._config['ft_dir'][:3])
        self._torque_dir = np.diag(self._config['ft_dir'][3:])

        self._pos_p_dir = np.diag([1, 1, 1]) ^ self._force_dir
        self._pos_o_dir = np.diag([1, 1, 1]) ^ self._torque_dir

        self._use_null_ctrl = self._config['use_null_space_control']

        if self._use_null_ctrl:

            self._null_Kp = np.diag(self._config['null_kp'])

            self._null_ctrl_wt = self._config['null_ctrl_wt']

        self._pos_threshold = self._config['linear_error_thr']

        self._angular_threshold = self._config['angular_error_thr']

    def set_active(self, status=True):
        if status:
            self._goal_pos, self._goal_ori = self._robot.ee_pose()
            self._goal_vel, self._goal_omg = np.zeros(3), np.zeros(3)
            self._goal_force, self._goal_torque = np.zeros(
                3), np.zeros(3)
        self._is_active = status

    def _compute_cmd(self):
        # calculate the jacobian of the end effector
        jac_ee = self._robot.jacobian()

        # get position of the end-effector
        curr_pos, curr_ori = self._robot.ee_pose()
        curr_vel, curr_omg = self._robot.ee_velocity()

        curr_force, curr_torque = self._robot.get_ft_reading()

        delta_pos = self._goal_pos - curr_pos

        delta_vel = self._goal_vel - curr_vel

        delta_force = self._force_dir.dot(self._goal_force - curr_force)

        delta_torque = self._torque_dir.dot(self._goal_torque - curr_torque)

        # if np.linalg.norm(delta_pos) <= self._pos_threshold:
        #     delta_pos = np.zeros(delta_pos.shape)
        #     delta_vel = np.zeros(delta_pos.shape)
        # threshold = 0.01
        # print("Delta force \t", delta_force)

        if self._goal_ori is not None:
            delta_ori = quatdiff_in_euler(curr_ori, self._goal_ori)
        else:
            delta_ori = np.zeros(delta_pos.shape)

        tot_error = np.linalg.norm(
            delta_pos) + np.linalg.norm(delta_ori) + np.linalg.norm(delta_force) + np.linalg.norm(delta_torque)
        
        # if tot_error <= threshold:
        #     return self._cmd
        # print (tot_error)

        delta_ori = self._pos_o_dir.dot(delta_ori)

        delta_omg = self._pos_o_dir.dot(self._goal_omg - curr_omg)

        # force_control = self._force_dir.dot(self._kp_f.dot(delta_force) - np.abs(self._kd_f.dot(delta_vel)) + self._goal_force)

        # if np.linalg.norm(curr_force) < 3.:
        #     force_control = self._force_dir.dot(np.sign(self._goal_force)*5.)

        # else:
        force_control = self._force_dir.dot(self._kp_f.dot(delta_force) - np.abs(self._kd_f.dot(delta_vel)) + self._goal_force)
        # - (np.sign(delta_force)*np.abs(self._kd_f.dot(delta_vel)))
        # force_control = self._force_dir.dot(
        #     self._kp_f.dot(delta_force))

        # print("ctrl", force_control, delta_force, self._goal_force, curr_force)

        position_control = self._pos_p_dir.dot(
            self._kp_p.dot(delta_pos) + self._kd_p.dot(delta_vel))

        x_des = position_control + force_control

        if self._goal_ori is not None:

            if np.linalg.norm(delta_ori) < self._angular_threshold:
                delta_ori = np.zeros(delta_ori.shape)
                delta_omg = np.zeros(delta_omg.shape)

            # torque_control = self._kp_t.dot(delta_torque) - np.abs(self._kd_t.dot(delta_omg)) + self._torque_dir.dot(self._goal_torque)
            ori_pos_ctrl = self._pos_o_dir.dot(
                self._kp_o.dot(delta_ori) + self._kd_o.dot(delta_omg))

            torque_f_ctrl = self._torque_dir.dot(self._kp_t.dot(
                delta_torque) - self._kd_t.dot(delta_omg) + self._goal_torque)

            omg_des = ori_pos_ctrl + torque_f_ctrl

        else:

            omg_des = np.zeros(3)

        f_ee = np.hstack([x_des, omg_des])  # Desired end-effector force

        # print "delta_omg \t", delta_omg
        # print "delta_vel \t", delta_vel

        u = np.dot(jac_ee.T, f_ee)

        if np.any(np.isnan(u)):
            u = self._cmd
        else:
            self._cmd = u

        # print("Computed torque: ", x_des, omg_des, f_ee, self._cmd)

        if self._use_null_ctrl:

            null_space_filter = self._null_Kp.dot(
                np.eye(7) - jac_ee.T.dot(np.linalg.pinv(jac_ee.T, rcond=1e-3)))

            self._cmd = self._cmd + \
                null_space_filter.dot(
                    self._robot._neutral_pose-self._robot.joint_positions()[:7])

        # Never forget to update the error
        self._error = {'linear': delta_pos, 'angular': delta_ori}

        return self._cmd

    def set_goal(self, goal_pos, goal_ori=None, goal_vel=np.zeros(3), goal_omg=np.zeros(3), goal_force=None, goal_torque=None):
        self._mutex.acquire()
        self._goal_pos = goal_pos
        self._goal_ori = goal_ori
        self._goal_vel = goal_vel
        self._goal_omg = goal_omg
        if goal_force is not None:
            self._goal_force = goal_force
        if goal_torque is not None:
            self._goal_torque = goal_torque
        self._mutex.release()

    def change_ft_dir(self, directions):
        self._mutex.acquire()
        self._force_dir = np.diag(directions[:3])

        self._torque_dir = np.diag(directions[3:])

        self._pos_p_dir = np.diag([1, 1, 1]) ^ self._force_dir

        self._pos_o_dir = np.diag([1, 1, 1]) ^ self._torque_dir
        self._mutex.release()